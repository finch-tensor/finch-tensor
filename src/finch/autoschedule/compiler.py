from __future__ import annotations

import logging
from collections.abc import Iterable

from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import (
    FinchOperator,
    FType,
    ffuncs,
    ftypes,
    is_static,
)
from finch.algebra.tensor import TensorFType
from finch.compile.lower import make_extent
from finch.finch_assembly import AssemblyKernelFType, AssemblyLibrary
from finch.finch_logic import (
    Alias,
    LogicLoader,
    StatsFactory,
    TensorStats,
    compute_shape_vars,
)
from finch.finch_notation import NotationInterpreter
from finch.finch_notation.stages import NotationLoader
from finch.symbolic import gensym
from finch.symbolic.traversal import PostOrderDFS
from finch.util.logging import LOG_NOTATION

from .stages import CompilerForm, FormattedForm, LogicNotationLowerer

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_NOTATION)


class PointwiseContext:
    def __init__(self, ctx: NotationContext):
        self.ctx = ctx

    def __call__(
        self,
        ex: lgc.LogicExpression,
        loops: dict[lgc.Field, ntn.Variable],
    ) -> ntn.NotationExpression:
        match ex:
            case lgc.MapJoin(lgc.Literal(op), args):
                return ntn.Call(
                    ntn.Literal(op),
                    tuple(
                        self(arg, {idx: loops[idx] for idx in arg.fields()})
                        for arg in args
                    ),
                )
            case lgc.Table(lgc.Alias() as var, idxs):
                return ntn.Unwrap(
                    ntn.Access(
                        self.ctx.slots[var.unfused],
                        ntn.Read(),
                        tuple(loops[idx] for idx in idxs),
                    )
                )
            case lgc.Literal(val):
                return ntn.Literal(val)
            case lgc.FillTable(lgc.Alias() as var):
                return ntn.FillValue(self.ctx.slots[var.unfused])
            case lgc.Relabel(arg, idxs):
                return self(
                    arg,
                    {
                        idx_1: loops[idx_2]
                        for idx_1, idx_2 in zip(arg.fields(), idxs, strict=True)
                    },
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def merge_shapes(a: ntn.Variable | None, b: ntn.Variable | None) -> ntn.Variable | None:
    if a and b:
        if a.name < b.name:
            return a
        return b
    return a or b


class NotationContext:
    """
    Compiles Finch Logic to Finch Notation. Holds the state of the
    compilation process.
    """

    def __init__(
        self,
        bindings: dict[lgc.Alias, TensorFType],
        args: dict[lgc.Alias, ntn.Variable],
        slots: dict[lgc.Alias, ntn.Slot],
        shapes: dict[lgc.Alias, tuple[ntn.Variable | None, ...]],
        shape_types: dict[lgc.Alias, tuple[FType | None, ...]] | None = None,
        epilogue: Iterable[ntn.NotationStatement] | None = None,
    ):
        self.bindings = bindings
        self.args = args
        self.slots = slots
        self.shapes = shapes
        self.equiv: dict[ntn.Variable, ntn.Variable] = {}
        if shape_types is None:
            shape_types = {var: val.shape_type for var, val in bindings.items()}
        self.shape_types = shape_types
        if epilogue is None:
            epilogue = ()
        self.epilogue = epilogue
        # The mode of each tensor at the current statement: frozen tensors are
        # Read, and thawed tensors are Updated with the op they were thawed
        # with.
        self.modes: dict[lgc.Alias, ntn.AccessMode] = {
            var: ntn.Read() for var in bindings
        }

    def freeze(self, var: lgc.Alias) -> tuple[ntn.NotationStatement, ...]:
        """Statements which leave `var` frozen, so that it can be read."""
        match self.modes[var]:
            case ntn.Update(op):
                self.modes[var] = ntn.Read()
                return (ntn.Freeze(self.slots[var], op),)
        return ()

    def thaw(
        self, var: lgc.Alias, op: ntn.Literal
    ) -> tuple[ntn.NotationStatement, ...]:
        """Statements which leave `var` thawed, so that it can be updated with
        `op`."""
        if self.modes[var] == ntn.Update(op):
            return ()
        stmts = self.freeze(var)
        self.modes[var] = ntn.Update(op)
        return (*stmts, ntn.Thaw(self.slots[var], op))

    def declare(
        self, var: lgc.Alias, init: ntn.NotationExpression, op: ntn.Literal
    ) -> tuple[ntn.NotationStatement, ...]:
        """Statements which reset `var` to `init`, leaving it thawed, so that it
        can be updated with `op`."""
        stmts = self.freeze(var)
        self.modes[var] = ntn.Update(op)
        return (*stmts, ntn.Declare(self.slots[var], init, op, ()))

    def lower_init(self, init: lgc.LogicExpression) -> ntn.NotationExpression:
        """The value of `init`, a zero-dimensional expression. A fill expression
        whose fill is static is a literal, so that kernels may specialize on it."""
        if lgc.is_fill_expression(init):
            fill = init.fill_type(
                {var: val.fill_type for var, val in self.bindings.items()}
            )
            if is_static(fill):
                return ntn.Literal(fill.value)
        return PointwiseContext(self)(init, {})

    def freeze_reads(self, expr: lgc.LogicNode) -> tuple[ntn.NotationStatement, ...]:
        """Statements which leave every tensor that `expr` reads frozen."""
        reads = dict.fromkeys(
            node.unfused for node in PostOrderDFS(expr) if isinstance(node, lgc.Alias)
        )
        return tuple(stmt for var in reads for stmt in self.freeze(var))

    def overwrite(
        self, var: lgc.Alias, idxs: tuple[lgc.Field, ...], rhs: lgc.Reorder
    ) -> ntn.NotationStatement:
        """Overwrite `var` with `rhs`, reducing the fields `var` lacks."""
        # Every element of var is written, so var is reset to its fill value
        # first. An empty reduction writes nothing and leaves var as it was, so
        # var is only reset when every reduced extent is nonzero.
        arg_dims = rhs.arg.dimmap(merge_shapes, self.shapes)
        dims = dict(zip(rhs.arg.fields(), arg_dims, strict=True))
        reduced = [dims.get(idx) for idx in rhs.idxs if idx not in idxs]
        nonempty = tuple(
            ntn.Call(ntn.Literal(ffuncs.gt), (dim, ntn.Literal(dim.result_type(0))))
            for dim in reduced
            if dim is not None
        )
        # An overwrite keeps the static fill of var, and otherwise gives var the
        # fill of rhs.
        fill = self.bindings[var].fill_type
        init = (
            ntn.Literal(fill.value)
            if is_static(fill)
            else self.lower_init(lgc.fill_expression(rhs))
        )
        if not nonempty:
            return self.fold(var, idxs, ffuncs.overwrite, rhs, init)
        # Each branch must leave every tensor in the same mode.
        stmts = (*self.freeze_reads(rhs), *self.freeze(var))
        body = self.fold(var, idxs, ffuncs.overwrite, rhs, init)
        cond = nonempty[0]
        if len(nonempty) > 1:
            cond = ntn.Call(ntn.Literal(ffuncs.and_), nonempty)
        return ntn.Block((*stmts, ntn.If(cond, ntn.Block((body, *self.freeze(var))))))

    def fold(
        self,
        var: lgc.Alias,
        idxs: tuple[lgc.Field, ...],
        op: FinchOperator,
        rhs: lgc.Reorder,
        init: ntn.NotationExpression | None,
    ) -> ntn.NotationStatement:
        """Fold `rhs` into `var` with `op`, after resetting `var` to `init`
        unless it is None."""
        stmts = self.freeze_reads(rhs)
        if init is None:
            stmts += self.thaw(var, ntn.Literal(op))
        else:
            reduced = any(idx not in idxs for idx in rhs.fields())
            match init:
                case ntn.Literal(val) if op == ffuncs.overwrite and not reduced:
                    op = ffuncs.init_write(val)
            stmts += self.declare(var, init, ntn.Literal(op))
        match rhs:
            case lgc.Reorder(lgc.Table() as arg, loop_order):
                body = self._lower_query_of_reorder(var, op, arg, idxs, loop_order)
            case lgc.Reorder():
                body = self._lower_query_of_aggregate(var, op, rhs, idxs)
        return ntn.Block((*stmts, body))

    def _lower_query_of_reorder(
        self,
        query_lhs: lgc.Alias,
        op: FinchOperator,
        arg: lgc.Table,
        reorder_idxs: tuple[lgc.Field, ...],
        loop_order: tuple[lgc.Field, ...],
    ):
        # The table is broadcast over the fields only the output has.
        arg_dims = arg.dimmap(merge_shapes, self.shapes)
        shapes_map = {
            **dict(zip(reorder_idxs, self.shapes[query_lhs], strict=True)),
            **dict(zip(arg.idxs, arg_dims, strict=True)),
        }
        shapes = {
            idx: shapes_map.get(idx) or ntn.Literal(ftypes.intp(1))
            for idx in loop_order
        }
        arg_types = arg.shape_type(self.shape_types)
        shape_type_map = {
            **dict(zip(reorder_idxs, self.shape_types[query_lhs], strict=True)),
            **dict(zip(arg.idxs, arg_types, strict=True)),
        }
        shape_type = {idx: shape_type_map.get(idx) or ftypes.intp for idx in loop_order}
        # Visit the output in loop order. Revisit an input field with a fresh
        # index when its storage order conflicts, restricting that loop to the
        # original index below. This keeps sparse reads and writes concordant.
        loop_idxs = []
        remap_idxs = {}
        read_idxs = []
        pending = iter(arg.idxs)
        read_idx = next(pending, None)
        for idx in loop_order:
            loop_idxs.append(idx)
            if idx == read_idx:
                read_idxs.append(idx)
                read_idx = next(pending, None)
            while read_idx is not None and read_idx in loop_idxs:
                new_idx = lgc.Field(gensym(f"{read_idx.name}_"))
                remap_idxs[new_idx] = read_idx
                loop_idxs.append(new_idx)
                read_idxs.append(new_idx)
                read_idx = next(pending, None)
        assert read_idx is None
        loops = {
            idx: ntn.Variable(
                gensym(idx.name),
                shape_type.get(idx) or shape_type[remap_idxs[idx]],
            )
            for idx in loop_idxs
        }
        ctx = PointwiseContext(self)
        rhs = ctx(lgc.Table(arg.tns, tuple(read_idxs)), loops)
        lhs_access = ntn.Access(
            self.slots[query_lhs],
            ntn.Update(ntn.Literal(op)),
            tuple(loops[idx] for idx in reorder_idxs),
        )
        body: ntn.NotationStatement = ntn.Increment(lhs_access, rhs)
        for idx in reversed(loop_idxs):
            stop = shapes.get(idx) or shapes[remap_idxs[idx]]
            ext = ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(stop.result_type(0)), stop),
            )
            if idx in remap_idxs:
                body = ntn.If(
                    ntn.Call(
                        ntn.Literal(ffuncs.eq),
                        (loops[idx], loops[remap_idxs[idx]]),
                    ),
                    body,
                )
            body = ntn.Loop(
                loops[idx],
                ext,
                body,
            )

        return body

    def _lower_query_of_aggregate(
        self,
        query_lhs: lgc.Alias,
        agg_op: FinchOperator,
        agg_arg: lgc.Reorder,
        output_idxs: tuple[lgc.Field, ...],
    ):
        # The loop order holds every field of the output, and the argument is
        # broadcast over the fields only the output has.
        loop_idxs = agg_arg.idxs
        arg_dims = agg_arg.arg.dimmap(merge_shapes, self.shapes)
        shapes_map = {
            **dict(zip(output_idxs, self.shapes[query_lhs], strict=True)),
            **dict(zip(agg_arg.arg.fields(), arg_dims, strict=True)),
        }
        shapes = {idx: shapes_map.get(idx) or ntn.Literal(1) for idx in loop_idxs}
        arg_types = agg_arg.arg.shape_type(self.shape_types)
        shape_type_map = {
            **dict(zip(output_idxs, self.shape_types[query_lhs], strict=True)),
            **dict(zip(agg_arg.arg.fields(), arg_types, strict=True)),
        }
        shape_type = {idx: shape_type_map.get(idx) or ftypes.intp for idx in loop_idxs}
        loops = {
            idx: ntn.Variable(gensym(idx.name), shape_type[idx]) for idx in loop_idxs
        }
        ctx = PointwiseContext(self)
        rhs = ctx(agg_arg.arg, loops)
        lhs_access = ntn.Access(
            self.slots[query_lhs],
            ntn.Update(ntn.Literal(agg_op)),
            tuple(loops[idx] for idx in output_idxs),
        )
        body: ntn.NotationStatement = ntn.Increment(lhs_access, rhs)
        for idx in reversed(loop_idxs):
            ext = ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(shape_type[idx](0)), shapes[idx]),
            )
            body = ntn.Loop(
                loops[idx],
                ext,
                body,
            )

        return body

    def __call__(self, prgm: lgc.LogicStatement) -> ntn.NotationStatement:
        """
        Lower Finch Logic to Finch Notation. First we check for early
        simplifications, then we call the normal lowering for the outermost
        node.
        """
        match prgm:
            case lgc.Plan(bodies):
                stmts = (self(body) for body in bodies)
                return ntn.Block(tuple(s for s in stmts if s != ntn.Block(())))
            case lgc.Query(lgc.Table(lgc.Alias() as lhs, idxs), lgc.Table() as arg):
                # A copy starts from the fill value of arg, and the fields of
                # arg which lhs lacks are reduced by overwriting.
                loop_order = (*idxs, *(idx for idx in arg.idxs if idx not in idxs))
                return self.fold(
                    lhs.unfused,
                    idxs,
                    ffuncs.overwrite,
                    lgc.Reorder(arg, loop_order),
                    self.lower_init(lgc.fill_expression(arg)),
                )
            case lgc.Query(
                lgc.Table(lgc.Alias() as lhs, idxs),
                lgc.Aggregate(lgc.Literal(op), init, lgc.Reorder() as arg, _),
            ):
                stmts = self.freeze_reads(init)
                body = self.fold(lhs.unfused, idxs, op, arg, self.lower_init(init))
                return ntn.Block((*stmts, body))
            case lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, idxs),
                lgc.Literal(ffuncs.overwrite),
                lgc.Literal(init),
            ):
                lhs = lhs.unfused
                op = ntn.Literal(ffuncs.overwrite)
                fill = self.bindings[lhs].fill_type
                if is_static(fill) and not bool(ffuncs.same(init, fill.value)):
                    # A static fill can't change, so lhs is declared with its
                    # fill, and then every element is overwritten with init.
                    stmts = self.declare(lhs, ntn.Literal(fill.value), op)
                    arg = lgc.Reorder(lgc.Literal(init), idxs)
                    body = self._lower_query_of_aggregate(
                        lhs, ffuncs.overwrite, arg, idxs
                    )
                    return ntn.Block((*stmts, body))
                return ntn.Block(self.declare(lhs, ntn.Literal(init), op))
            case lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, idxs),
                lgc.Literal(op),
                lgc.Reorder() as rhs,
            ):
                if op == ffuncs.overwrite:
                    return self.overwrite(lhs.unfused, idxs, rhs)
                return self.fold(lhs.unfused, idxs, op, rhs, None)
            case lgc.Produces(args):
                vars: list[lgc.Alias] = []
                for var in args:
                    assert isinstance(var, lgc.Alias)
                    vars.append(var.unfused)
                return ntn.Block(
                    (
                        *(stmt for var in self.bindings for stmt in self.freeze(var)),
                        *self.epilogue,
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(ffuncs.make_tuple),
                                tuple(self.args[var] for var in vars),
                            )
                        ),
                    )
                )
            case _:
                raise Exception(f"Unrecognized logic: {prgm}")


class CompilerFormLowerer(FormattedForm, LogicLoader):
    """
    Pass a program in FormattedForm to a loader which expects CompilerForm.
    CompilerForm keeps the definitions and updates of FormattedForm, so the
    program is unchanged.
    """

    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
        lgc.LogicStatement,
    ]:
        return self.ctx(prgm, bindings, stats, stats_factory)


class NotationGenerator(LogicNotationLowerer):
    def lower(
        self,
        term: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> ntn.Module:
        preamble: list[ntn.NotationStatement] = []
        epilogue: list[ntn.NotationStatement] = []
        args: dict[lgc.Alias, ntn.Variable] = {}
        slots: dict[lgc.Alias, ntn.Slot] = {}
        shapes: dict[lgc.Alias, tuple[ntn.Variable | None, ...]] = {}
        for arg in bindings:
            args[arg] = ntn.Variable(gensym(f"{arg.name}"), bindings[arg])
            slots[arg] = ntn.Slot(gensym(f"_{arg.name}"), bindings[arg])
            preamble.append(
                ntn.Unpack(
                    slots[arg],
                    args[arg],
                )
            )
            shape: list[ntn.Variable] = []
            for i, t in enumerate(bindings[arg].shape_type):
                dim = ntn.Variable(gensym(f"{arg.name}_dim_{i}"), t)
                shape.append(dim)
                preamble.append(
                    ntn.Assign(dim, ntn.Dimension(slots[arg], ntn.Literal(i)))
                )
            shapes[arg] = tuple(shape)
            epilogue.append(
                ntn.Repack(
                    slots[arg],
                    args[arg],
                )
            )
        ctx = NotationContext(
            bindings,
            args,
            slots,
            shapes,
            epilogue=epilogue,
        )
        body = ctx(term)
        ret_t = None
        for node in PostOrderDFS(body):
            match node:
                case ntn.Return(expr):
                    ret_t = expr.result_type
        assert ret_t is not None
        return ntn.Module(
            (
                ntn.Function(
                    ntn.Variable(
                        "main",
                        AssemblyKernelFType(
                            "main",
                            tuple(arg.result_type for arg in args.values()),
                            ret_t,
                        ),
                    ),
                    tuple(args.values()),
                    ntn.Block((*preamble, body)),
                ),
            )
        )


class LogicCompiler(CompilerForm, LogicLoader):
    def __init__(
        self,
        ctx_load: NotationLoader | None = None,
        ctx_lower: LogicNotationLowerer | None = None,
    ):
        if ctx_load is None:
            ctx_load = NotationInterpreter()
        if ctx_lower is None:
            ctx_lower = NotationGenerator()
        self.ctx_load: NotationLoader = ctx_load
        self.ctx_lower: LogicNotationLowerer = ctx_lower

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
        lgc.LogicStatement,
    ]:
        mod = self.ctx_lower(prgm, bindings, stats, stats_factory)
        logger.debug(mod)
        lib = self.ctx_load(mod)
        shape_vars = compute_shape_vars(prgm, bindings)
        return lib, bindings, shape_vars, prgm
