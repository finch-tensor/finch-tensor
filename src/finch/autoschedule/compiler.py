from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

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
)
from finch.finch_notation import NotationInterpreter
from finch.finch_notation.stages import NotationLoader
from finch.symbolic import (
    CompilerMode,
    Fixpoint,
    PostWalk,
    Rewrite,
    TermTree,
    gensym,
)
from finch.symbolic.traversal import PostOrderDFS
from finch.tensor.patterns import EyeTensor
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
                        tuple(
                            loops[idx]
                            for idx in self.ctx.stored_idxs(var.unfused, idxs)
                        ),
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


@dataclass
class FusedLoop:
    """An enclosing Fuse node being lowered to a loop over `idx`."""

    idx: lgc.Field
    var: ntn.Variable
    # The statements which run before the loop, rather than in each iteration.
    hoisted: list[ntn.NotationStatement] = field(default_factory=list)
    # The modes of the tensors which hoisted statements declare.
    modes: dict[lgc.Alias, ntn.AccessMode] = field(default_factory=dict)


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
        *,
        mode: CompilerMode | None = None,
    ):
        self.bindings = bindings
        self.mode = mode if mode is not None else CompilerMode()
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
        # The enclosing fused loops, from outermost to innermost.
        self.fused: list[FusedLoop] = []

    def stored_idxs(
        self, var: lgc.Alias, idxs: tuple[lgc.Field, ...]
    ) -> tuple[lgc.Field, ...]:
        """The fields of a table of `var` which its storage holds. A fused alias
        may store only the fields after its fused ones, within the loops which
        iterate over them."""
        split = len(idxs) - self.bindings[var].ndim
        fused = [loop.idx for loop in self.fused]
        if not all(idx in fused for idx in idxs[:split]):
            raise ValueError(
                f"{var} stores only the fields {idxs[split:]} of {idxs}, outside "
                "of the Fuse nodes which iterate over the rest."
            )
        return idxs[split:]

    def fused_level(
        self,
        var: lgc.Alias,
        idxs: tuple[lgc.Field, ...],
        reduced: tuple[lgc.Field, ...] = (),
    ) -> int | None:
        """The outermost fused loop which iterates over a stored field of `var`,
        or a field which a query reduces into `var`, so that `var` is declared
        before it rather than in each iteration."""
        spanned = (*self.stored_idxs(var, idxs), *reduced)
        for level, loop in enumerate(self.fused):
            if loop.idx in spanned:
                return level
        return None

    def hoist(
        self, level: int, var: lgc.Alias, stmts: tuple[ntn.NotationStatement, ...]
    ) -> None:
        self.fused[level].hoisted.extend(stmts)
        for loop in self.fused[level:]:
            loop.modes[var] = self.modes[var]

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
        shape = cast(tuple[ntn.NotationExpression, ...], self.shapes[var])
        shape = shape[len(shape) - self.bindings[var].ndim :]
        return (*stmts, ntn.Declare(self.slots[var], init, op, shape))

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

    def query_shape(
        self, var: lgc.Alias, idxs: tuple[lgc.Field, ...], rhs: lgc.LogicExpression
    ) -> tuple[ntn.NotationStatement, ...]:
        """Bind the output shape from the RHS at this definition."""
        dims = dict(
            zip(rhs.fields(), rhs.dimmap(merge_shapes, self.shapes), strict=True)
        )
        shape = []
        stmts = list(self.check_shapes(rhs))
        # A fused alias may store only its trailing fields, so its leading
        # fused fields take the types of their extents.
        stored_types = self.bindings[var].shape_type
        dim_types = (None,) * (len(idxs) - len(stored_types)) + tuple(stored_types)
        for i, (idx, dim_t) in enumerate(zip(idxs, dim_types, strict=True)):
            value = dims.get(idx) or ntn.Literal((dim_t or ftypes.intp)(1))
            if dim_t is None:
                dim_t = value.result_type
            if value.result_type != dim_t:
                value = ntn.Call(ntn.Literal(ffuncs.astype(dim_t)), (value,))
            dim = ntn.Variable(gensym(f"{var.name}_dim_{i}"), dim_t)
            shape.append(dim)
            stmts.append(ntn.Assign(dim, value))
        self.shapes[var] = tuple(shape)
        self.shape_types[var] = tuple(dim.result_type for dim in shape)
        return self.hoist_invariant(tuple(stmts))

    def hoist_invariant(
        self, stmts: tuple[ntn.NotationStatement, ...]
    ) -> tuple[ntn.NotationStatement, ...]:
        """Run `stmts`, which don't change across fused iterations, such as
        shapes and their checks, before the outermost fused loop."""
        if self.fused:
            self.fused[0].hoisted.extend(stmts)
            return ()
        return stmts

    def check_shapes(
        self, rhs: lgc.LogicExpression, lhs: lgc.Table | None = None
    ) -> tuple[ntn.NotationStatement, ...]:
        """Check shared fields before declaring or updating an output."""
        if not (self.mode.safe or self.mode.debug):
            return ()
        checks = []

        def merge(a, b):
            if a != b:
                left = a if a is not None else ntn.Literal(b.result_type(1))
                right = b if b is not None else ntn.Literal(a.result_type(1))
                check = ntn.Assert(ntn.Call(ntn.Literal(ffuncs.eq), (left, right)))
                if check not in checks:
                    checks.append(check)
            return merge_shapes(a, b)

        dims = rhs.dimmap(merge, self.shapes)
        if lhs is not None:
            lhs_dims = dict(zip(lhs.idxs, lhs.dimmap(merge, self.shapes), strict=True))
            for idx, dim in zip(rhs.fields(), dims, strict=True):
                if idx in lhs_dims:
                    merge(lhs_dims[idx], dim)
        return tuple(checks)

    def freeze_reads(self, expr: lgc.LogicNode) -> tuple[ntn.NotationStatement, ...]:
        """Statements which leave every tensor that `expr` reads frozen."""
        reads = dict.fromkeys(
            node.unfused for node in PostOrderDFS(expr) if isinstance(node, lgc.Alias)
        )
        return tuple(stmt for var in reads for stmt in self.freeze(var))

    def overwrite(
        self,
        var: lgc.Alias,
        idxs: tuple[lgc.Field, ...],
        arg: lgc.LogicExpression,
        loop_order: tuple[lgc.Field, ...],
    ) -> ntn.NotationStatement:
        """Overwrite `var` with `arg` in `loop_order`, reducing the fields `var`
        lacks."""
        # Every element of var is written, so var is reset to its fill value
        # first. An empty reduction writes nothing and leaves var as it was, so
        # var is only reset when every reduced extent is nonzero.
        dims = dict(
            zip(arg.fields(), arg.dimmap(merge_shapes, self.shapes), strict=True)
        )
        reduced = tuple(idx for idx in loop_order if idx not in idxs)
        nonempty = tuple(
            ntn.Call(ntn.Literal(ffuncs.gt), (dim, ntn.Literal(dim.result_type(0))))
            for dim in (dims.get(idx) for idx in reduced)
            if dim is not None
        )
        # An overwrite keeps the static fill of var, and otherwise gives var the
        # fill of arg.
        fill = self.bindings[var].fill_type
        init = (
            ntn.Literal(fill.value)
            if is_static(fill)
            else self.lower_init(lgc.fill_expression(arg))
        )
        if not nonempty:
            return self.fold(var, idxs, ffuncs.overwrite, arg, loop_order, init)
        cond = nonempty[0]
        if len(nonempty) > 1:
            cond = ntn.Call(ntn.Literal(ffuncs.and_), nonempty)
        # The reset spans the fused loops over the reduced fields, so it runs
        # before them. The extents don't change across iterations, so neither
        # does the guard.
        level = self.fused_level(var, idxs, reduced)
        assert level is not None
        op = ntn.Literal(ffuncs.overwrite)
        reset = (
            *self.thaw(var, op),
            ntn.If(cond, ntn.Block(self.declare(var, init, op))),
        )
        self.hoist(level, var, reset)
        return self.fold(var, idxs, ffuncs.overwrite, arg, loop_order, None)

    def fold(
        self,
        var: lgc.Alias,
        idxs: tuple[lgc.Field, ...],
        op: FinchOperator,
        arg: lgc.LogicExpression,
        loop_order: tuple[lgc.Field, ...],
        init: ntn.NotationExpression | None,
    ) -> ntn.NotationStatement:
        """Fold `arg` into `var` with `op` in `loop_order`, after resetting `var`
        to `init` unless it is None."""
        stmts = self.freeze_reads(arg)
        reduced = tuple(idx for idx in loop_order if idx not in idxs)
        if init is None:
            reset = self.thaw(var, ntn.Literal(op))
        else:
            match init:
                case ntn.Literal(val) if op == ffuncs.overwrite and not reduced:
                    op = ffuncs.init_write(val)
            reset = self.declare(var, init, ntn.Literal(op))
        # The reset spans the fused loops over the fields which var stores or
        # which are reduced into it, so it runs before them.
        level = self.fused_level(var, idxs, reduced)
        if level is None:
            stmts += reset
        else:
            self.hoist(level, var, reset)
        body = self._lower_query(var, op, arg, idxs, loop_order)
        return ntn.Block((*stmts, body))

    def _lower_query(
        self,
        query_lhs: lgc.Alias,
        agg_op: FinchOperator,
        arg: lgc.LogicExpression,
        output_idxs: tuple[lgc.Field, ...],
        loop_order: tuple[lgc.Field, ...],
    ):
        # Every field of the loop order is fused, so the enclosing fused loops
        # iterate over them already.
        fused = {loop.idx: loop.var for loop in self.fused}
        loops = {idx: fused[idx] for idx in loop_order}
        ctx = PointwiseContext(self)
        rhs = ctx(arg, loops)
        lhs_access = ntn.Access(
            self.slots[query_lhs],
            ntn.Update(ntn.Literal(agg_op)),
            tuple(loops[idx] for idx in self.stored_idxs(query_lhs, output_idxs)),
        )
        return ntn.Increment(lhs_access, rhs)

    def overwrite_elements(
        self, var: lgc.Alias, idxs: tuple[lgc.Field, ...], init: ntn.Literal
    ) -> ntn.NotationStatement:
        """Overwrite every stored element of `var` with `init`. An
        initialization doesn't get wrapped in fused nods, so this loops
        over the fields which the enclosing fused loops don't."""
        fused = {loop.idx: loop.var for loop in self.fused}
        shapes = dict(zip(idxs, self.shapes[var], strict=True))
        shape_types = {
            idx: shape_type or ftypes.intp
            for idx, shape_type in zip(idxs, self.shape_types[var], strict=True)
        }
        loops = {
            idx: fused[idx]
            if idx in fused
            else ntn.Variable(gensym(idx.name), shape_types[idx])
            for idx in idxs
        }
        lhs_access = ntn.Access(
            self.slots[var],
            ntn.Update(ntn.Literal(ffuncs.overwrite)),
            tuple(loops[idx] for idx in self.stored_idxs(var, idxs)),
        )
        body: ntn.NotationStatement = ntn.Increment(lhs_access, init)
        for idx in reversed(idxs):
            if idx in fused:
                continue
            ext = ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(shape_types[idx](0)), shapes[idx] or ntn.Literal(1)),
            )
            body = ntn.Loop(loops[idx], ext, body)
        return body

    def fused_fields(
        self, lhs: lgc.Table, arg: lgc.LogicExpression
    ) -> tuple[lgc.Field, ...]:
        """The loop order of a query, which is its enclosing Fuse fields from
        outermost to innermost."""
        fields = {*lhs.idxs, *arg.fields()}
        order = tuple(loop.idx for loop in self.fused if loop.idx in fields)
        if len(order) != len(fields):
            unfused = ", ".join(sorted(idx.name for idx in fields - set(order)))
            raise ValueError(f"The fields {unfused} of a query need Fuse nodes.")
        return order

    def fused_extents(
        self, idx: lgc.Field, body: lgc.LogicStatement
    ) -> list[ntn.NotationExpression]:
        """The extent of the fused field `idx` in each statement of `body` which
        has it. The shapes of the tensors which `body` defines follow from the
        shapes of those it reads, so they are known before the loop."""
        shapes = body.infer_dimmap(merge_shapes, dict(self.shapes))
        extents = []
        for stmt in PostOrderDFS(body):
            match stmt:
                case lgc.Query() | lgc.QueryInto():
                    pass
                case _:
                    continue
            for node in PostOrderDFS(stmt):
                match node:
                    case lgc.Table(lgc.Alias() as var, idxs) if idx in idxs:
                        # A workspace which is only initialized has the shape
                        # of its storage, which lacks its fused fields.
                        shape = shapes[var.unfused]
                        pos = idxs.index(idx) - len(idxs) + len(shape)
                        dim = shape[pos] if pos >= 0 else None
                        if dim is not None:
                            extents.append(dim)
                            break
        return extents

    def lower_fuse(
        self, idx: lgc.Field, body: lgc.LogicStatement
    ) -> ntn.NotationStatement:
        """Lower `body` to a loop over `idx`. The shapes of the body, and the
        declarations of tensors which span its iterations, run before the loop."""
        # A field which only an output has is broadcast, with extent 1.
        extents = self.fused_extents(idx, body)
        dim = extents[0] if extents else ntn.Literal(ftypes.intp(1))
        # Every statement of the body has the extent of the loop.
        extent_checks = []
        if self.mode.safe or self.mode.debug:
            for extent in extents:
                check = ntn.Assert(ntn.Call(ntn.Literal(ffuncs.eq), (extent, dim)))
                if extent != dim and check not in extent_checks:
                    extent_checks.append(check)
        checks = self.hoist_invariant(tuple(extent_checks))
        loop = FusedLoop(idx, ntn.Variable(gensym(idx.name), dim.result_type))
        # The tensors which the body only reads are frozen before the loop,
        # rather than in each iteration.
        written = set()
        for node in PostOrderDFS(body):
            match node:
                case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                    lgc.Table(lgc.Alias() as lhs, _), _, _
                ):
                    written.add(lhs.unfused)
        reads = dict.fromkeys(
            node.unfused
            for node in PostOrderDFS(body)
            if isinstance(node, lgc.Alias) and node.unfused not in written
        )
        frozen = tuple(stmt for var in reads for stmt in self.freeze(var))
        self.fused.append(loop)
        start = dict(self.modes)
        stmts = [self(body)]
        # Each iteration leaves every tensor in the mode the next one expects.
        for var, mode in {**start, **loop.modes}.items():
            match mode:
                case ntn.Read():
                    stmts += self.freeze(var)
                case ntn.Update(ntn.Literal() as op):
                    stmts += self.thaw(var, op)
        self.fused.pop()
        ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(dim.result_type(0)), dim))
        # Hoisted declarations may read the frozen tensors in their inits.
        return ntn.Block(
            (
                *frozen,
                *loop.hoisted,
                *checks,
                ntn.Loop(loop.var, ext, ntn.Block(tuple(stmts))),
            )
        )

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
            case lgc.Fuse(idx, body):
                return self.lower_fuse(idx, body)
            case lgc.Query(
                lgc.Table(lgc.Alias() as lhs, idxs) as table, lgc.Table() as arg
            ):
                # A copy starts from the fill value of arg, and the fields of
                # arg which lhs lacks are reduced by overwriting.
                loop_order = self.fused_fields(table, arg)
                stmts = self.query_shape(lhs.unfused, idxs, arg)
                body = self.fold(
                    lhs.unfused,
                    idxs,
                    ffuncs.overwrite,
                    arg,
                    loop_order,
                    self.lower_init(lgc.fill_expression(arg)),
                )
                return ntn.Block((*stmts, body))
            case lgc.Query(
                lgc.Table(lgc.Alias() as lhs, idxs) as table,
                lgc.Aggregate(lgc.Literal(op), init, arg, _) as rhs,
            ):
                loop_order = self.fused_fields(table, arg)
                stmts = self.query_shape(lhs.unfused, idxs, rhs) + self.freeze_reads(
                    init
                )
                body = self.fold(
                    lhs.unfused, idxs, op, arg, loop_order, self.lower_init(init)
                )
                return ntn.Block((*stmts, body))
            case lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, idxs),
                lgc.Literal(ffuncs.overwrite),
                lgc.Literal(init),
            ):
                lhs = lhs.unfused
                op = ntn.Literal(ffuncs.overwrite)
                fill = self.bindings[lhs].fill_type
                level = self.fused_level(lhs, idxs)
                if level is not None:
                    if is_static(fill) and not bool(ffuncs.same(init, fill.value)):
                        raise NotImplementedError(
                            f"Initializing {lhs} across fused iterations to a "
                            "value other than its static fill"
                        )
                    self.hoist(level, lhs, self.declare(lhs, ntn.Literal(init), op))
                    return ntn.Block(())
                if is_static(fill) and not bool(ffuncs.same(init, fill.value)):
                    # A static fill can't change, so lhs is declared with its
                    # fill, and then every element is overwritten with init.
                    stmts = self.declare(lhs, ntn.Literal(fill.value), op)
                    body = self.overwrite_elements(lhs, idxs, ntn.Literal(init))
                    return ntn.Block((*stmts, body))
                return ntn.Block(self.declare(lhs, ntn.Literal(init), op))
            case lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, idxs) as table, lgc.Literal(op), arg
            ):
                loop_order = self.fused_fields(table, arg)
                checks = self.hoist_invariant(self.check_shapes(arg, table))
                if op == ffuncs.overwrite:
                    body = self.overwrite(lhs.unfused, idxs, arg, loop_order)
                else:
                    body = self.fold(lhs.unfused, idxs, op, arg, loop_order, None)
                return ntn.Block((*checks, body)) if checks else body
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
    ]:
        return self.ctx(prgm, bindings, stats, stats_factory)


class NotationGenerator(LogicNotationLowerer):
    def __init__(self, *, mode: CompilerMode | None = None):
        self.mode = mode if mode is not None else CompilerMode()

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
            mode=self.mode,
        )
        # Concordization may add diagonal masks, which the kernel takes as
        # workspaces after its bindings.
        body, masks = concordize(ntn.Block((*preamble, ctx(term))), self.mode)
        mask_args = tuple(
            ntn.Variable(gensym("eye"), mask.result_type) for mask in masks
        )
        func_args = (*args.values(), *mask_args)
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
                            tuple(arg.result_type for arg in func_args),
                            ret_t,
                        ),
                    ),
                    func_args,
                    ntn.Block((*map(ntn.Unpack, masks, mask_args), body)),
                ),
            )
        )


class LogicCompiler(CompilerForm, LogicLoader):
    """Compile logic using the notation loader's safety mode for shape checks."""

    def __init__(
        self,
        ctx_load: NotationLoader | None = None,
        ctx_lower: LogicNotationLowerer | None = None,
    ):
        if ctx_load is None:
            ctx_load = NotationInterpreter()
        if ctx_lower is None:
            ctx_lower = NotationGenerator(mode=ctx_load.mode)
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
    ]:
        mod = self.ctx_lower(prgm, bindings, stats, stats_factory)
        logger.debug(mod)
        lib = self.ctx_load(mod)
        workspaces = mod.funcs[0].args[len(bindings) :]
        return lib, {
            **bindings,
            **{lgc.HardAlias(arg.name): arg.result_type for arg in workspaces},
        }


class DepthCalculator:
    def __init__(self, rec: dict[ntn.Variable, int]):
        self.rec = rec

    def __call__(self, node: ntn.NotationNode) -> int:
        return max(
            (
                self.rec.get(x, 0)
                for x in PostOrderDFS(node)
                if isinstance(x, ntn.Variable)
            ),
            default=0,
        )


def depth_calculator(root: ntn.NotationNode) -> DepthCalculator:
    return DepthCalculator(DepthCalculatorVisitor()(root))


@dataclass
class DepthCalculatorVisitor:
    depth: int = 1
    rec: dict[ntn.Variable, int] = field(default_factory=dict)

    def __call__(self, node: Any) -> dict[ntn.Variable, int]:
        match node:
            case ntn.Loop(idx, ext, body):
                self(ext)
                self.rec[idx] = self.depth
                ctx_2 = DepthCalculatorVisitor(depth=self.depth + 1, rec=self.rec)
                ctx_2(body)
            case ntn.Assign(lhs, rhs):
                self.rec[lhs] = self.depth
                self(rhs)
            case TermTree():
                for child in node.children:
                    self(child)
        return self.rec


@dataclass
class ConcordizeContext:
    """The loop indices and their extents, and the diagonal masks which
    restrict the selected indices, one for each dimension."""

    indices: set[ntn.NotationExpression]
    extents: dict[ntn.NotationExpression, ntn.Call]
    masks: dict[ntn.NotationExpression, ntn.Slot] = field(default_factory=dict)
    # A loop unfurls the masks which it indexes, so it declares them first.
    declares: dict[ntn.NotationExpression, dict[ntn.Slot, ntn.NotationExpression]] = (
        field(default_factory=dict)
    )

    def mask(self, dim: ntn.NotationExpression) -> ntn.Slot:
        if dim not in self.masks:
            eye = EyeTensor((0, 0), dtype=np.bool_)
            self.masks[dim] = ntn.Slot(gensym("_eye"), eye.ftype)
        return self.masks[dim]


def declare_mask(
    mask: ntn.Slot, dim: ntn.NotationExpression
) -> tuple[ntn.NotationStatement, ...]:
    op = ntn.Literal(ffuncs.overwrite)
    fill = ntn.Literal(mask.result_type.fill_type.value)
    return (ntn.Declare(mask, fill, op, (dim, dim)), ntn.Freeze(mask, op))


class ConcordizeVisitor:
    def __init__(self, ctx: ConcordizeContext, scope: set[ntn.NotationExpression]):
        self.ctx = ctx
        self.scope = scope

    def isbound(self, x: Any) -> bool:
        match x:
            case ntn.Literal():
                return True
            case ntn.Variable() | ntn.Slot():
                return x in self.scope
            case ntn.Access(tns, _, idxs):
                return tns in self.scope and all(map(self.isbound, idxs))
            case TermTree():
                return all(map(self.isbound, x.children))
        return False

    def isboundindex(self, x: ntn.NotationExpression) -> bool:
        return x in self.ctx.indices and self.isbound(x)

    def isboundnotindex(self, x: ntn.NotationExpression) -> bool:
        return x not in self.ctx.indices and self.isbound(x)

    def __call__(self, node: Any) -> Any:
        selects = []

        # Select the first bound expression after the bound indices of an
        # access with a fresh index.
        def select(x):
            match x:
                case ntn.Access(tns, mode, idxs):
                    for n, j in enumerate(idxs):
                        if self.isboundnotindex(j):
                            j_2 = ntn.Variable(gensym("s"), j.result_type)
                            selects.append((j_2, j, tns, n))
                            self.scope.add(j_2)
                            self.ctx.indices.add(j_2)
                            return ntn.Access(
                                tns, mode, (*idxs[:n], j_2, *idxs[n + 1 :])
                            )
                        if not self.isboundindex(j):
                            break
            return None

        match node:
            case ntn.Loop() | ntn.Increment() | ntn.Assign() | ntn.If():
                node = Rewrite(PostWalk(Fixpoint(select)))(node)

        match node:
            case ntn.Loop(idx, ext, body):
                ctx_2 = ConcordizeVisitor(self.ctx, self.scope | {idx})
                node = ntn.Loop(idx, ext, ctx_2(body))
                declares = self.ctx.declares.pop(idx, {})
                if declares:
                    stmts = (
                        stmt
                        for mask, dim in declares.items()
                        for stmt in declare_mask(mask, dim)
                    )
                    node = ntn.Block((*stmts, node))
            case ntn.Assign(var, _) | ntn.Declare(var, _, _, _):
                self.scope.add(var)
            case TermTree():
                node = node.make_term(node.head(), *map(self, node.children))

        # Each selected index iterates over the extent of its expression, where
        # it agrees with the expression on the diagonal of a mask.
        for select_idx, idx_ex, tns, n in reversed(selects):
            stmts: tuple[ntn.NotationStatement, ...] = ()
            match idx_ex:
                case ntn.Call(ntn.Literal(ffuncs.identity), (var,)) if (
                    var in self.ctx.extents
                ):
                    ext = self.ctx.extents[var]
                    dim = ext.args[1]
                    mask = self.ctx.mask(dim)
                    self.ctx.declares.setdefault(var, {})[mask] = dim
                case _:
                    var = ntn.Variable(gensym("v"), idx_ex.result_type)
                    dim = ntn.Dimension(tns, ntn.Literal(n))
                    ext = ntn.Call(
                        ntn.Literal(make_extent),
                        (ntn.Literal(select_idx.result_type(0)), dim),
                    )
                    mask = self.ctx.mask(dim)
                    stmts = (ntn.Assign(var, idx_ex), *declare_mask(mask, dim))
            cond = ntn.Unwrap(ntn.Access(mask, ntn.Read(), (var, select_idx)))
            node = ntn.Block((*stmts, ntn.Loop(select_idx, ext, ntn.If(cond, node))))

        return node


def concordize(
    root: ntn.NotationStatement, mode: CompilerMode
) -> tuple[ntn.NotationStatement, list[ntn.Slot]]:
    """
    A raw index is an index expression consisting of a single index node (i.e.
    `A[i]` as opposed to `A[i + 1]`). A Finch program is concordant when all
    indices are raw and row major with respect to the program loop ordering.
    The `concordize` transformation ensures that tensor indices are concordant
    by inserting loops over selected indices, which a diagonal mask restricts
    to the lifted index expressions or transposed indices. It returns the
    masks, which the program takes as workspaces.

    For example,

    ```
    for i in extent(0, n):
        b[] += A[f(i)]
    ```
    becomes
    ```
    for i in extent(0, n):
        v = f(i)
        for s in extent(0, dim(A, 0)):
            if eye[v, s]:
                b[] += A[s]
    ```

    and

    ```
    for i in extent(0, n):
        for j in extent(0, m):
            b[] += A[j, i]
    ```
    becomes
    ```
    for i in extent(0, n):
        for j in extent(0, m):
            for s in extent(0, n):
                if eye[i, s]:
                    b[] += A[j, s]
    ```
    """
    depth = depth_calculator(root)
    if mode.safe:
        for node in PostOrderDFS(root):
            match node:
                case ntn.Access(_, _, idxs):
                    for n, idx in enumerate(idxs):
                        if 1 <= depth(idx) < max(map(depth, idxs[:n]), default=0):
                            logger.warning(
                                "Performance Warning: non-concordant traversal of "
                                "%s (hint: most arrays prefer row major or last "
                                "index fast, run in fast mode to ignore this "
                                "warning)",
                                node,
                            )
    extents: dict[ntn.NotationExpression, ntn.Call] = {}
    for node in PostOrderDFS(root):
        match node:
            case ntn.Loop(idx, ntn.Call() as ext, _):
                extents[idx] = ext

    def transpose(node):
        match node:
            case ntn.Access(tns, mode, idxs):
                for n, j in enumerate(idxs):
                    if j in extents and depth(j) <= max(
                        map(depth, idxs[:n]), default=0
                    ):
                        j_2 = ntn.Call(ntn.Literal(ffuncs.identity), (j,))
                        return ntn.Access(tns, mode, (*idxs[:n], j_2, *idxs[n + 1 :]))
        return None

    root = Rewrite(PostWalk(Fixpoint(transpose)))(root)
    tnss = {node.tns for node in PostOrderDFS(root) if isinstance(node, ntn.Access)}
    ctx = ConcordizeContext(set(extents), extents)
    root = ConcordizeVisitor(ctx, tnss)(root)
    return root, list(ctx.masks.values())
