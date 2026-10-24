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
from finch.symbolic import CompilerMode, TermTree, gensym
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

    def declare(
        self, var: lgc.Alias, init: ntn.NotationExpression, op: ntn.Literal
    ) -> ntn.NotationStatement:
        """Reset `var` to `init`, so that it can be updated with `op`."""
        shape = cast(tuple[ntn.NotationExpression, ...], self.shapes[var])
        shape = shape[len(shape) - self.bindings[var].ndim :]
        return ntn.Declare(self.slots[var], init, op, shape)

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
        self.fused[level].hoisted.append(ntn.If(cond, self.declare(var, init, op)))
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
        reduced = tuple(idx for idx in loop_order if idx not in idxs)
        reset: tuple[ntn.NotationStatement, ...] = ()
        if init is not None:
            match init:
                case ntn.Literal(val) if op == ffuncs.overwrite and not reduced:
                    op = ffuncs.init_write(val)
            reset = (self.declare(var, init, ntn.Literal(op)),)
        body = self._lower_query(var, op, arg, idxs, loop_order)
        # The reset spans the fused loops over the fields which var stores or
        # which are reduced into it, so it runs before them.
        level = self.fused_level(var, idxs, reduced)
        if level is None:
            return ntn.Block((*reset, body))
        self.fused[level].hoisted.extend(reset)
        return body

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
        self.fused.append(loop)
        body_2 = self(body)
        self.fused.pop()
        ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(dim.result_type(0)), dim))
        return ntn.Block((*loop.hoisted, *checks, ntn.Loop(loop.var, ext, body_2)))

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
                stmts = self.query_shape(lhs.unfused, idxs, rhs)
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
                    self.fused[level].hoisted.append(
                        self.declare(lhs, ntn.Literal(init), op)
                    )
                    return ntn.Block(())
                if is_static(fill) and not bool(ffuncs.same(init, fill.value)):
                    # A static fill can't change, so lhs is declared with its
                    # fill, and then every element is overwritten with init.
                    reset = self.declare(lhs, ntn.Literal(fill.value), op)
                    body = self.overwrite_elements(lhs, idxs, ntn.Literal(init))
                    return ntn.Block((reset, body))
                return self.declare(lhs, ntn.Literal(init), op)
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
        body = enforce_lifecycles(
            ntn.Block((*map(ntn.Unpack, masks, mask_args), body))
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
                    body,
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


def concordize(
    root: ntn.NotationStatement, mode: CompilerMode
) -> tuple[ntn.NotationStatement, list[ntn.Slot]]:
    """
    A Finch program is concordant when the indices of each access are loop
    indices in the order of their loops. The `concordize` transformation
    selects each out of order index with a fresh index, in a loop over the same
    extent which agrees with the index on the diagonal of a mask. Each select
    loop wraps the outermost statement inside the loops of the indices it
    depends on. It returns the masks, one for each dimension, which the program
    takes as workspaces.

    For example,

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
    masks: dict[ntn.NotationExpression, ntn.Slot] = {}

    def mask(ext: ntn.Call) -> ntn.Slot:
        dim = ext.args[1]
        if dim not in masks:
            masks[dim] = ntn.Slot(
                gensym("_eye"), EyeTensor((0, 0), dtype=np.bool_).ftype
            )
        return masks[dim]

    def visit(node: Any, loops: tuple[ntn.Loop, ...]) -> tuple[Any, list]:
        """`node` with concordant accesses, and the selects it needs which are
        lifted out of it, as (index, loop of the selected index, dependencies)."""
        match node:
            case ntn.Loop(idx, ext, body):
                body, pending = visit(body, (*loops, node))
                node = ntn.Loop(idx, ext, body)
                # The loop unfurls the masks which its index selects from, so it
                # declares them first.
                op = ntn.Literal(ffuncs.overwrite)
                unfurled = {
                    x.tns
                    for x in PostOrderDFS(body)
                    if isinstance(x, ntn.Access) and x.idxs[:1] == (idx,)
                }
                decls = [
                    ntn.Declare(eye, ntn.Literal(np.False_), op, (dim, dim))
                    for dim, eye in masks.items()
                    if eye in unfurled
                ]
                if decls:
                    node = ntn.Block((*decls, node))
            case ntn.Access(tns, acc_mode, idxs):
                # An index is in order when its loop is inside the loop of the
                # index before it. A select loop is innermost, so the indices
                # after it are out of order too.
                order = [loop.idx for loop in loops]
                latest, idxs_2, pending = -1, [], []
                for idx in idxs:
                    if idx in order and order.index(idx) > latest:
                        latest = order.index(idx)
                    elif idx in order:
                        s = ntn.Variable(gensym("s"), idx.result_type)
                        pending.append((s, loops[order.index(idx)], {idx, *idxs_2}))
                        idx, latest = s, len(order)
                    idxs_2.append(idx)
                if pending and mode.safe:
                    logger.warning(
                        "Performance Warning: non-concordant traversal of %s (hint: "
                        "most arrays prefer row major or last index fast, run in "
                        "fast mode to ignore this warning)",
                        node,
                    )
                return ntn.Access(tns, acc_mode, tuple(idxs_2)), pending
            case TermTree():
                children = [visit(child, loops) for child in node.children]
                node = node.make_term(node.head(), *(child for child, _ in children))
                pending = [select for _, selects in children for select in selects]
            case _:
                return node, []
        if not isinstance(node, ntn.NotationStatement):
            return node, pending
        # A select loop has one live iteration, so it may be lifted out of a
        # loop whose index it doesn't depend on.
        lifted = []
        for s, loop, deps in reversed(pending):
            if loops and loops[-1].idx not in deps:
                lifted.insert(0, (s, loop, deps))
                continue
            cond = ntn.Unwrap(ntn.Access(mask(loop.ext), ntn.Read(), (loop.idx, s)))
            node = ntn.Loop(s, loop.ext, ntn.If(cond, node))
        return node, lifted

    root, _ = visit(root, ())
    return root, list(masks.values())


def enforce_lifecycles(root: ntn.NotationStatement) -> ntn.NotationStatement:
    """
    Insert the freezes and thaws which put each tensor in the mode that each
    statement needs. A tensor is read and declared while frozen, and updated
    while thawed with the op of the update. A loop or if statement may repeat
    or skip its body, so each tensor is put in the mode its body leaves it in
    beforehand, and its body leaves it in the mode it started in.
    """

    def uses(stmt: Any) -> list[tuple[Any, ntn.AccessMode | None, ntn.AccessMode]]:
        """The tensors of a statement, in order, with the mode they need and the
        mode they are left in."""
        read = ntn.Read()
        reads: list[tuple[Any, ntn.AccessMode | None, ntn.AccessMode]] = [
            (x.tns, read, read)
            for x in PostOrderDFS(stmt)
            if isinstance(x, ntn.Access) and x.mode == read
        ]
        match stmt:
            case ntn.Increment(ntn.Access(tns, mode, _), _):
                return [*reads, (tns, mode, mode)]
            case ntn.Declare(tns, _, op, _):
                return [*reads, (tns, read, ntn.Update(op))]
            case ntn.Unpack(tns, _):
                return [(tns, None, read)]
            case ntn.Repack(tns, _):
                return [(tns, read, read)]
        return reads

    def ends(node: Any) -> dict[Any, ntn.AccessMode]:
        """The mode each tensor of `node` is left in."""
        match node:
            case ntn.Block(bodies):
                return {tns: mode for body in bodies for tns, mode in ends(body).items()}
            case ntn.Loop(_, _, body):
                return ends(body)
            case ntn.If(cond, body):
                return {**ends(cond), **ends(body)}
        return {tns: after for tns, _, after in uses(node)}

    modes: dict[Any, ntn.AccessMode] = {}

    def transition(tns: Any, mode: ntn.AccessMode) -> list[ntn.NotationStatement]:
        stmts: list[ntn.NotationStatement] = []
        match modes.get(tns, ntn.Read()):
            case ntn.Update(op) if mode != ntn.Update(op):
                stmts.append(ntn.Freeze(tns, op))
        match mode:
            case ntn.Update(op) if modes.get(tns, ntn.Read()) != mode:
                stmts.append(ntn.Thaw(tns, op))
        modes[tns] = mode
        return stmts

    def visit(node: Any) -> Any:
        match node:
            case ntn.Block(bodies):
                return ntn.Block(tuple(visit(body) for body in bodies))
            case ntn.Loop() | ntn.If():
                stmts = [
                    stmt
                    for tns, mode in ends(node).items()
                    for stmt in transition(tns, mode)
                ]
                match node:
                    case ntn.Loop(idx, ext, body):
                        node = ntn.Loop(idx, ext, visit(body))
                    case ntn.If(cond, body):
                        node = ntn.If(cond, visit(body))
            case _:
                stmts = []
                for tns, need, after in uses(node):
                    if need is not None:
                        stmts += transition(tns, need)
                    modes[tns] = after
        return ntn.Block((*stmts, node)) if stmts else node

    return visit(root)
