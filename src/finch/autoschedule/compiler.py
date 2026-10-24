from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

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
from finch.symbolic import CompilerMode, gensym
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
    revisits: list[tuple[ntn.Variable, ntn.NotationExpression, ntn.Variable]] = field(
        default_factory=list
    )


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
        match arg:
            case lgc.Table():
                body = self._lower_query_of_table(var, op, arg, idxs, loop_order)
            case _:
                body = self._lower_query_of_aggregate(var, op, arg, idxs, loop_order)
        return ntn.Block((*stmts, body))

    def _lower_query_of_table(
        self,
        query_lhs: lgc.Alias,
        op: FinchOperator,
        arg: lgc.Table,
        output_idxs: tuple[lgc.Field, ...],
        loop_order: tuple[lgc.Field, ...],
    ):
        # The table is broadcast over the fields only the output has.
        arg_dims = arg.dimmap(merge_shapes, self.shapes)
        shapes_map = {
            **dict(zip(output_idxs, self.shapes[query_lhs], strict=True)),
            **dict(zip(arg.idxs, arg_dims, strict=True)),
        }
        shapes = {
            idx: shapes_map.get(idx) or ntn.Literal(ftypes.intp(1))
            for idx in loop_order
        }
        arg_types = arg.shape_type(self.shape_types)
        shape_type_map = {
            **dict(zip(output_idxs, self.shape_types[query_lhs], strict=True)),
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
        # Every field of the loop order is fused, so the enclosing fused loops
        # iterate over them already, and only the revisits need new indices.
        fused = {loop.idx: loop.var for loop in self.fused}
        loops = {idx: fused[idx] for idx in loop_order}
        for new_idx, idx in remap_idxs.items():
            loops[new_idx] = ntn.Variable(gensym(new_idx.name), shape_type[idx])
        ctx = PointwiseContext(self)
        rhs = ctx(lgc.Table(arg.tns, tuple(read_idxs)), loops)
        lhs_access = ntn.Access(
            self.slots[query_lhs],
            ntn.Update(ntn.Literal(op)),
            tuple(loops[idx] for idx in self.stored_idxs(query_lhs, output_idxs)),
        )
        # A transposed read revisits a field after its predecessor in storage
        # order, before any deeper enclosing Fuse loops.
        for new_idx, idx in reversed(remap_idxs.items()):
            stop = shapes[idx]
            ext = ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(stop.result_type(0)), stop),
            )
            preceding = loop_idxs[: loop_idxs.index(new_idx)]
            parent = next(
                loop for loop in reversed(self.fused) if loop.idx in preceding
            )
            parent.revisits.insert(0, (loops[new_idx], ext, loops[idx]))
        return ntn.Increment(lhs_access, rhs)

    def _lower_query_of_aggregate(
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
        inner: ntn.NotationStatement = ntn.Block(tuple(stmts))
        for var, extent, original in reversed(loop.revisits):
            cond = ntn.Call(ntn.Literal(ffuncs.eq), (var, original))
            inner = ntn.Loop(var, extent, ntn.If(cond, inner))
        # Hoisted declarations may read the frozen tensors in their inits.
        return ntn.Block(
            (
                *frozen,
                *loop.hoisted,
                *checks,
                ntn.Loop(loop.var, ext, inner),
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
        return lib, bindings

struct DepthCalculator
    rec
end

depth_calculator(root) = DepthCalculator(DepthCalculatorVisitor()(root))

function (ctx::DepthCalculator)(node::FinchNode)
    maximum(node -> get(ctx.rec, node, 0), PostOrderDFS(node))
end

@kwdef struct DepthCalculatorVisitor
    depth = 1
    rec = Dict()
end

function (ctx::DepthCalculatorVisitor)(node::FinchNode)
    if node.kind === loop
        ctx(node.ext)
        ctx.rec[node.idx] = ctx.depth
        ctx_2 = DepthCalculatorVisitor(; depth=(ctx.depth + 1), rec=ctx.rec)
        ctx_2(node.body)
    elseif node.kind === define
        ctx.rec[node.lhs] = ctx.depth
        ctx(node.body)
    elseif istree(node)
        for child in children(node)
            ctx(child)
        end
    end
    return ctx.rec
end

struct ConcordizeVisitor
    ctx
    scope
end

freshen(ctx::ConcordizeVisitor, tags...) = freshen(ctx.ctx, tags...)

function (ctx::ConcordizeVisitor)(node::FinchNode)
    function isbound(x::FinchNode)
        if isconstant(x) || x in ctx.scope
            return true
        elseif @capture x access(~tns, ~mode, ~idxs...)
            return getroot(tns) in ctx.scope && all(isbound, idxs)
        elseif istree(x)
            return all(isbound, arguments(x))
        else
            return false
        end
    end
    isboundindex(x) = isindex(x) && isbound(x)
    isboundnotindex(x) = !isindex(x) && isbound(x)

    selects = []

    if node.kind === loop || node.kind === assign || node.kind === define ||
        node.kind === sieve
        node = Rewrite(
            Postwalk(
                Fixpoint(
                    @rule access(
                        ~tns, ~mode, ~i..., ~j::isboundnotindex, ~k::All(isboundindex)...
                    ) => begin
                        j_2 = index(freshen(ctx, :s))
                        push!(selects, j_2 => j)
                        push!(ctx.scope, j_2)
                        access(tns, mode, i..., j_2, k...)
                    end
                ),
            ),
        )(
            node
        )
    end

    if node.kind === loop
        ctx_2 = ConcordizeVisitor(ctx.ctx, union(ctx.scope, [node.idx]))
        node = loop(node.idx, node.ext, ctx_2(node.body))
    elseif node.kind === define
        ctx_2 = ConcordizeVisitor(ctx.ctx, union(ctx.scope, [node.lhs]))
        node = define(node.lhs, node.rhs, ctx_2(node.body))
    elseif node.kind === declare
        push!(ctx.scope, node.tns)
    elseif istree(node)
        node = similarterm(node, operation(node), map(ctx, children(node)))
    end

    for (select_idx, idx_ex) in reverse(selects)
        var = variable(freshen(ctx, :v))

        node = define(var, idx_ex, loop(select_idx, VirtualExtent(var, var), node))
    end

    node
end

"""
    concordize(ctx, root)

A raw index is an index expression consisting of a single index node (i.e.
`A[i]` as opposed to `A[i + 1]`). A Finch program is concordant when all indices
are raw and column major with respect to the program loop ordering.  The
`concordize` transformation ensures that tensor indices are concordant by
inserting loops and lifting index expressions or transposed indices into the
loop bounds.

For example,

```
@finch for i = :
    b[] += A[f(i)]
end
```
becomes
```
@finch for i = :
    t = f(i)
    for s = t:t
        b[] += A[s]
    end
end
```

and

```
@finch for i = :, j = :
    b[] += A[i, j]
end
```
becomes
```
@finch for i = :, j = :, s = i:i
    b[] += A[s, j]
end
```
"""
function concordize(ctx::AbstractCompiler, root)
    depth = depth_calculator(root)
    if issafe(get_mode_flag(ctx))
        for node in PostOrderDFS(root)
            if @capture node access(~tns, ~mode, ~i...)
                for n in 1:length(i)
                    if 1 <= depth(i[n]) < maximum(depth.(i[(n + 1):end]); init=0)
                        push_preamble!(
                            ctx,
                            quote
                                @warn(
                                    "Performance Warning: non-concordant traversal of $($(sprint(Finch.FinchNotation.display_expression, MIME"text/plain"(), node))) (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)"
                                )
                            end,
                        )
                    end
                end
            end
        end
    end
    root = Rewrite(
        Postwalk(
            Fixpoint(
                @rule access(~tns, ~mode, ~i..., ~j::isindex, ~k...) => begin
                    if depth(j) <= maximum(depth.(k); init=0)
                        access(~tns, ~mode, ~i..., call(identity, j), ~k...)
                    end
                end
            ),
        ),
    )(
        root
    )
    tnss = unique([
        getroot(node.tns) for node in PostOrderDFS(root) if node.kind === access
    ])
    ConcordizeVisitor(ctx, tnss)(root)
end
