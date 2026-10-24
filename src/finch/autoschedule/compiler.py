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
    Compiles Finch Logic in CompilerForm to Finch Notation. Each Fuse becomes a
    loop, and each query resets its output before the loops it spans. By the
    contract of FusedForm, every enclosing Fuse is a field of the query, and
    the n fused fields of a table are its n outermost enclosing Fuse fields.
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
        """The fields of a table of `var` which its storage holds, after its
        fused fields."""
        return idxs[len(idxs) - self.bindings[var].ndim :]

    def reset(
        self, var: lgc.Alias, idxs: tuple[lgc.Field, ...], stmt: ntn.NotationStatement
    ) -> tuple[ntn.NotationStatement, ...]:
        """Run `stmt`, which resets a table of `var`, before the loops it spans,
        which are every enclosing loop but the ones over its fused fields."""
        n = len(idxs) - self.bindings[var].ndim
        if n < len(self.fused):
            self.fused[n].hoisted.append(stmt)
            return ()
        return (stmt,)

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
        stmts = []
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
        shapes, before the outermost fused loop."""
        if self.fused:
            self.fused[0].hoisted.extend(stmts)
            return ()
        return stmts

    def reduce(
        self,
        var: lgc.Alias,
        idxs: tuple[lgc.Field, ...],
        op: FinchOperator,
        arg: lgc.LogicExpression,
        init: ntn.NotationExpression | None,
    ) -> ntn.NotationStatement:
        """Reduce `arg` into a table of `var` with `op`, after resetting `var` to
        `init` unless it is None."""
        loops = {loop.idx: loop.var for loop in self.fused}
        reset: tuple[ntn.NotationStatement, ...] = ()
        if init is not None:
            match init:
                case ntn.Literal(val) if op == ffuncs.overwrite and set(
                    arg.fields()
                ) <= set(idxs):
                    op = ffuncs.init_write(val)
            reset = self.reset(var, idxs, self.declare(var, init, ntn.Literal(op)))
        lhs = ntn.Access(
            self.slots[var],
            ntn.Update(ntn.Literal(op)),
            tuple(loops[idx] for idx in self.stored_idxs(var, idxs)),
        )
        body = ntn.Increment(lhs, PointwiseContext(self)(arg, loops))
        return ntn.Block((*reset, body)) if reset else body

    def fused_extent(
        self, idx: lgc.Field, body: lgc.LogicStatement
    ) -> ntn.NotationExpression:
        """The extent of the fused field `idx`, from a table of `body` which
        stores it. The shapes of the tensors which `body` defines follow from
        the shapes of those it reads, so they are known before the loop. A
        field which only an output has is broadcast, with extent 1."""
        shapes = body.infer_dimmap(merge_shapes, dict(self.shapes))
        for node in PostOrderDFS(body):
            match node:
                case lgc.Table(lgc.Alias() as var, idxs) if idx in idxs:
                    # A workspace which is only initialized has the shape of
                    # its storage, which lacks its fused fields.
                    shape = shapes[var.unfused]
                    pos = idxs.index(idx) - len(idxs) + len(shape)
                    if pos >= 0 and (dim := shape[pos]) is not None:
                        return dim
        return ntn.Literal(ftypes.intp(1))

    def __call__(self, prgm: lgc.LogicStatement) -> ntn.NotationStatement:
        match prgm:
            case lgc.Plan(bodies):
                stmts = (self(body) for body in bodies)
                return ntn.Block(tuple(s for s in stmts if s != ntn.Block(())))
            case lgc.Fuse(idx, body):
                dim = self.fused_extent(idx, body)
                loop = FusedLoop(idx, ntn.Variable(gensym(idx.name), dim.result_type))
                self.fused.append(loop)
                body = self(body)
                self.fused.pop()
                ext = ntn.Call(
                    ntn.Literal(make_extent), (ntn.Literal(dim.result_type(0)), dim)
                )
                # The shapes of the body, and the resets of tensors which span
                # its iterations, run before the loop.
                return ntn.Block((*loop.hoisted, ntn.Loop(loop.var, ext, body)))
            case lgc.Query(lgc.Table(lgc.Alias() as lhs, idxs), rhs):
                # A copy is a reduction which overwrites, from the fill of arg.
                match rhs:
                    case lgc.Aggregate(lgc.Literal(op), init, arg, _):
                        pass
                    case _:
                        op, init, arg = ffuncs.overwrite, lgc.fill_expression(rhs), rhs
                stmts = self.query_shape(lhs.unfused, idxs, rhs)
                body = self.reduce(lhs.unfused, idxs, op, arg, self.lower_init(init))
                return ntn.Block((*stmts, body))
            case lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, idxs),
                lgc.Literal(ffuncs.overwrite),
                lgc.Literal(init),
            ):
                lhs = lhs.unfused
                op = ntn.Literal(ffuncs.overwrite)
                fill = self.bindings[lhs].fill_type
                if not is_static(fill) or bool(ffuncs.same(init, fill.value)):
                    return ntn.Block(
                        self.reset(lhs, idxs, self.declare(lhs, ntn.Literal(init), op))
                    )
                if len(idxs) - self.bindings[lhs].ndim < len(self.fused):
                    raise NotImplementedError(
                        f"Initializing {lhs} across fused iterations to a value "
                        "other than its static fill"
                    )
                # A static fill can't change, so lhs is declared with its fill,
                # and then every element is overwritten with init. Its stored
                # fields aren't fused, so this loops over them.
                fused = {loop.idx: loop.var for loop in self.fused}
                shapes = dict(zip(idxs, self.shapes[lhs], strict=True))
                shape_types = {
                    idx: shape_type or ftypes.intp
                    for idx, shape_type in zip(idxs, self.shape_types[lhs], strict=True)
                }
                loops = {
                    idx: fused.get(idx)
                    or ntn.Variable(gensym(idx.name), shape_types[idx])
                    for idx in idxs
                }
                lhs_access = ntn.Access(
                    self.slots[lhs],
                    ntn.Update(op),
                    tuple(loops[idx] for idx in self.stored_idxs(lhs, idxs)),
                )
                body: ntn.NotationStatement = ntn.Increment(
                    lhs_access, ntn.Literal(init)
                )
                for idx in reversed(idxs):
                    if idx not in fused:
                        ext = ntn.Call(
                            ntn.Literal(make_extent),
                            (
                                ntn.Literal(shape_types[idx](0)),
                                shapes[idx] or ntn.Literal(1),
                            ),
                        )
                        body = ntn.Loop(loops[idx], ext, body)
                return ntn.Block((self.declare(lhs, ntn.Literal(fill.value), op), body))
            case lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, idxs), lgc.Literal(op), arg
            ):
                lhs = lhs.unfused
                guard: tuple[ntn.NotationStatement, ...] = ()
                init = None
                if op == ffuncs.overwrite:
                    # Every element of lhs is written, so lhs is reset first. An
                    # overwrite keeps the static fill of lhs, and otherwise
                    # gives lhs the fill of arg.
                    fill = self.bindings[lhs].fill_type
                    init = (
                        ntn.Literal(fill.value)
                        if is_static(fill)
                        else self.lower_init(lgc.fill_expression(arg))
                    )
                    # An empty reduction writes nothing and leaves lhs as it
                    # was, so lhs is only reset when every reduced extent is
                    # nonzero. The extents don't change across iterations, so
                    # neither does the guard.
                    dims = dict(
                        zip(
                            arg.fields(),
                            arg.dimmap(merge_shapes, self.shapes),
                            strict=True,
                        )
                    )
                    nonempty = tuple(
                        ntn.Call(
                            ntn.Literal(ffuncs.gt),
                            (dim, ntn.Literal(dim.result_type(0))),
                        )
                        for idx, dim in dims.items()
                        if idx not in idxs and dim is not None
                    )
                    if nonempty:
                        cond = nonempty[0]
                        if len(nonempty) > 1:
                            cond = ntn.Call(ntn.Literal(ffuncs.and_), nonempty)
                        reset = self.declare(lhs, init, ntn.Literal(op))
                        guard = self.reset(lhs, idxs, ntn.If(cond, reset))
                        init = None
                body = self.reduce(lhs, idxs, op, arg, init)
                return ntn.Block((*guard, body)) if guard else body
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
        )
        # Concordization may add diagonal masks, which the kernel takes as
        # workspaces after its bindings.
        body, masks = concordize(ntn.Block((*preamble, ctx(term))), self.mode)
        mask_args = tuple(
            ntn.Variable(gensym("eye"), mask.result_type) for mask in masks
        )
        body = enforce_lifecycles(ntn.Block((*map(ntn.Unpack, masks, mask_args), body)))
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
                return {
                    tns: mode for body in bodies for tns, mode in ends(body).items()
                }
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
