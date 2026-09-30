from dataclasses import dataclass
from typing import Any

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    MutableStructFType,
    ffuncs,
    ftype,
    ftypes,
    is_dynamic,
    np_dtype,
)
from finch.compile import looplets as lplt
from finch.finch_assembly import parse_assembly

from .level import Level, LevelFType, SingleDimensionLevel, SingleDimensionLevelFType


@dataclass(unsafe_hash=True)
class SparseListLevelFType(SingleDimensionLevelFType, MutableStructFType):
    _lvl_t: LevelFType
    dimension_type: ftypes.FDTypeInteger = ftypes.intp

    def __post_init__(self):
        self.dimension_type = ftype(self.dimension_type)

    @property
    def struct_name(self):
        return "SparseListLevelFType"

    @property
    def p_t(self):
        return self.position_type

    @property
    def struct_fields(self):
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("ptr", self.ptr_type),
            ("idx", self.idx_type),
            ("qos_fill", self.position_type),
            ("qos_stop", self.position_type),
            ("prev_pos", self.position_type),
            ("frozen", ftypes.bool_),
        ]

    def __str__(self):
        return f"SparseListLevelFType({self.lvl_t})"

    @property
    def ndim(self):
        return 1 + self.lvl_t.ndim

    @property
    def fill_value(self):
        return self.lvl_t.fill_value

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.lvl_t.element_type

    @property
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return (self.dimension_type, *self.lvl_t.shape_type)

    @property
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.lvl_t.position_type

    @property
    def buffer_type(self):
        return self.lvl_t.buffer_type

    @property
    def buffer_factory(self):
        """
        Returns the ftype of the buffer used for the fibers.
        """
        return self.lvl_t.buffer_factory

    @property
    def ptr_type(self):
        return self.buffer_factory(self.position_type)

    @property
    def idx_type(self):
        return self.buffer_factory(self.dimension_type)

    def level_cost(self, fields, stats, stats_factory, num_pos, lvl) -> float:
        pos_size = np_dtype(self.position_type).itemsize
        size_ptr = (num_pos + 1) * pos_size
        reduce_fields = fields[lvl + 1 :]
        if reduce_fields:
            reduced_stats = stats_factory.aggregate(
                ffuncs.or_, False, reduce_fields, stats
            )
        else:
            reduced_stats = stats
        nnz_prefix = reduced_stats.estimate_non_fill_values()
        size_idx = nnz_prefix * pos_size

        return (
            size_ptr
            + size_idx
            + self.lvl_t.level_cost(fields, stats, stats_factory, nnz_prefix, lvl + 1)
        )

    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "SparseListLevel":
        """
        Creates an instance of SparseListLevel.

        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl_t.construct(shape=shape[1:], pos=0)
        return SparseListLevel(
            lvl,
            self.dimension_type(shape[0]),
            self.ptr_type(int(pos) + 1),
            self.idx_type(0),
        )

    def __call__(self, val: Any) -> "SparseListLevel":
        """
        Convert a level to this sparse list level type.

        Args:
            val: A value to convert to this type.
        Returns:
            A SparseListLevel instance of this type.
        """
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    @property
    def lvl_t(self):
        return self._lvl_t

    def level_format_properties(self, n):
        return self.lvl_t.level_format_properties(n + 1)

    def level_lower_dim(self, ctx, obj, r):
        if r == 0:
            return asm.GetAttr(obj, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(
            ctx, asm.GetAttr(obj, asm.Literal("lvl")), r - 1
        )

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        p_t = self.ptr_type.length_type
        p = asm.Variable(ctx.freshen("p"), p_t)
        stop = asm.Variable(ctx.freshen("pos_stop"), p_t)
        to_size = asm.Literal(ffuncs.astype(p_t))
        zero = asm.Literal(self.position_type(0))
        expr = """finch
        stop = to_size(pos) + 1
        resize(tns.ptr, stop)
        for (p in 0:stop)
            tns.ptr[p] = zero
        end
        tns.qos_fill = zero
        tns.qos_stop = zero
        tns.prev_pos = zero
        tns.frozen = false
        """
        ctx.exec(parse_assembly(expr, locals(), position_type=p_t))
        return self.lvl_t.level_lower_declare(
            ctx, asm.GetAttr(tns, asm.Literal("lvl")), init, op, shape, zero
        )

    def level_lower_thaw(self, ctx, lvl, op, pos):
        p_t = self.ptr_type.length_type
        pos_stop = asm.Variable(ctx.freshen("pos_stop"), p_t)
        p = asm.Variable(ctx.freshen("p"), p_t)
        to_size = asm.Literal(ffuncs.astype(p_t))
        to_pos = asm.Literal(ffuncs.astype(self.position_type))
        zero = asm.Literal(self.position_type(0))
        expr = """finch
        pos_stop = to_size(pos)
        if (lvl.frozen)
            lvl.qos_fill = lvl.ptr[pos_stop]
            lvl.qos_stop = lvl.qos_fill
            lvl.prev_pos = zero
            p = pos_stop
            // Difference backwards while preceding entries are still prefix sums.
            while (p > 0)
                lvl.ptr[p] -= lvl.ptr[p - 1]
                if (lvl.ptr[p] != zero)
                    if (lvl.prev_pos == zero)
                        lvl.prev_pos = to_pos(p)
                    end
                end
                p -= 1
            end
            lvl.frozen = false
        end
        """
        ctx.exec(parse_assembly(expr, locals(), position_type=p_t))
        return self.lvl_t.level_lower_thaw(
            ctx,
            asm.GetAttr(lvl, asm.Literal("lvl")),
            op,
            asm.GetAttr(lvl, asm.Literal("qos_fill")),
        )

    def level_lower_freeze(self, ctx, lvl, op, pos):
        p_t = self.ptr_type.length_type
        pos_stop = asm.Variable(ctx.freshen("pos_stop"), p_t)
        qos_stop = asm.Variable(ctx.freshen("qos_stop"), self.position_type)
        p = asm.Variable(ctx.freshen("p"), p_t)
        to_size = asm.Literal(ffuncs.astype(p_t))
        to_idx_size = asm.Literal(ffuncs.astype(self.idx_type.length_type))
        expr = """finch
        pos_stop = to_size(pos)
        resize(lvl.ptr, pos_stop + 1)
        if (lvl.frozen == false)
            for (p in 0:pos_stop)
                lvl.ptr[p + 1] += lvl.ptr[p]
            end
        end
        qos_stop = lvl.ptr[pos_stop]
        resize(lvl.idx, to_idx_size(qos_stop))
        lvl.qos_fill = qos_stop
        lvl.qos_stop = qos_stop
        lvl.frozen = true
        """
        ctx.exec(parse_assembly(expr, locals(), position_type=p_t))
        return self.lvl_t.level_lower_freeze(
            ctx,
            asm.GetAttr(lvl, asm.Literal("lvl")),
            op,
            asm.GetAttr(lvl, asm.Literal("qos_fill")),
        )

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(
            "SparseListLevelFType does not support level_lower_increment."
        )

    def level_lower_assemble(self, ctx, lvl, start, stop):
        p_t = self.ptr_type.length_type
        p = asm.Variable(ctx.freshen("p"), p_t)
        p_start = asm.Variable(ctx.freshen("p_start"), p_t)
        p_stop = asm.Variable(ctx.freshen("p_stop"), p_t)
        to_size = asm.Literal(ffuncs.astype(p_t))
        length = asm.Length(asm.GetAttr(lvl, asm.Literal("ptr")))
        zero = asm.Literal(self.position_type(0))
        expr = """finch
        p_start = to_size(start) + 1
        p_stop = to_size(stop) + 1
        if (length < p_stop)
            resize(lvl.ptr, p_stop)
        end
        for (p in p_start:p_stop)
            lvl.ptr[p] = zero
        end
        """
        ctx.exec(parse_assembly(expr, locals(), position_type=p_t))

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "SparseListLevelFType does not support level_lower_unwrap."
        )

    def level_unfurl(
        self, ctx, fiber: ntn.Fiber, ext, mode: ntn.AccessMode, proto, pos
    ):
        match mode:
            case ntn.Update():
                return self.level_unfurl_update(ctx, fiber, ext, mode, proto, pos)
        tns = fiber
        level = tns.lvl
        lvl_asm = ctx(level)
        ptr_s = asm.GetAttr(lvl_asm, asm.Literal("ptr"))
        idx_s = asm.GetAttr(lvl_asm, asm.Literal("idx"))

        q = asm.Variable(ctx.freshen("q"), self.position_type)
        q_stop = asm.Variable(ctx.freshen("q_stop"), self.position_type)
        i_stop = asm.Variable(ctx.freshen("i_stop"), self.position_type)
        i_last = asm.Variable(ctx.freshen("i_last"), self.position_type)
        fill = (
            ntn.Value(self.lower_fill(lvl_asm), self.element_type)
            if is_dynamic(self.fill_value)
            else ntn.Literal(self.fill_value.value)
        )
        full = ntn.Full(
            fill,
            tuple(
                ntn.Value(self.level_lower_dim(ctx, lvl_asm, r), self.shape_type[r])
                for r in range(1, self.ndim)
            ),
        )
        tmp_locals = locals()

        def thunk_preamble(ctx, idx):
            expr = """finch
            q = ptr_s[pos]
            q_stop = ptr_s[pos + 1]
            i_stop = 1
            i_last = 0
            if (q < q_stop)
                i_stop = idx_s[q]
                i_last = idx_s[q_stop - 1]
            end
            """
            return parse_assembly(expr, tmp_locals, position_type=self.position_type)

        def seek_fn(ctx, ext):
            start = ctx.ctx(ext.get_start())

            code = f"""finch
            if (idx_s[q] < {start})
              q = scansearch(idx_s, {start}, q, q_stop - 1)
            end
            """

            return parse_assembly(
                code,
                tmp_locals | asm.get_vars_in_expr(start),
                position_type=self.position_type,
            )

        def chunk_tail_fn(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(asm.Assign(pos_2, q))
            return lplt.Run(
                ntn.Fiber(
                    ntn.Child(level),
                    ntn.Value(pos_2, self.position_type),
                    (*tns.idxs, idx),
                )
            )

        return lplt.Thunk(
            preamble=thunk_preamble,
            body=lambda ctx, ext: lplt.Sequence(
                head=lambda ctx, idx: lplt.Stepper(
                    preamble=lambda ctx: asm.IfElse(
                        asm.Call(asm.L(ffuncs.lt), (q, q_stop)),
                        asm.Block((asm.Assign(i_stop, asm.Load(idx_s, q)),)),
                        asm.Block(
                            (
                                asm.Assign(
                                    i_stop,
                                    asm.GetAttr(lvl_asm, asm.Literal("dimension")),
                                ),
                            )
                        ),
                    ),
                    stop=lambda ctx: ntn.Variable(i_stop.name, self.position_type),
                    chunk=lplt.Sequence(
                        head=lambda ctx, idx: lplt.Run(full),
                        split=lambda ctx, ext: ntn.Variable(
                            i_stop.name, self.position_type
                        ),
                        tail=chunk_tail_fn,
                    ),
                    next=lambda ctx: asm.Block(
                        (
                            asm.Assign(
                                q,
                                asm.Call(asm.L(ffuncs.add), (q, asm.L(self.p_t(1)))),
                            ),
                        )
                    ),
                    seek=seek_fn,
                ),
                split=lambda ctx, idx: ntn.Call(
                    ntn.L(ffuncs.add),
                    (ntn.Variable(i_last.name, self.position_type), ext.get_unit()),
                ),
                tail=lambda ctx, idx: lplt.Run(full),
            ),
        )

    def level_unfurl_update(self, ctx, fiber, ext, mode, proto, pos):
        lvl = ctx(fiber.lvl)
        child = asm.GetAttr(lvl, asm.Literal("lvl"))
        p_t = self.position_type
        qos = asm.Variable(ctx.freshen("qos"), p_t)
        dirty = ntn.Variable(ctx.freshen("dirty"), ftypes.bool_)
        dirty_asm = ctx(dirty)
        to_ptr_size = asm.Literal(ffuncs.astype(self.ptr_type.length_type))
        to_idx_size = asm.Literal(ffuncs.astype(self.idx_type.length_type))
        to_index = asm.Literal(ffuncs.astype(self.dimension_type))
        to_pos = asm.Literal(ffuncs.astype(p_t))
        bindings = locals()

        def preamble(ctx, idx):
            body = parse_assembly(
                """finch
            qos = lvl.qos_fill
            """,
                bindings,
                position_type=p_t,
            )
            if ctx.mode.safe or ctx.mode.debug:
                return asm.Block(
                    (
                        *body.bodies,
                        asm.Assert(
                            asm.Call(
                                asm.Literal(ffuncs.not_),
                                (asm.GetAttr(lvl, asm.Literal("frozen")),),
                            )
                        ),
                        asm.Assert(
                            asm.Call(
                                asm.Literal(ffuncs.le),
                                (asm.GetAttr(lvl, asm.Literal("prev_pos")), pos),
                            )
                        ),
                    )
                )
            return body

        def lookup(ctx, idx):
            idx_asm = ctx.ctx(idx)

            def prepare(ctx, _):
                grow = ctx.ctx.block()
                grow.exec(
                    parse_assembly(
                        """finch
                if (lvl.qos_stop == 0)
                    lvl.qos_stop = 1
                else
                    lvl.qos_stop += lvl.qos_stop
                end
                resize(lvl.idx, to_idx_size(lvl.qos_stop))
                """,
                        bindings,
                        position_type=p_t,
                    )
                )
                self.lvl_t.level_lower_assemble(
                    grow, child, qos, asm.GetAttr(lvl, asm.Literal("qos_stop"))
                )
                return asm.Block(
                    (
                        asm.If(
                            asm.Call(
                                asm.Literal(ffuncs.ge),
                                (qos, asm.GetAttr(lvl, asm.Literal("qos_stop"))),
                            ),
                            asm.Block(tuple(grow.emit())),
                        ),
                        asm.Assign(dirty_asm, asm.Literal(False)),
                    )
                )

            def finish(ctx, _):
                written = parse_assembly(
                    """finch
                lvl.idx[to_idx_size(qos)] = to_index(idx_asm)
                qos += 1
                lvl.prev_pos = to_pos(pos + 1)
                """,
                    bindings | {"idx_asm": idx_asm},
                    position_type=p_t,
                )
                match fiber:
                    case ntn.HollowFiber(dirty=parent_dirty):
                        written = asm.Block(
                            (
                                *written.bodies,
                                asm.Assign(ctx.ctx(parent_dirty), asm.Literal(True)),
                            )
                        )
                return asm.If(dirty_asm, written)

            return lplt.Thunk(
                preamble=prepare,
                body=lambda ctx, ext: lplt.Run(
                    ntn.HollowFiber(
                        ntn.Child(fiber.lvl),
                        ntn.Value(qos, p_t),
                        (*fiber.idxs, idx),
                        dirty=dirty,
                    )
                ),
                epilogue=finish,
            )

        def epilogue(ctx, idx):
            return parse_assembly(
                """finch
            lvl.ptr[to_ptr_size(pos + 1)] += qos - lvl.qos_fill
            lvl.qos_fill = qos
            """,
                bindings,
                position_type=p_t,
            )

        return lplt.Thunk(
            preamble=preamble,
            body=lambda ctx, ext: lplt.Lookup(lookup),
            epilogue=epilogue,
        )

    def from_fields(
        self,
        lvl,
        dimension,
        ptr,
        idx,
        qos_fill=None,
        qos_stop=None,
        prev_pos=0,
        frozen=True,
    ) -> "SparseListLevel":
        return SparseListLevel(
            lvl, dimension, ptr, idx, qos_fill, qos_stop, prev_pos, frozen
        )


def sparse_list(lvl_t, dimension_type=None):
    if dimension_type is None:
        dimension_type = lvl_t.dimension_type
    return SparseListLevelFType(lvl_t, dimension_type)


@dataclass
class SparseListLevel(SingleDimensionLevel):
    """
    A class representing sparse list level.
    """

    lvl: Level
    dimension: np.integer
    ptr: Any = None
    idx: Any = None
    # These cursors must survive serialization while ptr holds per-parent counts.
    qos_fill: Any = None
    qos_stop: Any = None
    # One past the last occupied parent position; zero means none.
    prev_pos: Any = 0
    frozen: bool = True

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    def __post_init__(self):
        if self.ptr is None:
            self.ptr = self.lvl.buffer_factory(self.lvl.position_type)(1)
        if self.idx is None:
            self.idx = self.lvl.buffer_factory(ftype(self.dimension))(0)
        p_t = self.lvl.position_type
        self.qos_fill = p_t(
            self.idx.length() if self.qos_fill is None else self.qos_fill
        )
        self.qos_stop = p_t(
            self.idx.length() if self.qos_stop is None else self.qos_stop
        )
        self.prev_pos = p_t(self.prev_pos)

    @property
    def ftype(self) -> SparseListLevelFType:
        return SparseListLevelFType(self.lvl.ftype, ftype(self.dimension))

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self):
        return f"SparseListLevel(lvl={self.lvl}, dim={self.dimension})"
