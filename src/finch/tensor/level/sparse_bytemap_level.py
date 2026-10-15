from dataclasses import dataclass
from typing import Any

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import FType, MutableStructFType, ffuncs, ftype, ftypes, is_dynamic
from finch.compile import looplets as lplt
from finch.finch_assembly import BufferFType, parse_assembly

from .level import (
    Level,
    LevelFType,
    SingleDimensionLevel,
    SingleDimensionLevelFType,
    exec_block,
    lower_heapsort,
)

# Julia leaves an unoccupied parent's pointer at 0, which interop shifts to -1.
_UNSET = -1


@dataclass(unsafe_hash=True)
class SparseByteMapLevelFType(SingleDimensionLevelFType, MutableStructFType):
    _lvl_t: LevelFType
    dimension_type: ftypes.FDTypeInteger = ftypes.intp

    def __post_init__(self) -> None:
        if not isinstance(self._lvl_t, LevelFType):
            raise TypeError("SparseByteMapLevelFType lvl_t must be a level type")
        self.dimension_type = ftype(self.dimension_type)

    @property
    def struct_name(self) -> str:
        return "SparseByteMapLevelFType"

    @property
    def struct_fields(self) -> list[tuple[str, FType]]:
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("ptr", self.ptr_type),
            ("tbl", self.tbl_type),
            ("srt", self.srt_type),
            ("qos_fill", self.position_type),
            ("qos_stop", self.position_type),
        ]

    def __str__(self) -> str:
        return f"SparseByteMapLevelFType({self.lvl_t})"

    @property
    def ndim(self) -> int:
        return 1 + self.lvl_t.ndim

    @property
    def fill_type(self):
        return self.lvl_t.fill_type

    @property
    def element_type(self) -> FType:
        return self.lvl_t.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        return (self.dimension_type, *self.lvl_t.shape_type)

    @property
    def position_type(self) -> FType:
        return self.lvl_t.position_type

    @property
    def buffer_type(self) -> FType:
        return self.lvl_t.buffer_type

    @property
    def buffer_factory(self) -> Any:
        return self.lvl_t.buffer_factory

    @property
    def ptr_type(self) -> BufferFType:
        return self.buffer_factory(self.position_type)

    @property
    def tbl_type(self) -> FType:
        return self.buffer_factory(ftypes.bool_)

    @property
    def srt_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def lvl_t(self) -> LevelFType:
        return self._lvl_t

    def level_cost(self, fields, stats, stats_factory, num_pos, lvl): ...

    def level_format_properties(self, n):
        return self.lvl_t.level_format_properties(n + 1)

    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "SparseByteMapLevel":
        dimension = self.dimension_type(shape[0])
        child_pos = int(pos) * int(dimension)
        lvl = self.lvl_t.construct(shape=shape[1:], pos=child_pos)
        ptr = self.ptr_type(int(pos) + 1)
        for p in range(1, int(pos) + 1):
            ptr.store(p, _UNSET)
        return SparseByteMapLevel(
            lvl, dimension, ptr, self.tbl_type(child_pos), self.srt_type(0)
        )

    def __call__(self, val: Any) -> "SparseByteMapLevel":
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    def _bindings(self, lvl) -> dict[str, Any]:
        p_t = self.position_type
        l_t = self.ptr_type.length_type
        return {
            "ptr": asm.GetAttr(lvl, asm.Literal("ptr")),
            "tbl": asm.GetAttr(lvl, asm.Literal("tbl")),
            "srt": asm.GetAttr(lvl, asm.Literal("srt")),
            "n_ptr": asm.Length(asm.GetAttr(lvl, asm.Literal("ptr"))),
            "n_tbl": asm.Length(asm.GetAttr(lvl, asm.Literal("tbl"))),
            "to_len": asm.Literal(ffuncs.astype(l_t)),
            "to_pos": asm.Literal(ffuncs.astype(p_t)),
            "fld": asm.Literal(ffuncs.floordiv),
            "zero": asm.Literal(p_t(0)),
            "unset": asm.Literal(p_t(_UNSET)),
            "absent": asm.Literal(np.bool_(False)),
            "present": asm.Literal(np.bool_(True)),
        }

    def _var(self, ctx, name, type_=None):
        return asm.Variable(ctx.freshen(name), type_ or self.ptr_type.length_type)

    def level_lower_dim(self, ctx, obj, r):
        if r == 0:
            return asm.GetAttr(obj, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(
            ctx, asm.GetAttr(obj, asm.Literal("lvl")), r - 1
        )

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        l_t = self.ptr_type.length_type
        pos_stop, dim, q_stop, n_dirty, p, q, r, ptr_len, tbl_len = (
            self._var(ctx, name)
            for name in (
                "pos_stop",
                "dim",
                "q_stop",
                "n_dirty",
                "p",
                "q",
                "r",
                "ptr_len",
                "tbl_len",
            )
        )
        # Dirty positions are sorted, so visiting them streams through memory;
        # past about 1/16 of the positions, clearing them all is cheaper.
        expr = """finch
        pos_stop = to_len(pos)
        dim = to_len(tns.dimension)
        q_stop = mul(pos_stop, dim)
        n_dirty = to_len(tns.qos_fill)
        if (mul(n_dirty, 16) > n_tbl)
            for (p in 0:n_ptr)
                ptr[p] = unset
            end
            for (q in 0:n_tbl)
                tbl[q] = absent
            end
        else
            for (r in 0:n_dirty)
                q = to_len(srt[r])
                p = fld(q, dim)
                ptr[p] = unset
                ptr[add(p, 1)] = unset
                tbl[q] = absent
            end
        end
        tns.qos_fill = zero
        ptr_len = n_ptr
        if (ptr_len < add(pos_stop, 1))
            resize(ptr, add(pos_stop, 1))
            for (p in ptr_len:pos_stop + 1)
                ptr[p] = unset
            end
        end
        tbl_len = n_tbl
        if (tbl_len < q_stop)
            resize(tbl, q_stop)
            for (q in tbl_len:q_stop)
                tbl[q] = absent
            end
        end
        ptr[0] = zero
        """
        exec_block(
            ctx, parse_assembly(expr, locals() | self._bindings(tns), position_type=l_t)
        )
        child = asm.GetAttr(tns, asm.Literal("lvl"))
        child_pos = asm.Call(asm.Literal(ffuncs.mul), (pos, dim))
        self.lvl_t.level_lower_declare(ctx, child, init, op, shape, child_pos)
        self.lvl_t.level_lower_assemble(ctx, child, tbl_len, q_stop)

    def level_lower_assemble(self, ctx, lvl, start, stop):
        l_t = self.ptr_type.length_type
        p_start, p_stop, dim, q_start, q_stop, p, q, ptr_len, tbl_len, old = (
            self._var(ctx, name)
            for name in (
                "p_start",
                "p_stop",
                "dim",
                "q_start",
                "q_stop",
                "p",
                "q",
                "ptr_len",
                "tbl_len",
                "old",
            )
        )
        # Existing storage is already clean; only initialize what this call grows.
        expr = """finch
        p_start = to_len(start)
        p_stop = to_len(stop)
        dim = to_len(lvl.dimension)
        q_start = mul(p_start, dim)
        q_stop = mul(p_stop, dim)
        ptr_len = n_ptr
        if (ptr_len < add(p_stop, 1))
            resize(ptr, add(p_stop, 1))
        end
        old = max(add(p_start, 1), ptr_len)
        for (p in old:p_stop + 1)
            ptr[p] = unset
        end
        tbl_len = n_tbl
        old = max(q_start, tbl_len)
        if (tbl_len < q_stop)
            resize(tbl, q_stop)
        end
        for (q in old:q_stop)
            tbl[q] = absent
        end
        """
        exec_block(
            ctx, parse_assembly(expr, locals() | self._bindings(lvl), position_type=l_t)
        )
        self.lvl_t.level_lower_assemble(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), old, q_stop
        )

    def level_lower_thaw(self, ctx, lvl, op, pos):
        return self.lvl_t.level_lower_thaw(
            ctx,
            asm.GetAttr(lvl, asm.Literal("lvl")),
            op,
            asm.Call(
                asm.Literal(ffuncs.mul),
                (pos, asm.GetAttr(lvl, asm.Literal("dimension"))),
            ),
        )

    def level_lower_freeze(self, ctx, lvl, op, pos):
        l_t = self.ptr_type.length_type
        pos_stop, dim, n, r, p, p_prev = (
            self._var(ctx, name)
            for name in ("pos_stop", "dim", "n", "r", "p", "p_prev")
        )
        bindings = locals() | self._bindings(lvl) | {"unset_len": asm.Literal(l_t(-1))}
        exec_block(
            ctx,
            parse_assembly(
                """finch
                pos_stop = to_len(pos)
                dim = to_len(lvl.dimension)
                if (add(pos_stop, 1) < n_ptr)
                    resize(ptr, add(pos_stop, 1))
                end
                if (mul(pos_stop, dim) < n_tbl)
                    resize(tbl, mul(pos_stop, dim))
                end
                n = to_len(lvl.qos_fill)
                resize(srt, n)
                """,
                bindings,
                position_type=l_t,
            ),
        )
        lower_heapsort(
            ctx,
            bindings["srt"],
            n,
            lambda a, b: asm.Call(asm.Literal(ffuncs.lt), (a, b)),
        )
        # Bounds are only written around occupied parents, so ptr[p] stays
        # unset for the others.
        exec_block(
            ctx,
            parse_assembly(
                """finch
                p_prev = unset_len
                for (r in 0:n)
                    p = fld(to_len(srt[r]), dim)
                    if (p != p_prev)
                        ptr[add(p_prev, 1)] = to_pos(r)
                        ptr[p] = to_pos(r)
                    end
                    p_prev = p
                end
                ptr[add(p_prev, 1)] = lvl.qos_fill
                lvl.qos_stop = lvl.qos_fill
                """,
                bindings,
                position_type=l_t,
            ),
        )
        return self.lvl_t.level_lower_freeze(
            ctx,
            asm.GetAttr(lvl, asm.Literal("lvl")),
            op,
            asm.Call(asm.Literal(ffuncs.mul), (pos, dim)),
        )

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(
            "SparseByteMapLevelFType does not support level_lower_increment."
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "SparseByteMapLevelFType does not support level_lower_unwrap."
        )

    def level_unfurl(
        self, ctx, fiber: ntn.Fiber, ext, mode: ntn.AccessMode, proto, pos
    ):
        match mode:
            case ntn.Update():
                return self.level_unfurl_update(ctx, fiber, ext, mode, proto, pos)
        level = fiber.lvl
        lvl = ctx(level)
        p_t = self.position_type
        r, r_stop, q_offset, i_stop, i_last = (
            asm.Variable(ctx.freshen(name), p_t)
            for name in ("r", "r_stop", "q_offset", "i_stop", "i_last")
        )
        seeking = asm.Variable(ctx.freshen("seeking"), ftypes.bool_)
        fill = (
            ntn.Value(self.lower_fill(lvl), self.element_type)
            if is_dynamic(self.fill_type)
            else ntn.Literal(self.fill_type.value)
        )
        full = ntn.Full(
            fill,
            tuple(
                ntn.Value(self.level_lower_dim(ctx, lvl, r), self.shape_type[r])
                for r in range(1, self.ndim)
            ),
        )
        bindings = locals() | self._bindings(lvl)
        bindings |= {
            "one": asm.Literal(p_t(1)),
            "dimension": asm.Call(
                asm.Literal(ffuncs.astype(p_t)),
                (asm.GetAttr(lvl, asm.Literal("dimension")),),
            ),
        }

        def parse(expr, **kwargs):
            return parse_assembly(
                expr, bindings | kwargs, position_type=self.ptr_type.length_type
            )

        def stepper_preamble(ctx):
            return parse("""finch
            if (r < r_stop)
                i_stop = sub(srt[to_len(r)], q_offset)
            else
                i_stop = dimension
            end
            """)

        def thunk_preamble(ctx, idx):
            # An unoccupied parent's ptr is unset, though the next one's isn't.
            return parse("""finch
            q_offset = mul(to_pos(pos), dimension)
            r = ptr[to_len(pos)]
            r_stop = ptr[to_len(add(pos, 1))]
            if (r < zero)
                r = zero
                r_stop = zero
            end
            i_stop = dimension
            i_last = unset
            if (r < r_stop)
                i_stop = sub(srt[to_len(r)], q_offset)
                i_last = sub(srt[to_len(sub(r_stop, one))], q_offset)
            end
            """)

        def seek_fn(ctx, ext):
            seek = parse(
                """finch
                seeking = true
                while (seeking)
                    seeking = false
                    if (add(r, one) < r_stop)
                        if (srt[to_len(r)] < add(q_offset, to_pos(start)))
                            r = add(r, one)
                            seeking = true
                        end
                    end
                end
                """,
                start=ctx.ctx(ext.get_start()),
            )
            return asm.Block((*seek.bodies, *stepper_preamble(ctx).bodies))

        def chunk_tail_fn(ctx, idx):
            pos_2 = asm.Variable(ctx.freshen(idx, f"_pos_{self.ndim - 1}"), p_t)
            exec_block(ctx, parse("finch\npos_2 = srt[to_len(r)]", pos_2=pos_2))
            return lplt.Run(
                ntn.Fiber(ntn.Child(level), ntn.Value(pos_2, p_t), (*fiber.idxs, idx))
            )

        return lplt.Thunk(
            preamble=thunk_preamble,
            body=lambda ctx, ext: lplt.Sequence(
                head=lambda ctx, idx: lplt.Stepper(
                    preamble=stepper_preamble,
                    stop=lambda ctx: ntn.Variable(i_stop.name, p_t),
                    chunk=lplt.Sequence(
                        head=lambda ctx, idx: lplt.Run(full),
                        split=lambda ctx, ext: ntn.Variable(i_stop.name, p_t),
                        tail=chunk_tail_fn,
                    ),
                    next=lambda ctx: asm.Assign(
                        r, asm.Call(asm.Literal(ffuncs.add), (r, asm.Literal(p_t(1))))
                    ),
                    seek=seek_fn,
                ),
                split=lambda ctx, idx: ntn.Call(
                    ntn.L(ffuncs.add),
                    (ntn.Variable(i_last.name, p_t), ext.get_unit()),
                ),
                tail=lambda ctx, idx: lplt.Run(full),
            ),
        )

    def level_unfurl_update(self, ctx, fiber, ext, mode, proto, pos):
        lvl = ctx(fiber.lvl)
        p_t = self.position_type
        bindings = self._bindings(lvl) | {
            "lvl": lvl,
            "one": asm.Literal(p_t(1)),
        }

        def lookup(ctx, idx):
            q = asm.Variable(ctx.freshen("q"), p_t)
            dirty = ntn.Variable(ctx.freshen("dirty"), ftypes.bool_)
            dirty_asm = ctx.ctx(dirty)
            local = bindings | {"q": q, "dirty": dirty_asm, "pos": pos}

            def prepare(ctx, _):
                return parse_assembly(
                    """finch
                    q = add(mul(to_pos(pos), to_pos(lvl.dimension)), to_pos(idx))
                    dirty = false
                    """,
                    local | {"idx": ctx.ctx(idx)},
                )

            def finish(ctx, _):
                written = parse_assembly(
                    """finch
                    if (tbl[to_len(q)] == absent)
                        tbl[to_len(q)] = present
                        if (lvl.qos_fill == lvl.qos_stop)
                            if (lvl.qos_stop == zero)
                                lvl.qos_stop = one
                            else
                                lvl.qos_stop = mul(lvl.qos_stop, 2)
                            end
                            if (length_srt < to_len(lvl.qos_stop))
                                resize(srt, to_len(lvl.qos_stop))
                            end
                        end
                        srt[to_len(lvl.qos_fill)] = q
                        lvl.qos_fill = add(lvl.qos_fill, one)
                    end
                    """,
                    local | {"length_srt": asm.Length(bindings["srt"])},
                    position_type=p_t,
                )
                match fiber:
                    case ntn.HollowFiber(dirty=parent_dirty):
                        written = asm.Block(
                            (
                                asm.Assign(ctx.ctx(parent_dirty), asm.Literal(True)),
                                *written.bodies,
                            )
                        )
                return asm.If(dirty_asm, written)

            return lplt.Thunk(
                preamble=prepare,
                body=lambda ctx, ext: lplt.Run(
                    ntn.HollowFiber(
                        ntn.Child(fiber.lvl),
                        ntn.Value(q, p_t),
                        (*fiber.idxs, idx),
                        dirty=dirty,
                    )
                ),
                epilogue=finish,
            )

        return lplt.Lookup(lookup)

    def from_fields(
        self, lvl, dimension, ptr, tbl, srt, qos_fill=None, qos_stop=None
    ) -> "SparseByteMapLevel":
        return SparseByteMapLevel(lvl, dimension, ptr, tbl, srt, qos_fill, qos_stop)


def sparse_bytemap(lvl_t, dimension_type=None):
    return SparseByteMapLevelFType(lvl_t, dimension_type or ftypes.intp)


@dataclass
class SparseByteMapLevel(SingleDimensionLevel):
    """Byte map with sorted, zero-based packed child positions in ``srt``.

    Each child position is ``parent * dimension + index``; ``ptr`` groups
    these positions by parent, and ``tbl`` marks the stored children. An
    occupied parent ``p`` owns ``srt[ptr[p]:ptr[p + 1]]``, while an unoccupied
    parent's ``ptr[p]`` is -1. While assembling, ``srt[:qos_fill]`` lists the
    stored children in insertion order, and ``qos_stop`` is its capacity.
    """

    lvl: Level
    dimension: np.integer
    ptr: Any | None = None
    tbl: Any | None = None
    srt: Any | None = None
    qos_fill: Any = None
    qos_stop: Any = None

    def __post_init__(self) -> None:
        if self.ptr is None:
            self.ptr = self.lvl.buffer_factory(self.lvl.position_type)(1)
        if self.tbl is None:
            self.tbl = self.lvl.buffer_factory(ftypes.bool_)(0)
        if self.srt is None:
            self.srt = self.lvl.buffer_factory(self.lvl.position_type)(0)
        p_t = self.lvl.position_type
        self.qos_fill = p_t(
            self.srt.length() if self.qos_fill is None else self.qos_fill
        )
        self.qos_stop = p_t(
            self.srt.length() if self.qos_stop is None else self.qos_stop
        )

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    @property
    def ftype(self) -> SparseByteMapLevelFType:
        return SparseByteMapLevelFType(self.lvl.ftype, ftype(self.dimension))  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self) -> str:
        return f"SparseByteMapLevel(lvl={self.lvl}, dim={self.dimension})"
