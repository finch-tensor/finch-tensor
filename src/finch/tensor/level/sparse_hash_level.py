from dataclasses import dataclass
from secrets import randbits
from typing import Any

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    FType,
    MutableStructFType,
    TupleFType,
    ffuncs,
    ftype,
    ftypes,
    is_dynamic,
)
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

_POS_MULTIPLIER = np.uintp(randbits(np.dtype(np.uintp).itemsize * 8) | 1)
# Control bytes and key states, as in Finch.jl.
_CTRL_EMPTY = np.uint8(0x00)
_KEY_FREE = np.uint8(0x00)
_KEY_RETAINED = np.uint8(0x80)
# The end of the free chain, and Julia's 0 shifted by interop.
_NO_POS = -1


class _SparseHashLowering:
    """
    Emits the table operations of the hash level `lvl`, following Finch.jl's
    `sparse_hash_*` helpers. Keys hash as Julia's `hash` of the one-based key,
    so tables are shared with Julia unchanged.
    """

    def __init__(self, fmt: "SparseHashLevelFType", ctx, lvl):
        self.ctx = ctx
        self.fmt = fmt
        self.l_t = fmt.ptr_type.length_type
        p_t, u8, u64 = fmt.position_type, ftypes.uint8, ftypes.uint64

        def field(name):
            return asm.GetAttr(lvl, asm.Literal(name))

        ops = {
            "to_u64": ffuncs.astype(u64),
            "to_u8": ffuncs.astype(u8),
            "to_len": ffuncs.astype(self.l_t),
            "to_pos": ffuncs.astype(p_t),
            "to_idx": ffuncs.astype(fmt.dimension_type),
            "shl": ffuncs.lshift,
            "shr": ffuncs.rshift,
            "bxor": ffuncs.xor,
            "bor": ffuncs.or_,
            "band": ffuncs.and_,
            "bnot": ffuncs.invert,
            "fld": ffuncs.floordiv,
            "make_tuple": ffuncs.make_tuple,
        }
        consts = {
            "EMPTY": _CTRL_EMPTY,
            "FREE": _KEY_FREE,
            "RETAINED": _KEY_RETAINED,
            "ONE_U8": u8(1),
            "MAX_PENDING": u8(_KEY_RETAINED - 1),
            "no_pos": p_t(_NO_POS),
            "zero_pos": p_t(0),
            "one_pos": p_t(1),
            "x7f": u64(0x7F),
            "x80": u64(0x80),
            **{f"u{n}": u64(n) for n in (1, 2, 3, 4, 7, 8, 14, 21, 24, 28, 31, 63)},
        }
        self.bindings: dict[str, Any] = {
            "lvl": lvl,
            "B": field("subtables"),
            "mult": field("pos_multiplier"),
            **{
                name: field(name)
                for name in ("ptr", "tbl_ctrl", "tbl", "key", "perm", "tbl_count")
            },
            "n_ctrl": asm.Length(field("tbl_ctrl")),
            "n_key": asm.Length(field("key")),
            "n_perm": asm.Length(field("perm")),
            "n_count": asm.Length(field("tbl_count")),
            **{name: asm.Literal(f) for name, f in ops.items()},
            **{name: asm.Literal(value) for name, value in consts.items()},
        }

    def var(self, name, type_=None):
        return asm.Variable(self.ctx.freshen(name), type_ or self.l_t)

    def key_var(self):
        return self.var("k", self.fmt.key_entry_type)

    def parse(self, code, /, **bindings):
        """Statements of a template; integer literals are lengths."""
        return parse_assembly(
            "finch\n" + code, self.bindings | bindings, position_type=self.l_t
        ).bodies

    def expr(self, code, /, **bindings):
        (stmt,) = self.parse(f"result = {code}", result=self.var("result"), **bindings)
        return stmt.rhs

    def hash_u64(self, a):
        """Julia's `hash_64_64`, updating the `UInt64` variable `a`."""
        return self.parse(
            """
            a = add(bnot(a), shl(a, u21))
            a = bxor(a, shr(a, u24))
            a = add(add(a, shl(a, u3)), shl(a, u8))
            a = bxor(a, shr(a, u14))
            a = add(add(a, shl(a, u2)), shl(a, u4))
            a = bxor(a, shr(a, u28))
            a = add(a, shl(a, u31))
            """,
            a=a,
        )

    def key_hash(self, p, i):
        """`x = a * p + hash(i)` of the one-based key, as `sparse_hash_hash`."""
        x = self.var("x", ftypes.uint64)
        stmts = (
            *self.parse("x = to_u64(i + 1)", x=x, i=i),
            *self.hash_u64(x),
            *self.parse("x = add(mul(mult, to_u64(p + 1)), x)", x=x, p=p),
        )
        return stmts, x

    def record_hash(self, k):
        return self.key_hash(*self.key_fields(k))

    @staticmethod
    def key_fields(k):
        return (
            asm.GetAttr(k, asm.Literal("element_0")),
            asm.GetAttr(k, asm.Literal("element_1")),
        )

    def subtable(self, x):
        # The high log2(B) bits of x, without shifting by the full width.
        return self.expr("to_len(fld(shr(x, u1), fld(shl(u1, u63), to_u64(B))))", x=x)

    def slot_parts(self, x):
        """The bucket's first slot, the probe's starting offset, the offset
        mask, and `hash(x)`, as `sparse_hash_hash_slot_parts`."""
        base, off, mask = self.var("base"), self.var("off"), self.var("mask")
        y = self.var("y", ftypes.uint64)
        stmts = (
            *self.parse("mask = sub(fld(n_ctrl, B), 1)", mask=mask),
            *self.parse("y = x", x=x, y=y),
            *self.hash_u64(y),
            *self.parse(
                """
                base = mul(b, add(mask, 1))
                off = to_len(band(shr(y, u7), to_u64(mask)))
                """,
                base=base,
                b=self.subtable(x),
                mask=mask,
                off=off,
                y=y,
            ),
        )
        return stmts, base, off, mask, y

    def ctrl(self, y):
        return self.expr("to_u8(bor(x80, band(y, x7f)))", y=y)

    def restate(self, k, state):
        """The child record `k` with the writer state `state`."""
        return self.expr("make_tuple(k.element_0, k.element_1, s)", k=k, s=state)

    def find(self, p, i, x):
        """The slot holding `(p, i)`, which hashes to `x`, or else the empty slot
        ending its probe, as `sparse_hash_find`."""
        parts, base, off, mask, y = self.slot_parts(x)
        h, c, k = self.var("h"), self.var("c", ftypes.uint8), self.key_var()
        searching = self.var("searching", ftypes.bool_)
        stmts = self.parse(
            """
            h = add(base, off)
            searching = true
            while (searching)
                c = tbl_ctrl[h]
                if (c == EMPTY)
                    searching = false
                else
                    if (c == ctrl)
                        k = key[to_len(tbl[h])]
                        if (k.element_0 == p)
                            if (k.element_1 == i)
                                searching = false
                            end
                        end
                    end
                    if (searching)
                        off = band(add(off, 1), mask)
                        h = add(base, off)
                    end
                end
            end
            """,
            **locals(),
            ctrl=self.ctrl(y),
        )
        return (*parts, *stmts), h, y

    def vacancy(self, x):
        """The first empty slot on `x`'s probe, as `sparse_hash_vacancy`."""
        parts, base, off, mask, y = self.slot_parts(x)
        h = self.var("h")
        stmts = self.parse(
            """
            h = add(base, off)
            while (tbl_ctrl[h] != EMPTY)
                off = band(add(off, 1), mask)
                h = add(base, off)
            end
            """,
            **locals(),
        )
        return (*parts, *stmts), h, y

    def resize_table(self, cap):
        """Rebuild the table with `cap` slots from the live child records, as
        `sparse_hash_resize!`."""
        q, h, cap_2, k = self.var("q"), self.var("h"), self.var("cap"), self.key_var()
        hashing, x = self.record_hash(k)
        placing, h_2, y = self.vacancy(x)
        insert = (
            *hashing,
            *placing,
            *self.parse(
                """
                tbl_ctrl[h] = ctrl
                tbl[h] = to_pos(q)
                """,
                h=h_2,
                q=q,
                ctrl=self.ctrl(y),
            ),
        )
        return (
            *self.parse(
                """
                cap = cap_e
                resize(tbl_ctrl, cap)
                for (h in 0:cap)
                    tbl_ctrl[h] = EMPTY
                end
                resize(tbl, cap)
                """,
                cap=cap_2,
                cap_e=cap,
                h=h,
            ),
            asm.ForLoop(
                q,
                asm.Literal(self.l_t(0)),
                self.expr("to_len(lvl.qos_stop)"),
                asm.Block(
                    (
                        *self.parse("k = key[q]", k=k, q=q),
                        asm.If(
                            self.expr("k.element_2 != FREE", k=k), asm.Block(insert)
                        ),
                    )
                ),
            ),
        )

    def delete(self, h, x):
        """Backward-shift delete slot `h`, which holds a key hashing to `x`, as
        `sparse_hash_delete!`. Only slots move; child positions stay stable."""
        parts, base, _, mask, _ = self.slot_parts(x)
        hole, scan, q, k = (
            self.var("hole"),
            self.var("scan"),
            self.var("q"),
            self.key_var(),
        )
        hashing, x_2 = self.record_hash(k)
        home_parts, _, home, _, _ = self.slot_parts(x_2)
        step = (
            *self.parse(
                """
                q = to_len(tbl[add(base, scan)])
                k = key[q]
                """,
                **locals(),
            ),
            *hashing,
            *home_parts,
            *self.parse(
                """
                if (band(sub(hole, home), mask) < band(sub(scan, home), mask))
                    tbl_ctrl[add(base, hole)] = tbl_ctrl[add(base, scan)]
                    tbl[add(base, hole)] = to_pos(q)
                    hole = scan
                end
                scan = band(add(scan, 1), mask)
                """,
                **locals(),
            ),
        )
        return (
            *parts,
            *self.parse(
                """
                hole = sub(h, base)
                scan = band(add(hole, 1), mask)
                """,
                **locals(),
            ),
            asm.WhileLoop(
                self.expr("tbl_ctrl[add(base, scan)] != EMPTY", base=base, scan=scan),
                asm.Block(step),
            ),
            *self.parse("tbl_ctrl[add(base, hole)] = EMPTY", base=base, hole=hole),
        )


@dataclass(unsafe_hash=True)
class SparseHashLevelFType(SingleDimensionLevelFType, MutableStructFType):
    _lvl_t: LevelFType
    dimension_type: ftypes.FDTypeInteger = ftypes.intp

    def __post_init__(self) -> None:
        if not isinstance(self._lvl_t, LevelFType):
            raise TypeError("SparseHashLevelFType lvl_t must be a level type")
        self.dimension_type = ftype(self.dimension_type)

    @property
    def struct_name(self) -> str:
        return "SparseHashLevelFType"

    @property
    def struct_fields(self) -> list[tuple[str, FType]]:
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("subtables", ftypes.intp),
            ("pos_multiplier", ftype(np.uintp)),
            ("ptr", self.ptr_type),
            ("tbl_ctrl", self.tbl_ctrl_type),
            ("tbl", self.tbl_type),
            ("key", self.key_type),
            ("perm", self.perm_type),
            ("tbl_count", self.tbl_count_type),
            ("qos_stop", self.position_type),
            ("free_head", self.position_type),
            ("frozen", ftypes.bool_),
        ]

    def __str__(self) -> str:
        return f"SparseHashLevelFType({self.lvl_t})"

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
    def tbl_ctrl_type(self) -> FType:
        return self.buffer_factory(ftypes.uint8)

    @property
    def key_entry_type(self) -> TupleFType:
        return TupleFType.from_tuple(
            (self.position_type, self.dimension_type, ftypes.uint8)
        )

    @property
    def tbl_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def key_type(self) -> FType:
        return self.buffer_factory(self.key_entry_type)

    @property
    def perm_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def tbl_count_type(self) -> FType:
        return self.buffer_factory(ftypes.intp)

    @property
    def lvl_t(self) -> LevelFType:
        return self._lvl_t

    def level_cost(self, fields, stats, stats_factory, num_pos, lvl) -> float:
        pos_size = ftypes.np_dtype(self.position_type).itemsize
        size_ptr = (num_pos + 1) * pos_size
        reduce_fields = fields[lvl + 1 :]
        if reduce_fields:
            reduced_stats = stats_factory.aggregate(
                ffuncs.or_, False, reduce_fields, stats_factory.support(stats)
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

    def level_format_properties(self, n):
        return self.lvl_t.level_format_properties(n + 1)

    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "SparseHashLevel":
        lvl = self.lvl_t.construct(shape=shape[1:], pos=0)
        return SparseHashLevel(
            lvl,
            self.dimension_type(shape[0]),
            self.ptr_type(int(pos) + 1),
            self.tbl_ctrl_type(0),
            self.tbl_type(0),
            self.key_type(0),
            self.perm_type(0),
            self.tbl_count_type(1),
        )

    def __call__(self, val: Any) -> "SparseHashLevel":
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    def level_lower_dim(self, ctx, obj, r):
        if r == 0:
            return asm.GetAttr(obj, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(
            ctx, asm.GetAttr(obj, asm.Literal("lvl")), r - 1
        )

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        hl = _SparseHashLowering(self, ctx, tns)
        r, h, q, k = hl.var("r"), hl.var("h"), hl.var("q"), hl.key_var()
        hashing, x = hl.record_hash(k)
        parts, base, off, mask, _ = hl.slot_parts(x)
        # Clear the table without releasing its capacity. Visiting an entry
        # costs a hash and a random probe, while filling the control array costs
        # well under a nanosecond per slot, so only visit the frozen entries at
        # very low occupancy. Slots don't move while clearing, so probes continue
        # past emptied slots.
        visit = (
            *hl.parse(
                """
                q = to_len(perm[r])
                k = key[q]
                """,
                q=q,
                r=r,
                k=k,
            ),
            *hashing,
            *parts,
            *hl.parse(
                """
                h = add(base, off)
                while (bor(tbl_ctrl[h] == EMPTY, tbl[h] != to_pos(q)))
                    off = band(add(off, 1), mask)
                    h = add(base, off)
                end
                tbl_ctrl[h] = EMPTY
                """,
                **locals(),
            ),
        )
        ctx.exec(
            asm.IfElse(
                hl.expr("band(lvl.frozen, n_perm <= fld(n_ctrl, 4096))"),
                asm.ForLoop(
                    r, asm.Literal(np.intp(0)), hl.bindings["n_perm"], asm.Block(visit)
                ),
                asm.Block(
                    hl.parse(
                        """
                        for (h in 0:n_ctrl)
                            tbl_ctrl[h] = EMPTY
                        end
                        """,
                        h=h,
                    )
                ),
            )
        )
        exec_block(
            ctx,
            asm.Block(
                hl.parse(
                    """
                    resize(key, 0)
                    resize(perm, 0)
                    for (b in 0:n_count)
                        tbl_count[b] = 0
                    end
                    lvl.qos_stop = zero_pos
                    lvl.free_head = no_pos
                    lvl.frozen = false
                    """,
                    b=hl.var("b"),
                )
            ),
        )
        return self.lvl_t.level_lower_declare(
            ctx,
            asm.GetAttr(tns, asm.Literal("lvl")),
            init,
            op,
            shape,
            asm.Literal(self.position_type(0)),
        )

    def level_lower_assemble(self, ctx, lvl, start, stop):
        pass

    def level_lower_thaw(self, ctx, lvl, op, pos):
        hl = _SparseHashLowering(self, ctx, lvl)
        k = hl.key_var()
        # The free chain is local to a write phase. Rebuild it from the holes
        # of a frozen level; a dense set of child positions needs no scan.
        ctx.exec(
            asm.Block(
                hl.parse(
                    """
                    if (lvl.frozen)
                        lvl.qos_stop = to_pos(n_key)
                        lvl.free_head = no_pos
                        if (n_perm < n_key)
                            for (j in 0:n_key)
                                q = sub(sub(n_key, j), 1)
                                k = key[q]
                                if (k.element_2 == FREE)
                                    key[q] = freed
                                    lvl.free_head = to_pos(q)
                                end
                            end
                        end
                        lvl.frozen = false
                    end
                    """,
                    j=hl.var("j"),
                    q=hl.var("q"),
                    k=k,
                    freed=hl.expr("make_tuple(lvl.free_head, k.element_1, FREE)", k=k),
                )
            )
        )
        return self.lvl_t.level_lower_thaw(
            ctx,
            asm.GetAttr(lvl, asm.Literal("lvl")),
            op,
            asm.GetAttr(lvl, asm.Literal("qos_stop")),
        )

    def level_lower_freeze(self, ctx, lvl, op, pos):
        hl = _SparseHashLowering(self, ctx, lvl)
        pos_stop, qos_stop, n, extent, q, r, p = (
            hl.var(name)
            for name in ("pos_stop", "qos_stop", "n", "extent", "q", "r", "p")
        )
        k = hl.key_var()
        # Collect the live child positions, trimming free tails of key so that
        # an empty hash has an empty child.
        exec_block(
            ctx,
            asm.Block(
                hl.parse(
                    """
                    pos_stop = to_len(pos)
                    qos_stop = to_len(lvl.qos_stop)
                    n = 0
                    extent = 0
                    for (q in 0:qos_stop)
                        k = key[q]
                        if (k.element_2 != FREE)
                            n = add(n, 1)
                            extent = add(q, 1)
                        end
                    end
                    resize(perm, n)
                    r = 0
                    for (q in 0:qos_stop)
                        k = key[q]
                        if (k.element_2 != FREE)
                            perm[r] = to_pos(q)
                            r = add(r, 1)
                        end
                    end
                    """,
                    **locals(),
                )
            ),
        )

        # Order the entries for reading: perm[ptr[p]:ptr[p + 1]] lists parent
        # p's child positions by index.
        def lt(a, b):
            a_p, a_i, b_p, b_i = (
                field
                for q in (a, b)
                for field in hl.key_fields(hl.expr("key[to_len(q)]", q=q))
            )
            return hl.expr(
                "bor(a_p < b_p, band(a_p == b_p, a_i < b_i))",
                a_p=a_p,
                a_i=a_i,
                b_p=b_p,
                b_i=b_i,
            )

        lower_heapsort(ctx, hl.bindings["perm"], n, lt)
        exec_block(
            ctx,
            asm.Block(
                hl.parse(
                    """
                    resize(ptr, add(pos_stop, 1))
                    for (p in 0:pos_stop + 1)
                        ptr[p] = zero_pos
                    end
                    for (r in 0:n)
                        k = key[to_len(perm[r])]
                        p = add(to_len(k.element_0), 1)
                        ptr[p] = add(ptr[p], one_pos)
                    end
                    for (p in 0:pos_stop)
                        ptr[add(p, 1)] = add(ptr[add(p, 1)], ptr[p])
                    end
                    resize(key, extent)
                    """,
                    **locals(),
                )
            ),
        )
        self._lower_count_buckets(ctx, hl, n)
        exec_block(
            ctx,
            asm.Block(
                hl.parse(
                    """
                    lvl.qos_stop = to_pos(extent)
                    lvl.free_head = no_pos
                    lvl.frozen = true
                    """,
                    extent=extent,
                )
            ),
        )
        return self.lvl_t.level_lower_freeze(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), op, extent
        )

    def _lower_count_buckets(self, ctx, hl, n):
        """
        Checkpoint the frozen bucket counts every B ranks of perm, as
        `sparse_hash_count_buckets!`: walking perm backward, each block starts
        from the one after it.
        """
        m, jj, block, b, r, r_stop = (
            hl.var(name) for name in ("m", "jj", "block", "b", "r", "r_stop")
        )
        k = hl.key_var()
        hashing, x = hl.record_hash(k)
        count = (
            *hl.parse("k = key[to_len(perm[r])]", k=k, r=r),
            *hashing,
            *hl.parse(
                """
                bucket = add(block, b_x)
                tbl_count[bucket] = add(tbl_count[bucket], 1)
                """,
                bucket=hl.var("bucket"),
                block=block,
                b_x=hl.subtable(x),
            ),
        )
        exec_block(
            ctx,
            asm.Block(
                hl.parse(
                    """
                    m = add(fld(add(n, sub(B, 1)), B), 1)
                    resize(tbl_count, mul(m, B))
                    for (b in 0:B)
                        tbl_count[add(mul(sub(m, 1), B), b)] = 0
                    end
                    """,
                    m=m,
                    n=n,
                    b=hl.var("b"),
                )
            ),
        )
        ctx.exec(
            asm.ForLoop(
                jj,
                asm.Literal(np.intp(0)),
                hl.expr("sub(m, 1)", m=m),
                asm.Block(
                    (
                        *hl.parse(
                            """
                            block = mul(sub(sub(m, 2), jj), B)
                            for (b in 0:B)
                                tbl_count[add(block, b)] = after
                            end
                            r_stop = min(add(block, B), n)
                            """,
                            block=block,
                            m=m,
                            jj=jj,
                            b=b,
                            after=hl.expr(
                                "tbl_count[add(add(block, B), b)]", block=block, b=b
                            ),
                            r_stop=r_stop,
                            n=n,
                        ),
                        asm.ForLoop(r, block, r_stop, asm.Block(count)),
                    )
                ),
            )
        )

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(
            "SparseHashLevelFType does not support level_lower_increment."
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "SparseHashLevelFType does not support level_lower_unwrap."
        )

    def level_unfurl(
        self, ctx, fiber: ntn.Fiber, ext, mode: ntn.AccessMode, proto, pos
    ):
        match mode:
            case ntn.Update():
                return self.level_unfurl_update(ctx, fiber, ext, mode, proto, pos)
        level = fiber.lvl
        lvl = ctx(level)
        hl = _SparseHashLowering(self, ctx, lvl)
        p_t, d_t = self.position_type, self.dimension_type
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
        bindings = {
            "r": hl.var("r"),
            "r_stop": hl.var("r_stop"),
            "i_stop": hl.var("i_stop", d_t),
            "i_last": hl.var("i_last", d_t),
            "k": hl.key_var(),
            "pos": pos,
            "dimension": hl.expr("to_idx(lvl.dimension)"),
            "no_idx": asm.Literal(d_t(_NO_POS)),
        }
        r, i_stop, i_last = bindings["r"], bindings["i_stop"], bindings["i_last"]

        def stepper_preamble(ctx):
            return asm.Block(
                hl.parse(
                    """
                    if (r < r_stop)
                        k = key[to_len(perm[r])]
                        i_stop = k.element_1
                    else
                        i_stop = dimension
                    end
                    """,
                    **bindings,
                )
            )

        def thunk_preamble(ctx, idx):
            return asm.Block(
                hl.parse(
                    """
                    r = to_len(ptr[to_len(pos)])
                    r_stop = to_len(ptr[add(to_len(pos), 1)])
                    i_stop = dimension
                    i_last = no_idx
                    if (r < r_stop)
                        k = key[to_len(perm[r])]
                        i_stop = k.element_1
                        k = key[to_len(perm[sub(r_stop, 1)])]
                        i_last = k.element_1
                    end
                    """,
                    **bindings,
                )
            )

        def seek_fn(ctx, ext):
            start = hl.expr("to_idx(start)", start=ctx.ctx(ext.get_start()))
            return asm.Block(
                (
                    *self._lower_scansearch(hl, bindings, start),
                    *stepper_preamble(ctx).bodies,
                )
            )

        def chunk_tail_fn(ctx, idx):
            pos_2 = asm.Variable(ctx.freshen(idx, f"_pos_{self.ndim - 1}"), p_t)
            ctx.exec(asm.Assign(pos_2, hl.expr("perm[r]", r=r)))
            return lplt.Run(
                ntn.Fiber(ntn.Child(level), ntn.Value(pos_2, p_t), (*fiber.idxs, idx))
            )

        return lplt.Thunk(
            preamble=thunk_preamble,
            body=lambda ctx, ext: lplt.Sequence(
                head=lambda ctx, idx: lplt.Stepper(
                    preamble=stepper_preamble,
                    stop=lambda ctx: ntn.Variable(i_stop.name, d_t),
                    chunk=lplt.Sequence(
                        head=lambda ctx, idx: lplt.Run(full),
                        split=lambda ctx, ext: ntn.Variable(i_stop.name, d_t),
                        tail=chunk_tail_fn,
                    ),
                    next=lambda ctx: asm.Assign(r, hl.expr("add(r, 1)", r=r)),
                    seek=seek_fn,
                ),
                split=lambda ctx, idx: ntn.Call(
                    ntn.L(ffuncs.add),
                    (ntn.Variable(i_last.name, d_t), ext.get_unit()),
                ),
                tail=lambda ctx, idx: lplt.Run(full),
            ),
        )

    def _lower_scansearch(self, hl, bindings, x):
        """
        Advance the rank `r` to the first one before `r_stop` whose index is at
        least `x`, galloping as `sparse_hash_scansearch`.
        """
        return hl.parse(
            """
            if (r < r_stop)
                k = key[to_len(perm[r])]
                if (k.element_1 < x)
                    hi = sub(r_stop, 1)
                    d = 1
                    p = r
                    going = true
                    while (going)
                        going = false
                        if (p < hi)
                            k = key[to_len(perm[p])]
                            if (k.element_1 < x)
                                d = add(d, d)
                                p = add(p, d)
                                going = true
                            end
                        end
                    end
                    lo = sub(p, d)
                    hi = add(min(p, hi), 1)
                    while (lo < sub(hi, 1))
                        m = add(lo, fld(sub(hi, lo), 2))
                        k = key[to_len(perm[m])]
                        if (k.element_1 < x)
                            lo = m
                        else
                            hi = m
                        end
                    end
                    r = hi
                end
            end
            """,
            **bindings,
            **{name: hl.var(name) for name in ("d", "p", "lo", "hi", "m")},
            going=hl.var("going", ftypes.bool_),
            x=x,
        )

    def level_unfurl_update(self, ctx, fiber, ext, mode, proto, pos):
        lvl = ctx(fiber.lvl)
        child = asm.GetAttr(lvl, asm.Literal("lvl"))
        p_t, d_t = self.position_type, self.dimension_type

        def lookup(ctx, idx):
            hl = _SparseHashLowering(self, ctx, lvl)
            p, i, qos = hl.var("p", p_t), hl.var("i", d_t), hl.var("qos", p_t)
            b, old, q_stop, k = (
                hl.var("b"),
                hl.var("old"),
                hl.var("q_stop"),
                hl.key_var(),
            )
            shared = hl.var("shared", ftype(np.bool_))
            dirty = ntn.Variable(ctx.freshen("dirty"), ftypes.bool_)
            dirty_asm = ctx.ctx(dirty)
            hashing, x = hl.key_hash(p, i)
            finding, h, y = hl.find(p, i, x)
            names = locals()

            def prepare(ctx, _):
                grow = ctx.ctx.block()
                exec_block(
                    grow,
                    asm.Block(
                        hl.parse(
                            """
                            old = n_key
                            q_stop = max(mul(2, n_key), add(to_len(qos), 1))
                            resize(key, q_stop)
                            """,
                            **names,
                        )
                    ),
                )
                self.lvl_t.level_lower_assemble(grow, child, old, q_stop)
                # A new key gets a child position and is inserted tentatively
                # before its child is written.
                insert = (
                    *hl.parse(
                        """
                        tbl_count[b] = add(tbl_count[b], 1)
                        if (lvl.free_head == no_pos)
                            qos = lvl.qos_stop
                            lvl.qos_stop = add(lvl.qos_stop, one_pos)
                        else
                            qos = lvl.free_head
                            k = key[to_len(qos)]
                            lvl.free_head = k.element_0
                        end
                        """,
                        **names,
                    ),
                    asm.If(
                        hl.expr("to_len(qos) >= n_key", qos=qos),
                        asm.Block(tuple(grow.emit())),
                    ),
                    *hl.parse(
                        """
                        key[to_len(qos)] = make_tuple(p, i, ONE_U8)
                        tbl_ctrl[h] = ctrl
                        tbl[h] = qos
                        """,
                        **names,
                        ctrl=hl.ctrl(y),
                    ),
                )
                # Later writers join a tentative entry by its stable child
                # position. A retained entry needs no release.
                join = hl.parse(
                    """
                    qos = tbl[h]
                    k = key[to_len(qos)]
                    shared = k.element_2 != RETAINED
                    """,
                    **names,
                )
                if ctx.mode.safe or ctx.mode.debug:
                    join += (
                        asm.If(
                            shared,
                            asm.Assert(hl.expr("k.element_2 < MAX_PENDING", k=k)),
                        ),
                    )
                join += hl.parse(
                    """
                    if (shared)
                        key[to_len(qos)] = joined
                    end
                    """,
                    **names,
                    joined=hl.restate(k, hl.expr("add(k.element_2, ONE_U8)", k=k)),
                )
                return asm.Block(
                    (
                        *hl.parse(
                            """
                            p = to_pos(pos)
                            i = to_idx(idx)
                            """,
                            p=p,
                            i=i,
                            pos=pos,
                            idx=ctx.ctx(idx),
                        ),
                        *hashing,
                        asm.Assign(b, hl.subtable(x)),
                        # Grow before probing, so the probed slot stays valid if
                        # the key turns out to be new.
                        asm.If(
                            hl.expr(
                                "mul(mul(2, B), add(tbl_count[b], 1)) > n_ctrl", b=b
                            ),
                            asm.Block(
                                hl.resize_table(
                                    hl.expr("max(mul(2, n_ctrl), mul(4, B))")
                                )
                            ),
                        ),
                        *finding,
                        asm.Assign(shared, asm.Literal(np.bool_(True))),
                        asm.Assign(qos, hl.bindings["zero_pos"]),
                        asm.IfElse(
                            hl.expr("tbl_ctrl[h] == EMPTY", h=h),
                            asm.Block(insert),
                            asm.Block(join),
                        ),
                        asm.Assign(dirty_asm, asm.Literal(False)),
                    )
                )

            def finish(ctx, _):
                # The first dirty writer retains the entry. Otherwise, the last
                # writer backward-shift deletes it and frees its child position.
                k_2 = hl.key_var()
                p_2, i_2 = hl.key_fields(k_2)
                deleting, h_2, _ = hl.find(p_2, i_2, x)
                release = (
                    *deleting,
                    *hl.delete(h_2, x),
                    *hl.parse(
                        """
                        key[to_len(qos)] = make_tuple(lvl.free_head, k.element_1, FREE)
                        tbl_count[b] = sub(tbl_count[b], 1)
                        lvl.free_head = qos
                        """,
                        qos=qos,
                        k=k_2,
                        b=b,
                    ),
                )
                retain, unshare = (
                    hl.parse(
                        "key[to_len(qos)] = record",
                        qos=qos,
                        record=hl.restate(k_2, hl.expr(state, k=k_2)),
                    )
                    for state in ("RETAINED", "sub(k.element_2, ONE_U8)")
                )
                stmts = asm.If(
                    shared,
                    asm.Block(
                        (
                            *hl.parse("k = key[to_len(qos)]", k=k_2, qos=qos),
                            asm.If(
                                hl.expr("k.element_2 != RETAINED", k=k_2),
                                asm.IfElse(
                                    dirty_asm,
                                    asm.Block(retain),
                                    asm.IfElse(
                                        hl.expr("k.element_2 > ONE_U8", k=k_2),
                                        asm.Block(unshare),
                                        asm.Block(release),
                                    ),
                                ),
                            ),
                        )
                    ),
                )
                match fiber:
                    case ntn.HollowFiber(dirty=parent_dirty):
                        return asm.Block(
                            (
                                asm.If(
                                    dirty_asm,
                                    asm.Assign(
                                        ctx.ctx(parent_dirty), asm.Literal(True)
                                    ),
                                ),
                                stmts,
                            )
                        )
                return stmts

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

        return lplt.Lookup(lookup)

    def from_fields(
        self,
        lvl,
        dimension,
        subtables,
        pos_multiplier,
        ptr,
        tbl_ctrl,
        tbl,
        key,
        perm,
        tbl_count,
        qos_stop=None,
        free_head=_NO_POS,
        frozen=True,
    ) -> "SparseHashLevel":
        return SparseHashLevel(
            lvl,
            dimension,
            ptr,
            tbl_ctrl,
            tbl,
            key,
            perm,
            tbl_count,
            subtables=subtables,
            pos_multiplier=pos_multiplier,
            qos_stop=qos_stop,
            free_head=free_head,
            frozen=frozen,
        )


def sparse_hash(lvl_t, dimension_type=None):
    return SparseHashLevelFType(lvl_t, dimension_type or ftypes.intp)


@dataclass(init=False)
class SparseHashLevel(SingleDimensionLevel):
    """Hash slots map to child positions with separate ``(parent, index, state)`` keys.

    Positions and indices are zero-based. Key states are free (0), pending
    writer counts (1–127), or retained (128). The odd position multiplier must
    be preserved with the table's buffers. Frozen, ``perm[ptr[p]:ptr[p + 1]]``
    lists parent ``p``'s child positions by index. While assembling,
    ``key[:qos_stop]`` holds the child records, and ``free_head`` starts the
    chain of free child positions, linked through their parent fields.
    """

    lvl: Level
    dimension: np.integer
    ptr: Any | None
    tbl_ctrl: Any | None
    tbl: Any | None
    key: Any | None
    perm: Any | None
    tbl_count: Any | None
    subtables: np.intp
    pos_multiplier: np.uintp
    qos_stop: Any
    free_head: Any
    frozen: bool

    def __init__(
        self,
        lvl: Level,
        dimension: np.integer,
        ptr: Any | None = None,
        tbl_ctrl: Any | None = None,
        tbl: Any | None = None,
        key: Any | None = None,
        perm: Any | None = None,
        tbl_count: Any | None = None,
        *,
        subtables: int = 1,
        pos_multiplier: int | np.integer = _POS_MULTIPLIER,
        qos_stop: Any = None,
        free_head: Any = _NO_POS,
        frozen: bool = True,
    ) -> None:
        self.lvl = lvl
        self.dimension = dimension
        self.ptr = ptr
        self.tbl_ctrl = tbl_ctrl
        self.tbl = tbl
        self.key = key
        self.perm = perm
        self.tbl_count = tbl_count
        if not isinstance(subtables, (int, np.integer)):
            raise ValueError(
                "SparseHashLevel subtables must be a positive power of two"
            )
        self.subtables = np.intp(subtables)
        if (
            not isinstance(pos_multiplier, (int, np.integer))
            or not int(pos_multiplier) & 1
        ):
            raise ValueError(
                "SparseHashLevel position multiplier must be an odd integer"
            )
        self.pos_multiplier = np.uintp(pos_multiplier)
        self.qos_stop = qos_stop
        self.free_head = free_head
        self.frozen = bool(frozen)
        self.__post_init__()

    def __post_init__(self) -> None:
        subtables = int(self.subtables)
        if subtables < 1 or subtables & (subtables - 1) != 0:
            raise ValueError(
                "SparseHashLevel subtables must be a positive power of two"
            )
        if self.ptr is None:
            self.ptr = self.lvl.buffer_factory(self.lvl.position_type)(1)
        if self.tbl_ctrl is None:
            self.tbl_ctrl = self.lvl.buffer_factory(ftypes.uint8)(0)
        if self.tbl is None:
            self.tbl = self.lvl.buffer_factory(self.lvl.position_type)(0)
        if self.key is None:
            key_entry_type = TupleFType.from_tuple(
                (self.lvl.position_type, ftype(self.dimension), ftypes.uint8)
            )
            self.key = self.lvl.buffer_factory(key_entry_type)(0)
        if self.perm is None:
            self.perm = self.lvl.buffer_factory(self.lvl.position_type)(0)
        if self.tbl_count is None:
            self.tbl_count = self.lvl.buffer_factory(ftypes.intp)(self.subtables)
        p_t = self.lvl.position_type
        self.qos_stop = p_t(
            self.key.length() if self.qos_stop is None else self.qos_stop
        )
        self.free_head = p_t(self.free_head)

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    @property
    def ftype(self) -> SparseHashLevelFType:
        return SparseHashLevelFType(
            self.lvl.ftype,
            ftype(self.dimension),
        )  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self) -> str:
        return f"SparseHashLevel(lvl={self.lvl}, dim={self.dimension})"
