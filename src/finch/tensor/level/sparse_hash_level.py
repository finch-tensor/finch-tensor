from dataclasses import dataclass
from secrets import randbits
from typing import Any

import numpy as np

from finch.algebra import FType, ImmutableStructFType, TupleFType, ffuncs, ftype, ftypes

from .level import Level, LevelFType, SingleDimensionLevel, SingleDimensionLevelFType

_LOWERING_ERROR = "SparseHashLevelFType lowering is not implemented."
_POS_MULTIPLIER = np.uintp(randbits(np.dtype(np.uintp).itemsize * 8) | 1)


@dataclass(unsafe_hash=True)
class SparseHashLevelFType(SingleDimensionLevelFType, ImmutableStructFType):
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
    def ptr_type(self) -> FType:
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

    def level_asm_unpack(self, ctx, var_n, val):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_asm_repack(self, ctx, lvl_fields):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_dim(self, ctx, obj, r):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_freeze(self, ctx, lvl, op, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_thaw(self, ctx, lvl, op, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_unfurl(self, ctx, lvl, ext, mode, proto, pos):
        raise NotImplementedError(_LOWERING_ERROR)

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
        )


def sparse_hash(lvl_t, dimension_type=None):
    return SparseHashLevelFType(lvl_t, dimension_type or ftypes.intp)


@dataclass(init=False)
class SparseHashLevel(SingleDimensionLevel):
    """Hash slots map to child positions with separate ``(parent, index, state)`` keys.

    Positions and indices are zero-based. Key states are free (0), pending
    writer counts (1–127), or retained (128). The odd position multiplier must
    be preserved with the table's buffers.
    """

    lvl: Level
    dimension: np.integer
    ptr: Any | None
    tbl_ctrl: Any | None
    tbl: Any | None
    key: Any | None
    perm: Any | None
    tbl_count: Any | None
    subtables: int
    pos_multiplier: np.uintp

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
        self.subtables = int(subtables)
        if (
            not isinstance(pos_multiplier, (int, np.integer))
            or not int(pos_multiplier) & 1
        ):
            raise ValueError(
                "SparseHashLevel position multiplier must be an odd integer"
            )
        self.pos_multiplier = np.uintp(pos_multiplier)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.subtables < 1 or self.subtables & (self.subtables - 1) != 0:
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
