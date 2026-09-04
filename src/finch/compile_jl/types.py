import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

import finch as ft
from finch.algebra.fill import DynamicFill, StaticFill
from finch.algebra.ftypes import FDTypeNumpy, FType, TupleFType, ftype
from finch.tensor import DenseLevelFType, ElementLevelFType, LevelFType
from finch.tensor.bufferized_ndarray import BufferizedNDArrayFType
from finch.tensor.fiber_tensor import FiberTensorFType
from finch.tensor.level.sparse_bytemap_level import SparseByteMapLevelFType
from finch.tensor.level.sparse_coo_level import SparseCOOLevelFType
from finch.tensor.level.sparse_hash_level import SparseHashLevelFType
from finch.tensor.level.sparse_list_level import SparseListLevelFType
from finch.tensor.scalar import ScalarFType

from .julia import get_jl, jc

_py_bool = bool

int8: FDTypeNumpy = ft.int8
int16: FDTypeNumpy = ft.int16
int32: FDTypeNumpy = ft.int32
int64: FDTypeNumpy = ft.int64
int_: FDTypeNumpy = ft.intp
uint8: FDTypeNumpy = ft.uint8
uint16: FDTypeNumpy = ft.uint16
uint32: FDTypeNumpy = ft.uint32
uint64: FDTypeNumpy = ft.uint64
uint: FDTypeNumpy = uint32 if np.uintp == np.uint32 else uint64
float16: FDTypeNumpy = ft.float16
float32: FDTypeNumpy = ft.float32
float64: FDTypeNumpy = ft.float64
complex64: FDTypeNumpy = ft.complex64
complex128: FDTypeNumpy = ft.complex128
bool: FDTypeNumpy = ft.bool

finfo = ft.finfo
iinfo = ft.iinfo


class JuliaElementFType(ABC):
    @abstractmethod
    def julia_type(self):
        """
        Return the Julia type used for elements with this ftype.
        """
        ...

    @abstractmethod
    def julia_value(self, value: Any, *, offset: int = 0):
        """
        Convert a Python value with this ftype into a Julia element value.
        """
        ...

    def julia_vector(self, values, *, offset: int = 0):
        jl = get_jl()
        return jc.convert(
            jl.Vector[self.julia_type()],
            [self.julia_value(value, offset=offset) for value in values],
        )


@lru_cache
def _jl_dtype_to_fl() -> dict[Any, FType]:
    jl = get_jl()
    return {
        jl.Int8: int8,
        jl.Int16: int16,
        jl.Int32: int32,
        jl.Int64: int64,
        jl.UInt8: uint8,
        jl.UInt16: uint16,
        jl.UInt32: uint32,
        jl.UInt64: uint64,
        jl.Float16: float16,
        jl.Float32: float32,
        jl.Float64: float64,
        jl.ComplexF32: complex64,
        jl.ComplexF64: complex128,
        jl.Bool: bool,
    }


@lru_cache
def _fl_dtype_to_jl() -> dict[FType, Any]:
    jl = get_jl()
    return {
        **{v: k for k, v in _jl_dtype_to_fl().items()},
        ft.bool_: jl.Bool,
        ft.int_: jl.Int,
        ft.float_: jl.Float64,
        ft.complex_: jl.ComplexF64,
    }


def to_fl_dtype(x) -> FType:
    """Normalize a Julia DataType, numpy dtype/scalar type, Python builtin
    type, or finch FType into the corresponding finch FType."""
    if isinstance(x, FType):
        return x
    try:
        return ft.ftype(x)
    except NotImplementedError:
        pass
    try:
        return _jl_dtype_to_fl()[x]
    except (KeyError, TypeError):
        raise NotImplementedError(f"Cannot convert {x!r} to a Finch dtype") from None


def to_jl_type(T):
    T = to_fl_dtype(T)
    if isinstance(T, JuliaElementFType):
        return T.julia_type()
    if isinstance(T, TupleFType):
        jl = get_jl()
        return jl.Tuple[tuple(to_jl_type(field) for field in T.struct_fieldtypes)]
    try:
        return _fl_dtype_to_jl()[T]
    except KeyError:
        raise NotImplementedError(f"Cannot convert {T!r} to a Julia dtype") from None


def _as_julia_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuple_field(value, index: int, name: str):
    if isinstance(value, np.void) and value.dtype.fields is not None:
        return value[name]
    return value[index]


def to_jl_value(T, value, *, offset: int = 0):
    T = to_fl_dtype(T)
    if isinstance(T, JuliaElementFType):
        return T.julia_value(value, offset=offset)
    if isinstance(T, TupleFType):
        return tuple(
            to_jl_value(
                field_type,
                _tuple_field(value, index, field_name),
                offset=offset,
            )
            for index, (field_name, field_type) in enumerate(T.struct_fields)
        )
    if offset:
        value = value + offset
    return _as_julia_scalar(T(value))


def to_jl_vector(T, values, *, offset: int = 0):
    T = to_fl_dtype(T)
    if isinstance(T, JuliaElementFType):
        return T.julia_vector(values, offset=offset)
    if isinstance(T, TupleFType) or offset:
        jl = get_jl()
        return jc.convert(
            jl.Vector[to_jl_type(T)],
            [to_jl_value(T, value, offset=offset) for value in values],
        )
    return get_jl().Vector(values)


def _julia_literal(value: Any) -> str:
    match value:
        case DynamicFill() as fill:
            value = ftype(fill.value)(0)
        case StaticFill() as fill:
            value = fill.value
    # NOTE: this module shadows the builtin `bool` (see `bool: FDTypeNumpy`
    # above), so `isinstance` must use `_py_bool` (captured before shadowing).
    if isinstance(value, (_py_bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        if math.isinf(value):
            return "-Inf" if value < 0 else "Inf"
        if math.isnan(value):
            return "NaN"
    return str(value)


def _leaf_type_str(T: Any) -> str:
    return str(get_jl().string(to_jl_type(T)))


def _plus_one_ctor_str(elem_type_str: str) -> str:
    return f"Finch.PlusOneVector({elem_type_str}[])"


def _level_constructor_str(level_ftype: LevelFType) -> str:
    """
    Julia source text that constructs a minimal (empty-buffer, but real --
    Finch's virtualize() needs an actual value, not just a type) instance of
    `level_ftype`. Matches interop.py's level_to_jl buffer wrapping exactly
    (e.g. index buffers as PlusOneVector, for the 0-based/1-based indexing
    conversion), so the type this produces at eval time is identical to
    what a real argument would have at call time.
    """
    if isinstance(level_ftype, ElementLevelFType):
        elem_t = _leaf_type_str(level_ftype.element_type)
        fill = _julia_literal(level_ftype.fill_value)
        return f"Finch.ElementLevel({fill}, {elem_t}[])"
    if isinstance(level_ftype, DenseLevelFType):
        # interop.py's level_to_jl does `int(dimension)`, so the "shape"
        # scalar (level type param Ti) is always plain Julia Int (Int64),
        # regardless of the declared dimension_type -- only index/pointer
        # *buffers* actually preserve dimension_type/position_type.
        return f"Finch.DenseLevel({_level_constructor_str(level_ftype.lvl_t)}, 1)"
    if isinstance(level_ftype, SparseListLevelFType):
        pos_t = _leaf_type_str(level_ftype.position_type)
        dim_t = _leaf_type_str(level_ftype.dimension_type)
        return (
            f"Finch.SparseListLevel({_level_constructor_str(level_ftype.lvl_t)}, "
            f"1, {_plus_one_ctor_str(pos_t)}, {_plus_one_ctor_str(dim_t)})"
        )
    if isinstance(level_ftype, SparseByteMapLevelFType):
        pos_t = _leaf_type_str(level_ftype.position_type)
        dim_t = _leaf_type_str(level_ftype.dimension_type)
        return (
            f"Finch.SparseByteMapLevel({_level_constructor_str(level_ftype.lvl_t)}, "
            f"1, {_plus_one_ctor_str(pos_t)}, Bool[], "
            f"{_plus_one_ctor_str(dim_t)})"
        )
    if isinstance(level_ftype, SparseCOOLevelFType):
        pos_t = _leaf_type_str(level_ftype.position_type)
        dim_ts = [
            _leaf_type_str(t)
            for t in level_ftype.coo_shape_tuple_type.struct_fieldtypes
        ]
        dims = ",".join("1" for _ in dim_ts)
        idxs = ",".join(_plus_one_ctor_str(t) for t in dim_ts)
        return (
            f"Finch.SparseCOOLevel{{{level_ftype.coo_ndim}}}("
            f"{_level_constructor_str(level_ftype.lvl_t)}, ({dims},), "
            f"{_plus_one_ctor_str(pos_t)}, ({idxs},))"
        )
    if isinstance(level_ftype, SparseHashLevelFType):
        pos_t = _leaf_type_str(level_ftype.position_type)
        dim_t = _leaf_type_str(level_ftype.dimension_type)
        single_writer = "true" if level_ftype.single_writer else "false"
        tbl_entry_t = f"Tuple{{{pos_t},{dim_t},{pos_t}}}"
        # interop.py derives Ti from `np.asarray(dimension).item()`, which
        # (like DenseLevel above) always yields plain Julia Int (Int64).
        return (
            f"Finch.SparseHashLevel{{Int,{single_writer}}}("
            f"{_level_constructor_str(level_ftype.lvl_t)}, 1, 1, "
            f"{_plus_one_ctor_str(pos_t)}, UInt8[], {tbl_entry_t}[], {pos_t}[], "
            f"{_plus_one_ctor_str(pos_t)})"
        )
    raise NotImplementedError(
        f"ftype_to_jl_constructor_str: unsupported level ftype "
        f"{type(level_ftype).__name__}"
    )


def ftype_to_jl_constructor_str(ftype: FType) -> str:
    """Julia source text constructing a minimal instance of `ftype`, for
    embedding directly in generated kernel source (see compiler.py)."""
    if isinstance(ftype, ScalarFType):
        elem_t = _leaf_type_str(ftype.element_type)
        fill = _julia_literal(ftype.fill_value)
        return f"Finch.Tensor(Finch.ElementLevel({fill}, {elem_t}[]))"
    if isinstance(ftype, FiberTensorFType):
        return f"Finch.Tensor({_level_constructor_str(ftype.lvl_t)})"
    if isinstance(ftype, BufferizedNDArrayFType):
        # Matches interop.py's _ndarray_to_jl_tensor: a plain dense buffer
        # wrapped in `ndim` nested DenseLevels, each with plain Int64 shape
        # (regardless of the ftype's own dimension_type).
        elem_t = _leaf_type_str(ftype.element_type)
        fill = _julia_literal(ftype.fill_value)
        ctor = f"Finch.ElementLevel({fill}, {elem_t}[])"
        for _ in range(ftype.ndim):
            ctor = f"Finch.DenseLevel({ctor}, 1)"
        return f"Finch.Tensor({ctor})"
    raise NotImplementedError(
        f"ftype_to_jl_constructor_str: unsupported ftype kind {type(ftype).__name__}"
    )


_TYPE_STR_CACHE: dict[FType, str] = {}


def ftype_to_jl_type_str(ftype: FType) -> str:
    """Julia type as source text (e.g. for a cache key). Cached per ftype."""
    cached = _TYPE_STR_CACHE.get(ftype)
    if cached is not None:
        return cached
    jl = get_jl()
    ctor = ftype_to_jl_constructor_str(ftype)
    type_str = str(jl.string(jl.typeof(jl.seval(ctor))))
    _TYPE_STR_CACHE[ftype] = type_str
    return type_str
