from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

import finch as ft
from finch.algebra import TensorFType
from finch.algebra.ftypes import FDTypeNumpy, FType, TupleFType
from finch.tensor import DenseLevelFType, ElementLevelFType, LevelFType

from .julia import get_jl, jc, jl

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


def _prototype(ftype: FType) -> Any:
    if isinstance(ftype, LevelFType):
        return ftype.construct(shape=tuple(1 for _ in range(ftype.ndim)), pos=0)
    if isinstance(ftype, TensorFType):
        return ftype.construct(tuple(1 for _ in range(ftype.ndim)))
    raise NotImplementedError(
        f"ftype_to_jl_type: unsupported ftype kind {type(ftype).__name__}"
    )


def _level_jl_type(level_ftype: LevelFType):
    if isinstance(level_ftype, ElementLevelFType):
        elem_t = to_jl_type(level_ftype.element_type)
        return jl.ElementLevel[
            _as_julia_scalar(level_ftype.fill_value),
            elem_t,
            to_jl_type(level_ftype.position_type),
            jl.Vector[elem_t],
        ]
    if isinstance(level_ftype, DenseLevelFType):
        return jl.DenseLevel[
            to_jl_type(level_ftype.dimension_type),
            _level_jl_type(level_ftype.lvl_t),
        ]
    from .interop import level_to_jl

    return jl.typeof(level_to_jl(_prototype(level_ftype)))


def ftype_to_jl_prototype(ftype: TensorFType):
    from .interop import tensor_to_jl

    return tensor_to_jl(_prototype(ftype))


def ftype_to_jl_type(ftype: FType):
    """Live Julia type object for a TensorFType/LevelFType."""
    if isinstance(ftype, TensorFType):
        return jl.typeof(ftype_to_jl_prototype(ftype))
    if isinstance(ftype, LevelFType):
        return _level_jl_type(ftype)
    raise NotImplementedError(
        f"ftype_to_jl_type: unsupported ftype kind {type(ftype).__name__}"
    )


_TYPE_STR_CACHE: dict[FType, str] = {}


def ftype_to_jl_type_str(ftype: FType) -> str:
    """Julia type as source text (e.g. for a cache key). Cached per ftype."""
    cached = _TYPE_STR_CACHE.get(ftype)
    if cached is not None:
        return cached
    type_str = str(jl.string(ftype_to_jl_type(ftype)))
    _TYPE_STR_CACHE[ftype] = type_str
    return type_str
