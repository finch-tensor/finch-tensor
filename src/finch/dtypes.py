import builtins

import numpy as np

import finchlite as fl
from finchlite.algebra.ftypes import FType

int8: FType = fl.int8
int16: FType = fl.int16
int32: FType = fl.int32
int64: FType = fl.int64
int_: FType = fl.intp
uint8: FType = fl.uint8
uint16: FType = fl.uint16
uint32: FType = fl.uint32
uint64: FType = fl.uint64
uint: FType = uint32 if np.uintp == np.uint32 else uint64
float16: FType = fl.float16
float32: FType = fl.float32
float64: FType = fl.float64
complex64: FType = fl.complex64
complex128: FType = fl.complex128
bool: FType = fl.bool

number = builtins.int | builtins.float | builtins.complex | builtins.bool

finfo = fl.finfo
iinfo = fl.iinfo

jl_to_np_dtype = {
    int_: int_.dtype,
    int8: int8.dtype,
    int16: int16.dtype,
    int32: int32.dtype,
    int64: int64.dtype,
    uint: uint.dtype,
    uint8: uint8.dtype,
    uint16: uint16.dtype,
    uint32: uint32.dtype,
    uint64: uint64.dtype,
    float16: float16.dtype,
    float32: float32.dtype,
    float64: float64.dtype,
    complex64: complex64.dtype,
    complex128: complex128.dtype,
    bool: bool.dtype,
    None: None,
}


def can_cast(from_, to, /) -> builtins.bool:
    if not isinstance(from_, FType) and hasattr(from_, "dtype"):
        from_ = from_.dtype
    return np.can_cast(jl_to_np_dtype[from_], jl_to_np_dtype[to])
