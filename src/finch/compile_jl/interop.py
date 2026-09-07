from __future__ import annotations

import weakref
from contextlib import suppress
from typing import Any, cast

import numpy as np

from finch.algebra import ftype
from finch.codegen import NumpyBuffer, NumpyBufferFType
from finch.finch_assembly import Buffer
from finch.tensor import (
    BufferizedNDArray,
    DenseLevel,
    ElementLevel,
    FiberTensor,
    Level,
    Scalar,
    SparseByteMapLevel,
    SparseCOOLevel,
    SparseHashLevel,
    SparseListLevel,
    element,
)
from finch.tensor.np_wrapper import NumPyWrapper

from . import types as jl_dtypes
from .buffer import MinusOneBuffer
from .julia import jc, jl

_OBJECT_OWNERS: dict[int, tuple[Any, ...]] = {}
_EXPORTED_BUFFER_OWNERS: dict[int, tuple[Any, ...]] = {}


def _owners(*owners: Any) -> tuple[Any, ...]:
    return tuple(owner for owner in owners if owner is not None)


def _forget_owner(
    key: int | None,
    pointers: tuple[int, ...],
    owners: tuple[Any, ...],
) -> None:
    if key is not None:
        _OBJECT_OWNERS.pop(key, None)
    for pointer in pointers:
        if _EXPORTED_BUFFER_OWNERS.get(pointer) is owners:
            _EXPORTED_BUFFER_OWNERS.pop(pointer, None)


def _track_python_owner(
    jl_obj: Any,
    *owners: Any,
    pointers: tuple[int, ...] = (),
) -> Any:
    owner_tuple = _owners(*owners)
    if not owner_tuple:
        return jl_obj

    pointer_tuple = tuple(int(pointer) for pointer in pointers)
    for pointer in pointer_tuple:
        _EXPORTED_BUFFER_OWNERS[pointer] = owner_tuple

    key = None
    try:
        jl_obj._finch_python_owners = owner_tuple
    except (AttributeError, TypeError):
        key = id(jl_obj)
        _OBJECT_OWNERS[key] = owner_tuple

    with suppress(TypeError):
        weakref.finalize(jl_obj, _forget_owner, key, pointer_tuple, owner_tuple)
    return jl_obj


def _exported_python_owner(pointer: int) -> tuple[Any, ...] | None:
    return _EXPORTED_BUFFER_OWNERS.get(int(pointer))


def _track_python_buffer_owner(buffer: NumpyBuffer, *owners: Any) -> NumpyBuffer:
    owner_tuple = _owners(*owners)
    if not owner_tuple:
        return buffer

    key = id(buffer)
    _OBJECT_OWNERS[key] = owner_tuple
    with suppress(TypeError):
        weakref.finalize(buffer, _OBJECT_OWNERS.pop, key, None)
    return buffer


def _python_buffer_owner(buffer: NumpyBuffer) -> tuple[Any, ...] | None:
    return _OBJECT_OWNERS.get(id(buffer))


def is_julia_obj(obj: Any) -> bool:
    return isinstance(obj, jc.AnyValue)


def _as_julia_scalar(val):
    if isinstance(val, np.bool_):
        return val.item()
    return val


def _buffer_to_jl(buffer: Buffer, *, offset: int = 0):
    if isinstance(buffer, MinusOneBuffer):
        if offset != 1:
            raise ValueError("MinusOneBuffer can only be unwrapped with offset=1")
        return _buffer_to_jl(buffer.data)
    if isinstance(buffer, NumpyBuffer):
        return jl_dtypes.to_jl_vector(
            buffer.ftype.element_type,
            buffer.arr,
            offset=offset,
            owner_tracker=_track_python_owner,
        )
    raise ValueError(f"Unsupported buffer type: {type(buffer)}")


def _plus_one_buffer_to_jl(buffer: Buffer):
    if isinstance(buffer, MinusOneBuffer):
        return _buffer_to_jl(buffer.data)
    return jl.Finch.PlusOneVector(_buffer_to_jl(buffer))


def level_to_jl(level: Level, pin_fill: bool = False):
    """Convert a level to its Julia counterpart. With `pin_fill`, the leaf
    fill is forced to a zero of its dtype -- see `zero_dynamic_fills` in
    `compile_jl.compiler` for why this backend does that."""
    match level:
        case ElementLevel():
            fill = level.fill_value
            if pin_fill:
                fill = ftype(fill)(0)
            return jl.ElementLevel(
                _as_julia_scalar(fill),
                _buffer_to_jl(level.val),
            )
        case DenseLevel(lvl=lvl, dimension=dimension):
            return jl.DenseLevel(level_to_jl(lvl, pin_fill), int(dimension))
        case SparseListLevel(lvl=lvl, dimension=dimension, ptr=ptr, idx=idx):
            if ptr is None or idx is None:
                raise ValueError("SparseListLevel must have ptr and idx buffers")
            return jl.SparseListLevel(
                level_to_jl(lvl, pin_fill),
                int(dimension),
                _plus_one_buffer_to_jl(cast(Buffer, ptr)),
                _plus_one_buffer_to_jl(cast(Buffer, idx)),
            )
        case SparseByteMapLevel(
            lvl=lvl, dimension=dimension, ptr=ptr, tbl=tbl, srt=srt
        ):
            if ptr is None or tbl is None or srt is None:
                raise ValueError(
                    "SparseByteMapLevel must have ptr, tbl, and srt buffers"
                )
            return jl.SparseByteMapLevel(
                level_to_jl(lvl, pin_fill),
                int(dimension),
                _plus_one_buffer_to_jl(cast(Buffer, ptr)),
                _buffer_to_jl(cast(Buffer, tbl)),
                _plus_one_buffer_to_jl(cast(Buffer, srt)),
            )
        case SparseCOOLevel(lvl=lvl, coo_shape=coo_shape, ptr=ptr, tbl=tbl):
            return jl.SparseCOOLevel(
                level_to_jl(lvl, pin_fill),
                tuple(int(dim) for dim in coo_shape),
                _plus_one_buffer_to_jl(ptr),
                tuple(_plus_one_buffer_to_jl(idx) for idx in tbl),
            )
        case SparseHashLevel(
            lvl=lvl,
            dimension=dimension,
            ptr=ptr,
            tbl_ctrl=tbl_ctrl,
            tbl=tbl,
            pool=pool,
            perm=perm,
            subtables=subtables,
            single_writer=single_writer,
        ):
            if (
                ptr is None
                or tbl_ctrl is None
                or tbl is None
                or pool is None
                or perm is None
            ):
                raise ValueError(
                    "SparseHashLevel must have ptr, tbl_ctrl, tbl, pool, and perm "
                    "buffers"
                )
            dimension = _as_julia_scalar(np.asarray(dimension).item())
            constructor = jl.SparseHashLevel[(jl.typeof(dimension), single_writer)]
            return constructor(
                level_to_jl(lvl, pin_fill),
                dimension,
                int(subtables),
                _plus_one_buffer_to_jl(cast(Buffer, ptr)),
                _buffer_to_jl(cast(Buffer, tbl_ctrl)),
                _buffer_to_jl(cast(Buffer, tbl), offset=1),
                _buffer_to_jl(cast(Buffer, pool)),
                _plus_one_buffer_to_jl(cast(Buffer, perm)),
            )
        case _:
            raise ValueError(f"Unsupported Finch level type: {type(level)}")


def _jl_array_to_python(v, *, dtype=None) -> NumpyBuffer:
    raw = np.asarray(v)
    if dtype is not None and raw.dtype != np.dtype(dtype):
        raise ValueError(f"Cannot avoid a copy converting Julia {raw.dtype} to {dtype}")
    owner = _exported_python_owner(raw.ctypes.data)
    if owner is not None:
        return _track_python_buffer_owner(NumpyBuffer(raw), *owner)
    return _track_python_buffer_owner(NumpyBuffer(raw), v)


def _jl_index_buffer_to_python(v) -> Buffer:
    """Converts a Julia index/position buffer, adjusting Julia's 1-based indexing
    to Python's 0-based indexing without copying."""
    if jl.isa(v, jl.Finch.PlusOneVector):
        return _jl_array_to_python(v.data)
    return MinusOneBuffer(_jl_array_to_python(v))


def _jl_buffer_to_python(v) -> NumpyBuffer:
    return _jl_array_to_python(v)


def _jl_tuple_buffer_to_python(v, n_fields: int, *, offset: int = 0) -> Buffer:
    """See _jl_index_buffer_to_python: offset is represented by a wrapper."""
    if offset not in (0, 1):
        raise ValueError("Only offset=0 or offset=1 can be represented without a copy")

    raw_buffer = _jl_buffer_to_python(v)
    raw = raw_buffer.arr

    src_fields = raw.dtype.fields
    assert src_fields is not None
    src_names = [f"f{i}" for i in range(n_fields)]
    dtype = np.dtype(
        {
            "names": [f"element_{i}" for i in range(n_fields)],
            "formats": [src_fields[name][0] for name in src_names],
            "offsets": [src_fields[name][1] for name in src_names],
            "itemsize": raw.dtype.itemsize,
        }
    )
    buffer = _track_python_buffer_owner(NumpyBuffer(raw.view(dtype)), raw_buffer)
    if offset:
        return MinusOneBuffer(buffer)
    return buffer


def jl_level_to_python(jl_lvl) -> Level:
    if jl.isa(jl_lvl, jl.Finch.ElementLevel):
        fill_value = jl.Finch.level_fill_value(jl.typeof(jl_lvl))
        val = _jl_buffer_to_python(jl_lvl.val)
        elem_ftype = element(
            jl_dtypes.to_fl_dtype(val.arr.dtype)(fill_value),
            jl_dtypes.to_fl_dtype(val.arr.dtype),
            jl_dtypes.int_,
            NumpyBufferFType,
        )
        return ElementLevel(elem_ftype, val)

    if jl.isa(jl_lvl, jl.Finch.DenseLevel):
        return DenseLevel(
            jl_level_to_python(jl_lvl.lvl),
            np.intp(int(jl_lvl.shape)),
        )

    if jl.isa(jl_lvl, jl.Finch.SparseListLevel):
        return SparseListLevel(
            jl_level_to_python(jl_lvl.lvl),
            np.intp(int(jl_lvl.shape)),
            _jl_index_buffer_to_python(jl_lvl.ptr),
            _jl_index_buffer_to_python(jl_lvl.idx),
        )

    if jl.isa(jl_lvl, jl.Finch.SparseByteMapLevel):
        return SparseByteMapLevel(
            jl_level_to_python(jl_lvl.lvl),
            np.intp(int(jl_lvl.shape)),
            _jl_index_buffer_to_python(jl_lvl.ptr),
            _jl_buffer_to_python(jl_lvl.tbl),
            _jl_index_buffer_to_python(jl_lvl.srt),
        )

    if jl.isa(jl_lvl, jl.Finch.SparseCOOLevel):
        coo_shape = tuple(np.intp(int(s)) for s in jl_lvl.shape)
        tbl = tuple(_jl_index_buffer_to_python(idx) for idx in jl_lvl.tbl)
        return SparseCOOLevel(
            jl_level_to_python(jl_lvl.lvl),
            coo_shape,
            _jl_index_buffer_to_python(jl_lvl.ptr),
            tbl,
        )

    if jl.isa(jl_lvl, jl.Finch.SparseHashLevel):
        single_writer = bool(jl.typeof(jl_lvl).parameters[1])
        return SparseHashLevel(
            jl_level_to_python(jl_lvl.lvl),
            np.intp(int(jl_lvl.shape)),
            _jl_index_buffer_to_python(jl_lvl.ptr),
            _jl_buffer_to_python(jl_lvl.tbl_ctrl),
            _jl_tuple_buffer_to_python(jl_lvl.tbl, 3, offset=1),
            _jl_buffer_to_python(jl_lvl.pool),
            _jl_index_buffer_to_python(jl_lvl.perm),
            subtables=int(jl_lvl.subtables),
            single_writer=single_writer,
        )

    raise ValueError(f"Unsupported Julia level type for recovery: {jl.typeof(jl_lvl)}")


def _ndarray_to_jl_tensor(
    arr: np.ndarray,
    fill_value: Any,
    *,
    copy: bool = False,
):
    if copy:
        arr = arr.copy() if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
    elif not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    jl_type = jl_dtypes._fl_dtype_to_jl()[ftype(arr.dtype)]
    buf = jl.wrap_numpy_ptr(arr.ctypes.data, arr.size, jl_type)
    fill = _as_julia_scalar(np.asarray(fill_value, dtype=arr.dtype)[()])
    lvl = jl.ElementLevel(fill, buf)
    for dim in reversed(arr.shape):
        lvl = jl.DenseLevel(lvl, int(dim))
    tensor = jl.Tensor(lvl)
    return _track_python_owner(tensor, arr, pointers=(arr.ctypes.data,))


def tensor_to_jl(obj, pin_fill: bool = False):
    """Convert a tensor to its Julia counterpart. With `pin_fill`, fills are
    forced to a zero of their dtype so the argument types line up with a
    kernel compiled under `zero_dynamic_fills`."""
    if is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor):
        return obj
    if isinstance(obj, FiberTensor):
        if obj.pos != 0:
            raise ValueError("Only root-position FiberTensor objects can use Julia")
        return _track_python_owner(jl.Tensor(level_to_jl(obj.lvl, pin_fill)), obj)
    if isinstance(obj, BufferizedNDArray):
        fill = ftype(obj.fill_value)(0) if pin_fill else obj.fill_value
        return _ndarray_to_jl_tensor(obj.to_numpy(), fill, copy=False)
    if isinstance(obj, NumPyWrapper):
        fill = ftype(obj.fill_value)(0) if pin_fill else obj.fill_value
        return _ndarray_to_jl_tensor(obj._data, fill, copy=False)
    if isinstance(obj, Scalar):
        return scalar_to_jl(obj.val)
    if isinstance(obj, np.ndarray):
        fill = np.asarray(0, dtype=obj.dtype)[()]
        return _ndarray_to_jl_tensor(obj, fill, copy=False)
    if isinstance(obj, np.generic):
        return scalar_to_jl(obj.item())
    if np.isscalar(obj):
        return scalar_to_jl(obj)
    if hasattr(obj, "val"):
        return scalar_to_jl(obj.val)
    raise ValueError(f"Unsupported Julia backend argument type: {type(obj)}")


def scalar_to_jl(val):
    if isinstance(val, np.generic):
        val = val.item()
    buf = np.asarray([val])
    jl_type = jl_dtypes._fl_dtype_to_jl()[ftype(buf.dtype)]
    tensor = jl.Tensor(
        jl.ElementLevel(
            _as_julia_scalar(buf.item()),
            jl.wrap_numpy_ptr(buf.ctypes.data, buf.size, jl_type),
        )
    )
    return _track_python_owner(tensor, buf, pointers=(buf.ctypes.data,))


def jl_tensor_to_python(obj):
    if not (is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor)):
        return obj
    return FiberTensor(jl_level_to_python(obj.lvl))
