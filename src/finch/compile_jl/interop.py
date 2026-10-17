from __future__ import annotations

from functools import lru_cache
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
from finch.tensor.patterns import (
    ChunkMaskTensor,
    EyeTensor,
    FillTensor,
    LowerTriangleTensor,
    OddEvenMergeSortLowerMaskTensor,
    OddEvenMergeSortPartnerMaskTensor,
    OneHotMaskTensor,
    PairCarryTensor,
    PairSumTensor,
    ParityMaskTensor,
    PatternTensor,
    RandomMaskTensor,
    RepeatTensor,
    ReshapeMaskTensor,
    ReverseTensor,
    RollTensor,
    SplitMaskTensor,
    UpperTriangleTensor,
)

from . import types as jl_dtypes
from .buffer import MinusOneBuffer
from .julia import jc, jl


def is_julia_obj(obj: Any) -> bool:
    return isinstance(obj, jc.AnyValue)


def _as_julia_scalar(val):
    if isinstance(val, np.bool_):
        return val.item()
    return val


def _buffer_to_jl(
    buffer: Buffer, *, offset: int = 0, index_fields: tuple[int, ...] | None = None
):
    if isinstance(buffer, MinusOneBuffer):
        if offset != 1 or index_fields != buffer.index_fields:
            raise ValueError(
                "MinusOneBuffer can only be unwrapped with matching offsets"
            )
        result = _buffer_to_jl(buffer.data, offset=offset, index_fields=index_fields)
        result.shifted = False
        return result
    if isinstance(buffer, NumpyBuffer):
        return jl_dtypes.to_jl_vector(
            buffer.ftype.element_type,
            buffer.arr,
            offset=offset,
            index_fields=index_fields,
        )
    raise ValueError(f"Unsupported buffer type: {type(buffer)}")


def level_to_jl(level: Level, pin_fill: bool = False):
    """Convert a level to its Julia counterpart, optionally pinning its fill."""
    match level:
        case ElementLevel():
            fill = level.fill_value
            if pin_fill:
                fill = jl_dtypes._zero_value(ftype(fill))
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
                _buffer_to_jl(cast(Buffer, ptr), offset=1),
                _buffer_to_jl(cast(Buffer, idx), offset=1),
            )
        case SparseByteMapLevel(
            lvl=lvl, dimension=dimension, ptr=ptr, tbl=tbl, srt=srt
        ):
            if ptr is None or tbl is None or srt is None:
                raise ValueError(
                    "SparseByteMapLevel must have ptr, tbl, and srt buffers"
                )
            constructor = jl.SparseByteMapLevel[jl_dtypes.to_jl_type(ftype(dimension))]
            return constructor(
                level_to_jl(lvl, pin_fill),
                int(dimension),
                _buffer_to_jl(cast(Buffer, ptr), offset=1),
                _buffer_to_jl(cast(Buffer, tbl)),
                _buffer_to_jl(cast(Buffer, srt), offset=1),
            )
        case SparseCOOLevel(lvl=lvl, coo_shape=coo_shape, ptr=ptr, tbl=tbl):
            assert isinstance(tbl, tuple)
            # Julia's COO lists its dims fastest first, Python's slowest first
            return jl.SparseCOOLevel(
                level_to_jl(lvl, pin_fill),
                tuple(int(dim) for dim in reversed(coo_shape)),
                _buffer_to_jl(ptr, offset=1),
                tuple(_buffer_to_jl(idx, offset=1) for idx in reversed(tbl)),
            )
        case SparseHashLevel(
            lvl=lvl,
            dimension=dimension,
            ptr=ptr,
            tbl_ctrl=tbl_ctrl,
            tbl=tbl,
            key=key,
            perm=perm,
            tbl_count=tbl_count,
            subtables=subtables,
            pos_multiplier=pos_multiplier,
        ):
            if (
                ptr is None
                or tbl_ctrl is None
                or tbl is None
                or key is None
                or perm is None
                or tbl_count is None
            ):
                raise ValueError(
                    "SparseHashLevel must have ptr, tbl_ctrl, tbl, key, perm, "
                    "and tbl_count buffers"
                )
            constructor = jl.SparseHashLevel[jl_dtypes.to_jl_type(ftype(dimension))]
            return constructor(
                level_to_jl(lvl, pin_fill),
                int(dimension),
                int(subtables),
                pos_multiplier,
                _buffer_to_jl(cast(Buffer, ptr), offset=1),
                _buffer_to_jl(cast(Buffer, tbl_ctrl)),
                _buffer_to_jl(cast(Buffer, tbl), offset=1),
                _buffer_to_jl(cast(Buffer, key), offset=1, index_fields=(0, 1)),
                _buffer_to_jl(cast(Buffer, perm), offset=1),
                _buffer_to_jl(cast(Buffer, tbl_count)),
            )
        case _:
            raise ValueError(f"Unsupported Finch level type: {type(level)}")


def _jl_array_to_python_no_copy(v):
    """Return a NumPy view of a Julia array without copying its data."""
    if jl.isa(v, jl.FinchInterop.Buffer):
        data, layout = jl.FinchInterop.storage(v)
        dtype = _dtype_with_layout(_jl_numpy_dtype(jl.eltype(v)), layout)
        raw = np.array(data, copy=False).view(dtype)
        if not v.writable:
            raw.flags.writeable = False
        return raw
    return v.to_numpy(copy=False)


@lru_cache
def _jl_numpy_dtype(type_):
    return jl.Vector[type_]().to_numpy(copy=False).dtype


def _dtype_with_layout(dtype, layout):
    if dtype.names is None:
        return dtype
    fields = tuple(layout[1])
    return np.dtype(
        {
            "names": dtype.names,
            "formats": [
                _dtype_with_layout(dtype.fields[name][0], field[1])
                for name, field in zip(dtype.names, fields, strict=True)
            ],
            "offsets": [field[0] for field in fields],
            "itemsize": layout[0],
        }
    )


def _jl_index_buffer_to_python(v) -> MinusOneBuffer | NumpyBuffer:
    """
    Converts a Julia index/position buffer, adjusting Julia's 1-based indexing
    to Python's 0-based indexing without copying.
    """
    if jl.isa(v, jl.Finch.PlusOneVector):
        raw = _jl_array_to_python_no_copy(v.data)
        return NumpyBuffer(raw)
    raw = _jl_array_to_python_no_copy(v)
    if jl.isa(v, jl.FinchInterop.Buffer) and v.shifted:
        return NumpyBuffer(raw)
    return MinusOneBuffer(NumpyBuffer(raw))


def _jl_buffer_to_python(v) -> NumpyBuffer:
    return NumpyBuffer(_jl_array_to_python_no_copy(v))


def _jl_tuple_buffer_to_python(
    v, n_fields: int, *, index_fields: tuple[int, ...] = ()
) -> Buffer:
    """Convert a Julia tuple buffer while preserving its Julia-owned data."""
    raw = _jl_array_to_python_no_copy(v)

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
    buffer = NumpyBuffer(raw.view(dtype))
    if index_fields and not (jl.isa(v, jl.FinchInterop.Buffer) and v.shifted):
        return MinusOneBuffer(buffer, index_fields=index_fields)
    return buffer


def jl_level_to_python(jl_lvl) -> Level:
    if jl.isa(jl_lvl, jl.Finch.ElementLevel):
        fill_value = jl.Finch.level_fill_value(jl.typeof(jl_lvl))
        val = _jl_array_to_python_no_copy(jl_lvl.val)
        elem_ftype = element(
            jl_dtypes.to_fl_dtype(val.dtype)(fill_value),
            jl_dtypes.to_fl_dtype(val.dtype),
            jl_dtypes.int_,
            NumpyBufferFType,
        )
        return ElementLevel(elem_ftype, NumpyBuffer(val))

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
            jl_dtypes.to_fl_dtype(jl.first(jl.typeof(jl_lvl).parameters))(jl_lvl.shape),
            _jl_index_buffer_to_python(jl_lvl.ptr),
            _jl_buffer_to_python(jl_lvl.tbl),
            _jl_index_buffer_to_python(jl_lvl.srt),
        )

    if jl.isa(jl_lvl, jl.Finch.SparseCOOLevel):
        coo_shape = tuple(np.intp(int(s)) for s in reversed(jl_lvl.shape))
        tbl = tuple(_jl_index_buffer_to_python(idx) for idx in reversed(jl_lvl.tbl))
        return SparseCOOLevel(
            jl_level_to_python(jl_lvl.lvl),
            coo_shape,
            _jl_index_buffer_to_python(jl_lvl.ptr),
            tbl,
        )

    if jl.isa(jl_lvl, jl.Finch.SparseHashLevel):
        return SparseHashLevel(
            jl_level_to_python(jl_lvl.lvl),
            jl_dtypes.to_fl_dtype(jl.first(jl.typeof(jl_lvl).parameters))(jl_lvl.shape),
            _jl_index_buffer_to_python(jl_lvl.ptr),
            _jl_buffer_to_python(jl_lvl.tbl_ctrl),
            _jl_index_buffer_to_python(jl_lvl.tbl),
            _jl_tuple_buffer_to_python(jl_lvl.key, 3, index_fields=(0, 1)),
            _jl_index_buffer_to_python(jl_lvl.perm),
            _jl_buffer_to_python(jl_lvl.tbl_count),
            subtables=int(jl_lvl.subtables),
            pos_multiplier=np.uintp(jl_lvl.pos_multiplier),
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

    buf = jl_dtypes.to_jl_vector(ftype(arr.dtype), arr.reshape(-1))
    fill = _as_julia_scalar(np.asarray(fill_value, dtype=arr.dtype)[()])
    lvl = jl.ElementLevel(fill, buf)
    for dim in reversed(arr.shape):
        lvl = jl.DenseLevel(lvl, int(dim))
    return jl.Tensor(lvl)


def _pattern_tensor_to_jl(obj: PatternTensor):
    # Python masks are row-major and Finch's are column-major, so each mask is
    # the Finch mask whose axes are reversed, like other interop tensors. The
    # row-major traversal of a Python mask then visits each Finch column first,
    # which is the efficient traversal of Finch's masks. Offsets shift the
    # second Julia axis, `offset(mask, 0, k)[i, j] = mask[i, j + k]`.
    swizzle = False
    match obj:
        case EyeTensor():
            mask = jl.Finch.diagmask
            if obj._k:
                mask = jl.Finch.offset(mask, 0, int(obj._k))
        case UpperTriangleTensor():
            mask = jl.Finch.offset(jl.Finch.lotrimask, 0, int(obj._k))
        case LowerTriangleTensor():
            mask = jl.Finch.offset(jl.Finch.uptrimask, 0, int(obj._k))
        case PairSumTensor():
            mask = jl.Finch.repeatmask(2)
        case PairCarryTensor():
            mask = jl.Finch.offset(jl.Finch.pairsummask, 0, -1)
        case ReverseTensor():
            mask = jl.Finch.reversemask(int(obj.shape[1]))
        case RollTensor():
            mask = jl.Finch.rollmask(int(obj.shape[1]), -int(obj._k))
        case RepeatTensor():
            mask = jl.Finch.repeatmask(int(obj._k))
        case ChunkMaskTensor():
            mask = jl.Finch.chunkmask(int(obj.shape[1]), int(obj._b))
        case SplitMaskTensor():
            mask = jl.Finch.splitmask(int(obj.shape[1]), int(obj.shape[0]))
        case RandomMaskTensor():
            mask = jl.Finch.randommask(
                tuple(int(dim) for dim in reversed(obj.shape)),
                obj._p,
                seed=jl.UInt64(obj._seed),
            )
        case OddEvenMergeSortPartnerMaskTensor():
            mask = jl.Finch.oddevenmergesortpartnermask(
                int(obj.shape[1]), int(obj._p), int(obj._k)
            )
        case OddEvenMergeSortLowerMaskTensor():
            mask = jl.Finch.oddevenmergesortlowermask(
                int(obj.shape[0]), int(obj._p), int(obj._k)
            )
        case OneHotMaskTensor():
            mask = jl.Finch.onehotmask(int(obj._index) + 1)
        case ParityMaskTensor():
            mask = jl.Finch.paritymask(int(obj._parity))
        case ReshapeMaskTensor():
            # Reshape masks have no preferred traversal, but reversing their
            # axes would flatten them in column-major order, so they are
            # swizzled instead.
            mask = jl.Finch.reshapemask(
                tuple(int(dim) for dim in obj._old_shape),
                tuple(int(dim) for dim in obj._new_shape),
            )
            swizzle = True
        case _:
            raise ValueError(f"Unsupported Julia pattern tensor type: {type(obj)}")
    # Masks infer extents by default; retain the Python shape even when the
    # mask is the only input.
    shape = obj.shape if swizzle else tuple(reversed(obj.shape))
    mask = jl.Finch.window(mask, *(jl.Finch.Extent(1, int(dim)) for dim in shape))
    if swizzle and obj.ndim > 1:
        mask = jl.Finch.swizzle(mask, *reversed(range(1, obj.ndim + 1)))
    return mask


def tensor_to_jl(obj, pin_fill: bool = False):
    """Convert a tensor to its Julia counterpart, optionally pinning its fill."""
    if hasattr(obj, "raw_julia_obj"):
        return obj.raw_julia_obj
    if is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor):
        return obj
    if isinstance(obj, FiberTensor):
        if obj.pos != 0:
            raise ValueError("Only root-position FiberTensor objects can use Julia")
        return jl.Tensor(level_to_jl(obj.lvl, pin_fill))
    if isinstance(obj, BufferizedNDArray):
        fill = (
            jl_dtypes._zero_value(ftype(obj.fill_value)) if pin_fill else obj.fill_value
        )
        return _ndarray_to_jl_tensor(obj.to_numpy(), fill, copy=False)
    if isinstance(obj, NumPyWrapper):
        fill = (
            jl_dtypes._zero_value(ftype(obj.fill_value)) if pin_fill else obj.fill_value
        )
        return _ndarray_to_jl_tensor(obj._data, fill, copy=False)
    if isinstance(obj, Scalar):
        return scalar_to_jl(obj.val, pin_fill=pin_fill)
    if isinstance(obj, PatternTensor):
        return _pattern_tensor_to_jl(obj)
    if isinstance(obj, FillTensor):
        # A fill tensor is its fill everywhere, and its extents are passed to
        # kernels separately, so it is passed as a Finch scalar.
        fill = jl_dtypes._zero_value(obj.element_type) if pin_fill else obj.fill_value
        scalar_t = jl.seval(jl_dtypes.scalar_type_str(fill, obj.element_type))
        return scalar_t(_as_julia_scalar(obj.fill_value))
    if isinstance(obj, np.ndarray):
        fill = np.asarray(0, dtype=obj.dtype)[()]
        return _ndarray_to_jl_tensor(obj, fill, copy=False)
    if np.isscalar(obj):
        return scalar_to_jl(obj, pin_fill=pin_fill)
    raise ValueError(f"Unsupported Julia backend argument type: {type(obj)}")


def scalar_to_jl(val, pin_fill: bool = False):
    fill = jl_dtypes._zero_value(ftype(val)) if pin_fill else val
    buf = np.asarray([val])
    return jl.Tensor(
        jl.ElementLevel(_as_julia_scalar(fill), jl_dtypes.to_jl_vector(ftype(val), buf))
    )


def jl_tensor_to_python(obj):
    if not (is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor)):
        return obj
    lvl = jl_level_to_python(obj.lvl)
    if isinstance(lvl, ElementLevel):
        # A julia tensor whose root is the element level has no dimensions, so
        # it is a scalar rather than a zero dimensional fiber tensor.
        val = lvl.val
        return Scalar(
            val.load(0) if val.length() else lvl.fill_value,
            fill_value=lvl.fill_value,
        )
    return FiberTensor(lvl)


class JuliaBufferContext:
    """Keep Julia-owned tensor buffers alive across kernel invocations."""

    def __init__(self):
        self._tensors: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    @staticmethod
    def _cache_key(obj):
        # defer() can create a fresh wrapper around the same NumPy
        # allocation, so wrapper identity alone would miss reuse.
        if isinstance(obj, BufferizedNDArray):
            arr = obj.to_numpy()
            pointer = arr.__array_interface__["data"][0]
            return ("numpy", pointer, arr.shape, arr.strides, arr.dtype.str)
        if isinstance(obj, NumPyWrapper):
            arr = obj._data
            pointer = arr.__array_interface__["data"][0]
            return ("numpy", pointer, arr.shape, arr.strides, arr.dtype.str)
        # FiberTensors reuse their ids so we restrict cache keys to id.
        return ("object", id(obj))

    def tensor_to_jl(self, obj, *, pin_fill: bool = False):
        key = (self._cache_key(obj), pin_fill)
        cached = self._tensors.get(key)
        if cached is not None:
            return cached[1]

        jl_obj = tensor_to_jl(obj, pin_fill=pin_fill)
        self._tensors[key] = (obj, jl_obj)
        return jl_obj

    def tensor_to_python(self, obj):
        result = jl_tensor_to_python(obj)
        if isinstance(result, FiberTensor):
            self._tensors[self._cache_key(result)] = (result, obj)
        return result

    def close(self):
        self._tensors.clear()
