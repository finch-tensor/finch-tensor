from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

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
from finch.tensor.patterns import FillTensor

from . import types as jl_dtypes
from .buffer import MinusOneBuffer
from .julia import jc, jl


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


def _jl_array_to_python_no_copy(v):
    """Return a NumPy view of a Julia array without copying its data."""
    return v.to_numpy(copy=False)


def _jl_index_buffer_to_python(v) -> Buffer:
    """
    Converts a Julia index/position buffer, adjusting Julia's 1-based indexing
    to Python's 0-based indexing without copying.
    """
    if jl.isa(v, jl.Finch.PlusOneVector):
        raw = _jl_array_to_python_no_copy(v.data).astype(np.intp, copy=False)
        return NumpyBuffer(raw)
    raw = _jl_array_to_python_no_copy(v).astype(np.intp, copy=False)
    return MinusOneBuffer(NumpyBuffer(raw))


def _jl_buffer_to_python(v) -> NumpyBuffer:
    return NumpyBuffer(_jl_array_to_python_no_copy(v))


def _jl_tuple_buffer_to_python(v, n_fields: int, *, offset: int = 0) -> Buffer:
    """Convert a Julia tuple buffer while preserving its Julia-owned data."""
    if offset not in (0, 1):
        raise ValueError("Only offset=0 or offset=1 can be represented without a copy")
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
    if offset:
        return MinusOneBuffer(buffer)
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

    buf = jl_dtypes.to_jl_vector(ftype(arr.dtype), arr.reshape(-1))
    fill = _as_julia_scalar(np.asarray(fill_value, dtype=arr.dtype)[()])
    lvl = jl.ElementLevel(fill, buf)
    for dim in reversed(arr.shape):
        lvl = jl.DenseLevel(lvl, int(dim))
    return jl.Tensor(lvl)


def tensor_to_jl(obj, pin_fill: bool = False):
    """Convert a tensor to its Julia counterpart. With `pin_fill`, fills are
    forced to a zero of their dtype so the argument types line up with a
    kernel compiled under `zero_dynamic_fills`."""
    if is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor):
        return obj
    if isinstance(obj, FiberTensor):
        if obj.pos != 0:
            raise ValueError("Only root-position FiberTensor objects can use Julia")
        return jl.Tensor(level_to_jl(obj.lvl, pin_fill))
    if isinstance(obj, BufferizedNDArray):
        fill = ftype(obj.fill_value)(0) if pin_fill else obj.fill_value
        return _ndarray_to_jl_tensor(obj.to_numpy(), fill, copy=False)
    if isinstance(obj, NumPyWrapper):
        fill = ftype(obj.fill_value)(0) if pin_fill else obj.fill_value
        return _ndarray_to_jl_tensor(obj._data, fill, copy=False)
    if isinstance(obj, Scalar):
        return scalar_to_jl(obj.val, pin_fill=pin_fill)
    if isinstance(obj, FillTensor):
        lvl = jl.PatternLevel()
        for dim in reversed(obj.shape):
            lvl = jl.DenseLevel(lvl, int(dim))
        return jl.Tensor(lvl)
    if isinstance(obj, np.ndarray):
        fill = np.asarray(0, dtype=obj.dtype)[()]
        return _ndarray_to_jl_tensor(obj, fill, copy=False)
    if np.isscalar(obj):
        return scalar_to_jl(obj, pin_fill=pin_fill)
    raise ValueError(f"Unsupported Julia backend argument type: {type(obj)}")


def scalar_to_jl(val, pin_fill: bool = False):
    fill = ftype(val)(0) if pin_fill else val
    buf = np.asarray([val])
    return jl.Tensor(jl.ElementLevel(_as_julia_scalar(fill), jl.Vector(buf)))


def jl_tensor_to_python(obj):
    if not (is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor)):
        return obj
    return FiberTensor(jl_level_to_python(obj.lvl))


@dataclass(frozen=True)
class JuliaKernelArgs:
    type_names: tuple[str | None, ...]
    dynamic_positions: tuple[int, ...]
    reset_positions: frozenset[int]
    return_positions: tuple[int, ...]

    @property
    def cache_key(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Return the metadata that selects a compatible compiled kernel."""
        return (
            tuple(type_name for type_name in self.type_names if type_name is not None),
            self.dynamic_positions,
        )


@dataclass(frozen=True, init=False)
class JuliaBufferGroup:
    type_name: str
    shape: tuple[int, ...]

    def __init__(self, obj, type_name: str | None = None):
        if type_name is None:
            type_name = str(jl.string(jl.typeof(obj)))
            shape = jl.size(obj)
        else:
            shape = obj.shape
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "shape", tuple(int(dim) for dim in shape))


class BufferID:
    @staticmethod
    def from_object(obj) -> BufferID:
        """Return the identifier for a Python object backing a Julia buffer."""
        # defer() can create a fresh wrapper around the same NumPy
        # allocation, so wrapper identity alone would miss reuse.
        if isinstance(obj, BufferizedNDArray):
            arr = obj.to_numpy()
            pointer = arr.__array_interface__["data"][0]
            return NumpyBufferID(pointer, arr.shape, arr.strides, arr.dtype.str)
        if isinstance(obj, NumPyWrapper):
            arr = obj._data
            pointer = arr.__array_interface__["data"][0]
            return NumpyBufferID(pointer, arr.shape, arr.strides, arr.dtype.str)
        # FiberTensors reuse their ids so we restrict cache keys to id.
        return ObjectBufferID(id(obj))


@dataclass(frozen=True)
class NumpyBufferID(BufferID):
    pointer: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ObjectBufferID(BufferID):
    object_id: int


@dataclass
class _JuliaBufferRecord:
    tensor: Any
    group: JuliaBufferGroup = field(init=False)
    py_wrapper: FiberTensor = field(init=False)
    owner_key: BufferID | None = None

    def __post_init__(self):
        self.group = JuliaBufferGroup(self.tensor)
        self.py_wrapper = jl_tensor_to_python(self.tensor)


@dataclass
class JuliaFreeBufferPool:
    records: dict[JuliaBufferGroup, list[_JuliaBufferRecord]] = field(
        default_factory=dict
    )

    def add(self, record: _JuliaBufferRecord) -> None:
        self.records.setdefault(record.group, []).append(record)

    def remove(self, group: JuliaBufferGroup) -> _JuliaBufferRecord | None:
        records = self.records.get(group)
        if not records:
            return None
        return records.pop()

    def clear(self) -> None:
        self.records.clear()


class ResolvedJuliaArguments(NamedTuple):
    julia_args: list[Any]
    argument_keys: tuple[BufferID, ...]


class JuliaBufferContext:
    """Own and reuse Julia tensor buffers across kernel invocations."""

    def __init__(self):
        self._owned_records: dict[BufferID, _JuliaBufferRecord] = {}
        self._result_records: dict[BufferID, _JuliaBufferRecord] = {}
        self._julia_id_to_record_map: dict[int, _JuliaBufferRecord] = {}
        self._free_pool = JuliaFreeBufferPool()


    @staticmethod
    def _is_poolable(obj) -> bool:
        """Return whether an object is a non-scalar Julia tensor that can be reused."""
        return (
            is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor) and len(jl.size(obj)) > 0
        )

    def _get_or_create_record(self, obj) -> _JuliaBufferRecord | None:
        """Get or register the pool record for a reusable Julia tensor."""
        if not self._is_poolable(obj):
            return None
        object_id = int(jl.objectid(obj))
        record = self._julia_id_to_record_map.get(object_id)
        if record is None:
            record = _JuliaBufferRecord(obj)
            self._julia_id_to_record_map[object_id] = record
        return record

    def _assign_owner(self, key: BufferID, record: _JuliaBufferRecord) -> None:
        """Assign a Python argument key as the active owner of a buffer record."""
        if record.owner_key is not None:
            self._owned_records.pop(record.owner_key, None)
        record.owner_key = key
        self._owned_records[key] = record

    def _release_owner(self, key: BufferID) -> _JuliaBufferRecord | None:
        """Remove an active owner mapping without placing its record in a pool."""
        record = self._owned_records.pop(key, None)
        if record is not None:
            record.owner_key = None
        return record

    def _claim_result_record(self, key: BufferID) -> _JuliaBufferRecord | None:
        """Move a returned record into the active-owner pool for an argument."""
        record = self._result_records.pop(key, None)
        if record is not None:
            self._assign_owner(key, record)
            return record
        return None

    def tensor_to_jl(self, obj, *, pin_fill: bool = False):
        """Return an existing Julia tensor mapping or materialize a new one."""
        key = BufferID.from_object(obj)

        # If the tensor is already owned reuse
        record = self._owned_records.get(key)
        if record is not None:
            return record.tensor

        # Otherwise claim the tensor from result records produced by another kernel
        record = self._claim_result_record(key)
        if record is not None:
            return record.tensor

        # Initialize the tensor in julia by copying from python. This is expensive
        # and should only be triggered for the first call of the kernel with said arguments.
        jl_obj = tensor_to_jl(obj, pin_fill=pin_fill)
        record = self._get_or_create_record(jl_obj)
        if record is not None:
            self._assign_owner(key, record)
        return jl_obj

    def tensor_to_python(self, obj):
        """Return a cached Python wrapper for a Julia tensor, creating one if needed."""
        record = self._get_or_create_record(obj)
        if record is None:
            return jl_tensor_to_python(obj)
        if record.owner_key is not None:
            self._release_owner(record.owner_key)
        key = BufferID.from_object(record.py_wrapper)
        self._result_records[key] = record
        return record.py_wrapper

    def resolve_arguments(
        self,
        args,
        *,
        kernel_args: JuliaKernelArgs,
    ) -> ResolvedJuliaArguments:
        """Resolve call arguments and lease compatible free buffers for reset inputs."""

        argument_keys = tuple(BufferID.from_object(arg) for arg in args)
        positions = tuple(
            position
            for position in range(len(args))
            if position not in kernel_args.reset_positions
        ) + tuple(
            position
            for position in range(len(args))
            if position in kernel_args.reset_positions
        )
        julia_args: list[Any] = [None] * len(args)
        for position in positions:
            arg = args[position]
            key = argument_keys[position]
            record = self._owned_records.get(key)
            if record is None:
                record = self._claim_result_record(key)
            if record is not None:
                julia_args[position] = record.tensor
                continue

            type_name = kernel_args.type_names[position]
            if position in kernel_args.reset_positions:
                group = JuliaBufferGroup(arg, type_name)
                record = self._free_pool.remove(group)
                if record is not None:
                    self._assign_owner(key, record)
                    julia_args[position] = record.tensor
                    continue

            julia_args[position] = self.tensor_to_jl(
                arg, pin_fill=position in kernel_args.dynamic_positions
            )
        return ResolvedJuliaArguments(julia_args, argument_keys)

    def release_reset_arguments(
        self,
        argument_keys: tuple[BufferID, ...],
        reset_positions: frozenset[int],
        return_positions: tuple[int, ...],
    ) -> None:
        """Release reset buffers that are not returned by the completed kernel."""
        returned_positions = set(return_positions)
        for position in reset_positions:
            record = self._release_owner(argument_keys[position])
            if record is not None and position not in returned_positions:
                self._free_pool.add(record)

    def close(self):
        """Discard all Python mappings and pooled Julia tensor records."""
        self._owned_records.clear()
        self._julia_id_to_record_map.clear()
        self._free_pool.clear()
        self._result_records.clear()
