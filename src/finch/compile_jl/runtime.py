from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from finch.algebra import Tensor, TensorFType
from finch.tensor import BufferizedNDArray
from finch.tensor.np_wrapper import NumPyWrapper

from .analyze import reset_argument_positions, returned_argument_positions
from .interop import jl_tensor_to_python, python_tensor_to_jl
from .julia import jl


class FinchJLRuntime(ABC):
    """Runtime services used by the Julia compiler and its kernels."""

    @abstractmethod
    def get_cached_kernel(self, key): ...

    @abstractmethod
    def cache_kernel(self, key, kernel): ...

    @abstractmethod
    def kernel_call(self, func_name, args): ...


@dataclass(frozen=True)
class _TensorCacheKey:
    pin_fill: bool
    kind: str
    identity: int
    shape: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    dtype: str = ""


@dataclass
class _BufferLease:
    raw: Any
    key: _BufferPoolKey


@dataclass(frozen=True)
class _KernalMetadata:
    reset_positions: frozenset[int]
    returned_positions: tuple[int, ...]


@dataclass(frozen=True)
class _BufferPoolKey:
    ftype: str
    shape: tuple[int, ...]
    pin_fill: bool


class JuliaOwnedTensor(Tensor):
    """A Finch tensor handle for storage owned by the Julia runtime."""

    def __init__(
        self,
        ftype: TensorFType,
        shape: tuple[int, ...],
        runtime: DefaultFinchJLRuntime,
        raw_julia_obj: Any,
        pin_fill: bool,
        lease: _BufferLease | None = None,
    ) -> None:
        self._ftype = ftype
        self._shape = shape
        self._runtime = runtime
        self._raw_julia_obj = raw_julia_obj
        self._pin_fill = pin_fill
        self._lease = lease

    def __del__(self) -> None:
        self._runtime.release(self)

    @property
    def ftype(self) -> TensorFType:
        return self._ftype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def raw_julia_obj(self) -> Any:
        return self._raw_julia_obj

    def _as_tensor(self) -> Tensor:
        return jl_tensor_to_python(self.raw_julia_obj)

    def item(self):
        if not self._shape:
            values = self.raw_julia_obj.lvl.val.to_numpy(copy=False)
            return values[0].item()
        return self._as_tensor().item()

    def to_numpy(self):
        return self._as_tensor().to_numpy()

    def to_scipy(self):
        return self._as_tensor().to_scipy()


class DefaultFinchJLRuntime(FinchJLRuntime):
    """Julia runtime with cached conversions and reusable result buffers."""

    def __init__(self) -> None:
        self._kernels: dict[Any, Any] = {}
        self._kernels_by_name: dict[str, Any] = {}
        self._kernel_metadata: dict[str, _KernalMetadata] = {}
        self._owned_by_buffer: dict[_TensorCacheKey, JuliaOwnedTensor] = {}
        self._source_finalizers: dict[_TensorCacheKey, weakref.finalize] = {}
        self.free_pool = _BufferPool()

    def get_cached_kernel(self, key):
        return self._kernels.get(key)

    def cache_kernel(self, key, kernel):
        self._kernels[key] = kernel
        self._kernels_by_name[kernel.func_name] = kernel

    def _kernel_metadata_for(self, kernel: Any) -> _KernalMetadata:
        metadata = self._kernel_metadata.get(kernel.func_name)
        if metadata is None:
            metadata = _KernalMetadata(
                reset_argument_positions(kernel.finch_program),
                returned_argument_positions(kernel.finch_program),
            )
            self._kernel_metadata[kernel.func_name] = metadata
        return metadata

    def kernel_call(self, func_name, args):
        kernel = self._kernels_by_name[func_name]
        metadata = self._kernel_metadata_for(kernel)

        # Lease Julia buffers only for resettable compiler-created outputs.
        owned_args: list[JuliaOwnedTensor] = []
        raw_args: list[Any] = []
        for position, tensor in enumerate(args):
            pin_fill = position in kernel.dynamic_args
            if position in metadata.reset_positions and not isinstance(
                tensor, JuliaOwnedTensor
            ):
                ftype = tensor.ftype
                lease = self.free_pool.acquire(ftype, tensor.shape, pin_fill=pin_fill)
                owned = JuliaOwnedTensor(
                    ftype,
                    lease.key.shape,
                    self,
                    lease.raw,
                    lease.key.pin_fill,
                    lease=lease,
                )
            else:
                owned = self._to_julia_owned_tensor(tensor, pin_fill)
            owned_args.append(owned)
            raw_args.append(owned.raw_julia_obj)

        # Julia returns the formal arguments that contain the computed results.
        getattr(jl, func_name)(*raw_args)

        # Associate returned buffers with their Python ownership handles.
        return tuple(owned_args[position] for position in metadata.returned_positions)

    def _tensor_to_jl(self, tensor: Tensor, pin_fill: bool = False):
        return self._to_julia_owned_tensor(tensor, pin_fill).raw_julia_obj

    def _to_julia_owned_tensor(
        self, tensor: Tensor, pin_fill: bool = False
    ) -> JuliaOwnedTensor:
        if isinstance(tensor, JuliaOwnedTensor):
            if tensor._pin_fill == pin_fill:
                return tensor
            source = tensor._as_tensor()
            self._tensor_to_jl(source, pin_fill=pin_fill)
            return self._owned_by_buffer[
                self._tensor_cache_key(source, pin_fill=pin_fill)
            ]
        key = self._tensor_cache_key(tensor, pin_fill=pin_fill)
        owned = self._owned_by_buffer.get(key)
        if owned is not None:
            return owned

        raw = python_tensor_to_jl(tensor, pin_fill=pin_fill)
        owned = JuliaOwnedTensor(
            tensor.ftype,
            tuple(int(dimension) for dimension in tensor.shape),
            self,
            raw,
            pin_fill,
        )
        self._owned_by_buffer[key] = owned
        self._source_finalizers[key] = weakref.finalize(
            tensor, self._drop_source_buffer, key
        )
        return owned

    def _drop_source_buffer(self, key: _TensorCacheKey) -> None:
        """Drop cached Julia state once its Python source tensor is collected.

        Cached conversions retain Julia-owned buffers, so their entries must not
        outlive the Python tensor that owns the backing storage.
        """
        self._source_finalizers.pop(key, None)
        self._owned_by_buffer.pop(key, None)

    @staticmethod
    def _tensor_cache_key(tensor: Tensor, pin_fill: bool = False) -> _TensorCacheKey:
        if isinstance(tensor, BufferizedNDArray):
            array = tensor.to_numpy()
            return _TensorCacheKey(
                pin_fill,
                "numpy",
                array.__array_interface__["data"][0],
                tuple(int(dimension) for dimension in array.shape),
                tuple(int(stride) for stride in array.strides),
                array.dtype.str,
            )
        if isinstance(tensor, NumPyWrapper):
            array = tensor._data
            return _TensorCacheKey(
                pin_fill,
                "numpy",
                array.__array_interface__["data"][0],
                tuple(int(dimension) for dimension in array.shape),
                tuple(int(stride) for stride in array.strides),
                array.dtype.str,
            )
        return _TensorCacheKey(pin_fill, "object", id(tensor))

    def release(self, tensor: JuliaOwnedTensor) -> None:
        if tensor._lease is not None:
            self.free_pool.release_lease(tensor._lease)

    def close(self) -> None:
        for finalizer in self._source_finalizers.values():
            finalizer.detach()
        self._kernels.clear()
        self._kernels_by_name.clear()
        self._kernel_metadata.clear()
        self._owned_by_buffer.clear()
        self._source_finalizers.clear()


class _BufferPool:
    def __init__(self) -> None:
        self._free: dict[_BufferPoolKey, dict[int, _BufferLease]] = defaultdict(dict)

    def acquire(
        self,
        ftype: TensorFType,
        shape: tuple[int, ...],
        pin_fill: bool,
    ) -> _BufferLease:
        key = _BufferPoolKey(
            repr(ftype),
            tuple(int(dimension) for dimension in shape),
            pin_fill,
        )
        if self._free[key]:
            return self._free[key].popitem()[1]
        tensor = ftype.construct(shape)
        return _BufferLease(
            python_tensor_to_jl(tensor, pin_fill=pin_fill),
            key,
        )

    def release_lease(self, lease: _BufferLease) -> None:
        free_leases = self._free[lease.key]
        free_leases.setdefault(id(lease), lease)
