from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from finch.algebra import Tensor
from finch.tensor import BufferizedNDArray
from finch.tensor.np_wrapper import NumPyWrapper

from .analyze import reset_argument_positions, returned_argument_positions
from .buffer_reuse import JuliaOwnedTensor, _BufferPool
from .interop import tensor_to_jl
from .julia import jl


@dataclass(frozen=True)
class _TensorCacheKey:
    pin_fill: bool
    kind: str
    identity: int
    shape: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    dtype: str = ""


@dataclass(frozen=True)
class _KernalMetadata:
    reset_positions: frozenset[int]
    returned_positions: tuple[int, ...]


class FinchJLRuntime(ABC):
    """Runtime services used by the Julia compiler and its kernels."""

    @abstractmethod
    def get_cached_kernel(self, key): ...

    @abstractmethod
    def cache_kernel(self, key, kernel): ...

    @abstractmethod
    def kernel_call(self, func_name, args): ...


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
                lease = self.free_pool.acquire(
                    ftype,
                    tensor.shape,
                    pin_fill,
                )
                owned = JuliaOwnedTensor(
                    ftype,
                    lease.key.shape,
                    self.release,
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

        raw = tensor_to_jl(tensor, pin_fill=pin_fill)
        owned = JuliaOwnedTensor(
            tensor.ftype,
            tuple(int(dimension) for dimension in tensor.shape),
            self.release,
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
