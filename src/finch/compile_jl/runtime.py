from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from finch.algebra import Tensor, is_dynamic
from finch.tensor import BufferizedNDArray
from finch.tensor.np_wrapper import NumPyWrapper

from .analyze import reset_argument_positions, returned_argument_positions
from .buffer_reuse import JuliaOwnedTensor, _BufferPool
from .interop import tensor_to_jl
from .julia import jl


@dataclass(frozen=True)
class _TranslationCacheKey:
    kind: str
    identity: int | None
    shape: tuple[int, ...] | None
    strides: tuple[int, ...] | None
    dtype: str | None


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
    def kernel_call(self, func_name, kernel, args): ...


class DefaultFinchJLRuntime(FinchJLRuntime):
    """Julia runtime with cached conversions and reusable result buffers."""

    def __init__(self) -> None:
        self._kernels: dict[Any, Any] = {}
        self._kernel_metadata: dict[int, _KernalMetadata] = {}
        self._translated_buffers: dict[_TranslationCacheKey, JuliaOwnedTensor] = {}
        self.free_pool = _BufferPool()

    def get_cached_kernel(self, key):
        return self._kernels.get(key)

    def cache_kernel(self, key, kernel):
        self._kernels[key] = kernel
        self._kernel_metadata[id(kernel)] = _KernalMetadata(
            reset_argument_positions(kernel.finch_program),
            returned_argument_positions(kernel.finch_program),
        )

    def kernel_call(self, func_name, kernel, args):
        metadata = self._kernel_metadata[id(kernel)]

        # Lease Julia buffers only for resettable compiler-created outputs.
        julia_buf_args: list[JuliaOwnedTensor] = []
        for position, tensor in enumerate(args):
            if position in metadata.reset_positions and not isinstance(
                tensor, JuliaOwnedTensor
            ):
                lease = self.free_pool.acquire(
                    tensor.ftype,
                    tensor.shape,
                    (
                        tensor.ftype.fill_value
                        if is_dynamic(tensor.ftype.fill_value)
                        else None
                    ),
                )
                julia_buf = JuliaOwnedTensor(
                    tensor.ftype,
                    lease.key.shape,
                    self.release,
                    lease.raw,
                    lease,
                )
            else:
                julia_buf = self._to_julia_owned_tensor(tensor)
            julia_buf_args.append(julia_buf)

        getattr(jl, func_name)(*(arg.raw_julia_obj for arg in julia_buf_args))

        # Associate returned buffers with their Python ownership handles.
        return tuple(
            julia_buf_args[position] for position in metadata.returned_positions
        )

    def _to_julia_owned_tensor(self, tensor: Tensor) -> JuliaOwnedTensor:
        if isinstance(tensor, JuliaOwnedTensor):
            return tensor
        key = self._translation_cache_key(tensor)
        julia_buf = self._translated_buffers.get(key)
        if julia_buf is not None:
            return julia_buf

        raw = tensor_to_jl(tensor)
        translation_finalizer = weakref.finalize(
            tensor, self._translated_buffers.pop, key, None
        )
        julia_buf = JuliaOwnedTensor(
            tensor.ftype,
            tuple(int(dimension) for dimension in tensor.shape),
            self.release,
            raw,
            translation_finalizer=translation_finalizer,
        )
        self._translated_buffers[key] = julia_buf
        return julia_buf

    @staticmethod
    def _translation_cache_key(tensor: Tensor) -> _TranslationCacheKey:
        if isinstance(tensor, BufferizedNDArray):
            array = tensor.to_numpy()
            return _TranslationCacheKey(
                "numpy",
                array.__array_interface__["data"][0],
                tuple(int(dimension) for dimension in array.shape),
                tuple(int(stride) for stride in array.strides),
                array.dtype.str,
            )
        if isinstance(tensor, NumPyWrapper):
            array = tensor._data
            return _TranslationCacheKey(
                "numpy",
                array.__array_interface__["data"][0],
                tuple(int(dimension) for dimension in array.shape),
                tuple(int(stride) for stride in array.strides),
                array.dtype.str,
            )
        return _TranslationCacheKey("object", id(tensor), None, None, None)

    def release(self, tensor: JuliaOwnedTensor) -> None:
        if tensor._lease is not None:
            self.free_pool.release_lease(tensor._lease)

    def close(self) -> None:
        self._kernels.clear()
        self._kernel_metadata.clear()
        self._translated_buffers.clear()
