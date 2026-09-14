from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from finch.tensor import BufferizedNDArray, FiberTensor
from finch.tensor.np_wrapper import NumPyWrapper

from .interop import jl_tensor_to_python, tensor_to_jl


class FinchJLRuntime(ABC):
    """Runtime services used by the Julia compiler and its kernels."""

    @abstractmethod
    def get_cached_kernel(self, key): ...

    @abstractmethod
    def cache_kernel(self, key, kernel): ...

    @abstractmethod
    def tensor_to_jl(self, obj, *, pin_fill: bool = False): ...

    @abstractmethod
    def tensor_to_python(self, obj): ...


class DefaultFinchJLRuntime(FinchJLRuntime):
    """Default runtime with kernel and Julia tensor-wrapper caches."""

    def __init__(self):
        self._kernels: dict[Any, Any] = {}
        self._tensors: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def get_cached_kernel(self, key):
        return self._kernels.get(key)

    def cache_kernel(self, key, kernel):
        self._kernels[key] = kernel

    @staticmethod
    def _tensor_cache_key(obj):
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
        key = self._tensor_cache_key(obj)
        cached = self._tensors.get(key)
        if cached is not None:
            return cached[1]

        jl_obj = tensor_to_jl(obj, pin_fill=pin_fill)
        self._tensors[key] = (obj, jl_obj)
        return jl_obj

    def tensor_to_python(self, obj):
        result = jl_tensor_to_python(obj)
        if isinstance(result, FiberTensor):
            self._tensors[self._tensor_cache_key(result)] = (result, obj)
        return result

    def close(self):
        self._kernels.clear()
        self._tensors.clear()
