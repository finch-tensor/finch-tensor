from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from finch.algebra import ftype
from finch.tensor import BufferizedNDArray, FiberTensor, Scalar
from finch.tensor.np_wrapper import NumPyWrapper
from finch.tensor.patterns import FillTensor

from .interop import (
    _ndarray_to_jl_tensor,
    is_julia_obj,
    jl_level_to_python,
    level_to_jl,
    scalar_to_jl,
)
from .julia import jl


class FinchJLRuntime(ABC):
    """Runtime services used by the Julia compiler and its kernels."""

    @abstractmethod
    def get_cached_kernel(self, key): ...

    @abstractmethod
    def cache_kernel(self, key, kernel): ...

    def tensor_to_jl(self, obj, *, pin_fill: bool = False):
        """Create a Julia representation without runtime-specific caching."""
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

    def tensor_to_python(self, obj):
        """Create a Python representation without runtime-specific caching."""
        if not (is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor)):
            return obj
        return FiberTensor(jl_level_to_python(obj.lvl))


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

        jl_obj = super().tensor_to_jl(obj, pin_fill=pin_fill)
        self._tensors[key] = (obj, jl_obj)
        return jl_obj

    def tensor_to_python(self, obj):
        result = super().tensor_to_python(obj)
        if isinstance(result, FiberTensor):
            self._tensors[self._tensor_cache_key(result)] = (result, obj)
        return result

    def close(self):
        self._kernels.clear()
        self._tensors.clear()
