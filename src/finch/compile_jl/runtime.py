from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from finch.algebra import Tensor, TensorFType
from finch.algebra.fill import DynamicFill
from finch.tensor import BufferizedNDArray, FiberTensorFType
from finch.tensor.np_wrapper import NumPyWrapper
from finch.tensor.override_tensor import OverrideTensor
from finch.tensor.scalar import Scalar

from .analyze import reset_argument_positions, returned_argument_positions
from .interop import jl_tensor_to_python, tensor_to_jl
from .julia import jl


@dataclass
class _StorageLease:
    raw: Any
    key: _StoragePoolKey


@dataclass(frozen=True)
class _StoragePoolKey:
    ftype: str
    shape: tuple[int, ...]
    pin_fill: bool


class _StoragePool:
    def __init__(self) -> None:
        self._free: dict[_StoragePoolKey, dict[int, _StorageLease]] = defaultdict(dict)

    def acquire(
        self,
        ftype: TensorFType,
        shape: tuple[int, ...],
        pin_fill: bool,
    ) -> _StorageLease:
        key = _StoragePoolKey(
            repr(ftype),
            tuple(int(dimension) for dimension in shape),
            pin_fill,
        )
        if self._free[key]:
            return self._free[key].popitem()[1]
        match ftype:
            case FiberTensorFType(fill_value=DynamicFill() as fill):
                # A dynamic-fill ftype can't construct storage without its fill.
                tensor = ftype.construct(shape, fill_value=fill)
            case _:
                tensor = ftype.construct(shape)
        return _StorageLease(
            tensor_to_jl(tensor, pin_fill=pin_fill),
            key,
        )

    def release_lease(self, lease: _StorageLease) -> None:
        free_leases = self._free[lease.key]
        free_leases.setdefault(id(lease), lease)


class JuliaOwnedTensor(OverrideTensor):
    """A Finch tensor handle for storage owned by the Julia runtime."""

    def __init__(
        self,
        ftype: TensorFType,
        shape: tuple[int, ...],
        release: Callable[[JuliaOwnedTensor], None],
        raw_julia_obj: Any,
        pin_fill: bool,
        lease: _StorageLease | None = None,
        translation_finalizer: weakref.finalize | None = None,
    ) -> None:
        self._ftype = ftype
        self._shape = shape
        self._release = release
        self._raw_julia_obj = raw_julia_obj
        self._pin_fill = pin_fill
        self._lease = lease
        # If this is a translation of a Python tensor, this finalizer triggers its
        # garbage collection when that tensor is killed.
        self._translation_finalizer = translation_finalizer

    def __del__(self) -> None:
        if self._translation_finalizer is not None:
            self._translation_finalizer.detach()
        self._release(self)

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

    def __getitem__(self, index):
        if not self._shape:
            return Scalar(self.item(), fill_value=self.fill_value, device=self.device)[
                index
            ]
        result = self._as_tensor().to_numpy()[index]
        if isinstance(result, np.ndarray):
            return BufferizedNDArray.from_numpy(
                result,
                fill_value=self.fill_value,
                device=self.device,
            )
        return Scalar(result, fill_value=self.fill_value, device=self.device)

    def to_numpy(self):
        return self._as_tensor().to_numpy()

    def __array__(self, dtype=None, copy=None):
        out = np.asarray(self.to_numpy())
        if dtype is not None and out.dtype != dtype:
            if copy is False:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            out = out.astype(dtype)
        if copy is True:
            return out.copy()
        return out

    def to_scipy(self):
        return self._as_tensor().to_scipy()


@dataclass(frozen=True)
class _TranslationCacheKey:
    pin_fill: bool
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
    """Julia runtime with cached conversions and reusable result storage."""

    def __init__(self) -> None:
        self._kernels: dict[Any, Any] = {}
        self._kernel_metadata: dict[int, _KernalMetadata] = {}
        self._translated_tensors: dict[_TranslationCacheKey, JuliaOwnedTensor] = {}
        self.free_pool = _StoragePool()

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

        # Lease Julia storage only for resettable compiler-created outputs.
        julia_buf_args: list[JuliaOwnedTensor] = []
        for position, tensor in enumerate(args):
            pin_fill = position in getattr(kernel, "dynamic_args", ())
            if position in metadata.reset_positions and not isinstance(
                tensor, JuliaOwnedTensor
            ):
                lease = self.free_pool.acquire(
                    tensor.ftype,
                    tensor.shape,
                    pin_fill,
                )
                julia_buf = JuliaOwnedTensor(
                    tensor.ftype,
                    lease.key.shape,
                    self.release,
                    lease.raw,
                    lease.key.pin_fill,
                    lease,
                )
            else:
                julia_buf = self._to_julia_owned_tensor(tensor, pin_fill)
            julia_buf_args.append(julia_buf)

        raw_args = [arg.raw_julia_obj for arg in julia_buf_args]
        raw_args += [
            int(args[pos].shape[axis]) for pos, axis in getattr(kernel, "extents", ())
        ]
        getattr(jl, func_name)(*raw_args)

        # Associate returned tensors with their Python ownership handles.
        return tuple(
            julia_buf_args[position] for position in metadata.returned_positions
        )

    def _to_julia_owned_tensor(
        self, tensor: Tensor, pin_fill: bool = False
    ) -> JuliaOwnedTensor:
        if isinstance(tensor, JuliaOwnedTensor):
            if tensor._pin_fill == pin_fill:
                return tensor
            return self._to_julia_owned_tensor(tensor._as_tensor(), pin_fill)
        key = self._translation_cache_key(tensor, pin_fill)
        julia_tensor = self._translated_tensors.get(key)
        if julia_tensor is not None:
            return julia_tensor

        raw = tensor_to_jl(tensor, pin_fill=pin_fill)
        translation_finalizer = weakref.finalize(
            tensor, self._translated_tensors.pop, key, None
        )
        julia_tensor = JuliaOwnedTensor(
            tensor.ftype,
            tuple(int(dimension) for dimension in tensor.shape),
            self.release,
            raw,
            pin_fill,
            translation_finalizer=translation_finalizer,
        )
        self._translated_tensors[key] = julia_tensor
        return julia_tensor

    @staticmethod
    def _translation_cache_key(
        tensor: Tensor, pin_fill: bool = False
    ) -> _TranslationCacheKey:
        if isinstance(tensor, BufferizedNDArray):
            array = tensor.to_numpy()
            return _TranslationCacheKey(
                pin_fill,
                "numpy",
                # The first data-interface entry is the array's memory address.
                array.__array_interface__["data"][0],
                tuple(int(dimension) for dimension in array.shape),
                tuple(int(stride) for stride in array.strides),
                array.dtype.str,
            )
        if isinstance(tensor, NumPyWrapper):
            array = tensor._data
            return _TranslationCacheKey(
                pin_fill,
                "numpy",
                # The first data-interface entry is the array's memory address.
                array.__array_interface__["data"][0],
                tuple(int(dimension) for dimension in array.shape),
                tuple(int(stride) for stride in array.strides),
                array.dtype.str,
            )
        return _TranslationCacheKey(pin_fill, "object", id(tensor), None, None, None)

    def release(self, tensor: JuliaOwnedTensor) -> None:
        if tensor._lease is not None:
            self.free_pool.release_lease(tensor._lease)

    def close(self) -> None:
        self._kernels.clear()
        self._kernel_metadata.clear()
        self._translated_tensors.clear()
