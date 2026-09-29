from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from finch.algebra import Tensor, TensorFType


@dataclass
class _BufferLease:
    raw: Any
    key: _BufferPoolKey


@dataclass(frozen=True)
class _BufferPoolKey:
    ftype: str
    shape: tuple[int, ...]
    pin_fill: bool


class _BufferPool:
    def __init__(self) -> None:
        self._free: dict[_BufferPoolKey, dict[int, _BufferLease]] = defaultdict(dict)

    def acquire(
        self,
        ftype: TensorFType,
        shape: tuple[int, ...],
        pin_fill: bool,
    ) -> _BufferLease:
        from .interop import tensor_to_jl

        key = _BufferPoolKey(
            repr(ftype),
            tuple(int(dimension) for dimension in shape),
            pin_fill,
        )
        if self._free[key]:
            return self._free[key].popitem()[1]
        tensor = ftype.construct(shape)
        return _BufferLease(
            tensor_to_jl(tensor, pin_fill=pin_fill),
            key,
        )

    def release_lease(self, lease: _BufferLease) -> None:
        free_leases = self._free[lease.key]
        free_leases.setdefault(id(lease), lease)


class JuliaOwnedTensor(Tensor):
    """A Finch tensor handle for storage owned by the Julia runtime."""

    def __init__(
        self,
        ftype: TensorFType,
        shape: tuple[int, ...],
        release: Callable[[JuliaOwnedTensor], None],
        raw_julia_obj: Any,
        pin_fill: bool,
        lease: _BufferLease | None = None,
    ) -> None:
        self._ftype = ftype
        self._shape = shape
        self._release = release
        self._raw_julia_obj = raw_julia_obj
        self._pin_fill = pin_fill
        self._lease = lease

    def __del__(self) -> None:
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
        from .interop import jl_tensor_to_python

        return jl_tensor_to_python(self.raw_julia_obj)

    def item(self):
        if not self._shape:
            values = self.raw_julia_obj.lvl.val.to_numpy(copy=False)
            return values[0].item()
        return self._as_tensor().item()

    def __getitem__(self, index):
        if not self._shape:
            return self.item()
        return self._as_tensor().to_numpy()[index]

    def to_numpy(self):
        return self._as_tensor().to_numpy()

    def __array__(self, dtype=None, copy=None):
        out = np.asarray(self.to_numpy())
        if dtype is not None and out.dtype != dtype:
            if copy is not None and not copy:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            out = out.astype(dtype)
        return out

    def to_scipy(self):
        return self._as_tensor().to_scipy()
