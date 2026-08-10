from __future__ import annotations

from typing import Any

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    DynamicFill,
    FType,
    ImmutableStructFType,
    TensorFType,
    ffuncs,
    ftype,
    is_dynamic,
    normalize_device,
)

from .override_tensor import OverrideTensor


class ScalarFType(TensorFType, ImmutableStructFType):
    def __init__(self, _element_type: FType, _fill_value: Any, _device=None):
        self._element_type = _element_type
        self._fill_value = _fill_value
        self._device = normalize_device(_device)

    def __eq__(self, other):
        if isinstance(other, ScalarFType):
            return (
                self._element_type == other._element_type
                and bool(np.all(ffuncs.same(self._fill_value, other._fill_value)))
                and self.device == other.device
            )
        return False

    def __hash__(self):
        return hash(
            (self._element_type, ffuncs.samehash(self._fill_value), self.device)
        )

    def construct(self, shape: tuple) -> Scalar:
        if shape != ():
            raise ValueError("ScalarFType can only be called with empty shape ()")
        return self._element_type(self._fill_value)

    def __call__(self, val: Any) -> Scalar:
        """
        Convert a tensor to this scalar tensor type.

        Args:
            val: A value to convert to this type.
        Returns:
            A Scalar instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, arr):
        return self(arr)

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def device(self):
        return self._device

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return ()

    @property
    def struct_name(self) -> str:
        return "Scalar"

    @property
    def struct_fields(self) -> list[tuple[str, FType]]:
        return [("val", self._element_type)]

    def from_fields(self, val):
        return Scalar(val, fill_value=self._fill_value, device=self._device)

    def fisinstance(self, other):
        other_t = ftype(other)
        if is_dynamic(self._fill_value) and isinstance(other_t, ScalarFType):
            # A dynamic-fill ftype accepts any fill value of matching dtype.
            other_t = ScalarFType(
                other_t._element_type, self._fill_value, other_t._device
            )
        return other_t == self

    def lower_unwrap(self, ctx, obj):
        match obj:
            case ntn.Fiber():
                # A slot-bound scalar argument: read the value struct field.
                return asm.GetAttr(obj.root, asm.Literal("val"))
            case _:
                # An inline scalar value (e.g. a sparse gap read).
                return ctx(obj)


class Scalar(OverrideTensor):
    def __init__(self, val: Any, fill_value: Any = None, device=None):
        if fill_value is None:
            fill_value = val
        self.val = val
        self._fill_value = fill_value
        self._device = normalize_device(device)

    @property
    def ftype(self):
        return ScalarFType(ftype(self.val), self._fill_value, self._device)

    @property
    def argument_ftype(self):
        # A scalar's fill is not part of its kernel identity, so it never enters
        # the cache key: `struct_fields` carries only `val`, and `lower_unwrap`
        # reads only `val`, so no kernel body can observe the fill. It is used
        # solely at bind time -- `infer_fill_value` reads it off the actual
        # instance -- which is why one kernel serves every fill of a dtype.
        # `ConstantScalar` overrides this to opt into value specialization.
        elem_t = ftype(self.val)
        return ScalarFType(elem_t, DynamicFill(elem_t), self._device)

    @property
    def shape(self):
        return ()

    @property
    def fill_value(self) -> Any:
        """Default value to fill the scalar."""
        return self.ftype.fill_value

    @property
    def device(self):
        return self._device

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError(f"stream argument is not supported; got {stream!r}")
        device = normalize_device(device)
        if device == self.device:
            return self
        return Scalar(self.val, fill_value=self._fill_value, device=device)

    @property
    def element_type(self) -> FType:
        """Data type of the scalar."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the scalar."""
        return self.ftype.shape_type

    def item(self):
        return self.val.item() if hasattr(self.val, "item") else self.val

    def __array__(self, dtype=None, copy=None):
        if copy is None:
            return np.asarray(self.val, dtype=dtype)
        return np.array(self.val, dtype=dtype, copy=copy)

    def __getitem__(self, idx):
        if idx == () or idx is Ellipsis or idx == (...,):
            return self
        raise IndexError("Too many indices for scalar tensor.")

    def __str__(self):
        return str(self.val)

    def to_numpy(self):
        return self.val

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")


class ConstantScalar(Scalar):
    """
    A scalar whose value is treated as a compile-time constant. Kernels may
    specialize on the value (and recompile per distinct value), unlike plain
    `Scalar`s, whose values are bound at call time.
    """

    def __init__(self, val: Any):
        super().__init__(val)

    @property
    def argument_ftype(self):
        # Constants opt into value specialization; normally they are inlined
        # before ever becoming a binding, so this is defensive.
        return self.ftype
