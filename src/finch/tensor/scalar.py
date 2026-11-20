from __future__ import annotations

from typing import Any

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    AbstractFill,
    AbstractFillFType,
    DynamicFillFType,
    FType,
    MutableStructFType,
    StaticFill,
    StaticFillFType,
    as_fill,
    ffuncs,
    ftype,
    is_dynamic,
    is_static,
    normalize_device,
    resolve_fill,
)
from finch.compile.lower import FinchTensorFType

from .override_tensor import OverrideTensor


class ScalarFType(FinchTensorFType, MutableStructFType):
    """
    A 0-dimensional tensor which stores its value in a field, apart from its
    fill. A static fill is part of the type, and a dynamic fill is a second
    field, which declaring the scalar sets. Kernels may update the value, which
    is written back to the scalar, unless the scalar is a compile-time constant.
    """

    def __init__(
        self,
        _element_type: FType,
        _fill_type: AbstractFillFType,
        _device=None,
        constant: bool = False,
    ):
        self._element_type = _element_type
        self._fill_type = _fill_type
        self._device = normalize_device(_device)
        self.constant = constant

    def __eq__(self, other):
        if isinstance(other, ScalarFType):
            return (
                self._element_type == other._element_type
                and self._fill_type == other._fill_type
                and self.device == other.device
                and self.constant == other.constant
            )
        return False

    def __hash__(self):
        return hash((self._element_type, self._fill_type, self.device, self.constant))

    def construct(self, shape: tuple, fill_value: Any = None) -> Scalar:
        if shape != ():
            raise ValueError("ScalarFType can only be called with empty shape ()")
        if fill_value is None and is_dynamic(self._fill_type):
            fill_value = self._fill_type(self._element_type(0))
        fill = resolve_fill(
            self._fill_type, None if fill_value is None else as_fill(fill_value)
        )
        val = self._element_type(fill.value)
        return self.from_fields(val, val) if is_dynamic(fill) else self.from_fields(val)

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
    def fill_type(self) -> AbstractFillFType:
        return self._fill_type

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
        if is_dynamic(self._fill_type):
            return [("val", self._element_type), ("fill", self._element_type)]
        return [("val", self._element_type)]

    def struct_getattr(self, obj, attr):
        # A dynamic fill is bound at call time through the "fill" field.
        if attr == "fill":
            return obj.fill_value
        return getattr(obj, attr)

    def struct_setattr(self, obj, attr, value):
        if attr == "fill":
            obj._fill_value = self._fill_type(value)
        else:
            setattr(obj, attr, value)

    def from_fields(self, val, fill=None):
        if self.constant:
            return ConstantScalar(val, device=self._device)
        if is_dynamic(self._fill_type):
            return Scalar(val, fill_value=self._fill_type(fill), device=self._device)
        return Scalar(
            val, fill_value=resolve_fill(self._fill_type), device=self._device
        )

    def with_fill(self, fill_type: AbstractFillFType) -> ScalarFType:
        return ScalarFType(self._element_type, fill_type, self._device)

    def fisinstance(self, other):
        other_t = ftype(other)
        if is_dynamic(self._fill_type) and isinstance(other_t, ScalarFType):
            # A dynamic-fill ftype accepts any fill value of matching dtype.
            other_t = other_t.with_fill(self._fill_type)
        return other_t == self

    def lower_fill(self, ctx, obj):
        if is_static(self._fill_type):
            return asm.Literal(self._fill_type.value)
        match obj:
            case ntn.Fiber():
                return asm.GetAttr(ctx(obj.lvl.root), asm.Literal("fill"))
            case _:
                # An inline scalar value (e.g. a sparse gap read) is its fill.
                return ctx(obj)

    def lower_unwrap(self, ctx, obj):
        match obj:
            case ntn.Fiber():
                # A slot-bound scalar argument: read the value struct field.
                return asm.GetAttr(ctx(obj.lvl.root), asm.Literal("val"))
            case _:
                # An inline scalar value (e.g. a sparse gap read).
                return ctx(obj)

    def get_child_type(self, attr: str) -> FType:
        return self

    def get_child(self, obj, attr: str):
        return obj

    def lower_dim(self, ctx, obj, r):
        raise IndexError(f"A scalar has no dimension {r}")

    def _set(self, ctx, obj, attr: str, val: asm.AssemblyExpression):
        if self.constant:
            raise ValueError(f"Cannot write to the constant scalar {obj}")
        if val.result_type != self._element_type:
            val = asm.Call(asm.Literal(ffuncs.astype(self._element_type)), (val,))
        ctx.exec(asm.SetAttr(ctx(obj.lvl.root), asm.Literal(attr), val))

    def lower_declare(self, ctx, tns, init, op, shape):
        # As for other tensors, declaring sets a dynamic fill to the init.
        if is_dynamic(self._fill_type):
            self._set(ctx, tns, "fill", init)
        self._set(ctx, tns, "val", init)

    def lower_increment(self, ctx, obj, op, val):
        cur = asm.GetAttr(ctx(obj.lvl.root), asm.Literal("val"))
        self._set(ctx, obj, "val", asm.Call(ctx(op), (cur, ctx(val))))

    def lower_freeze(self, ctx, tns, op):
        return tns

    def lower_thaw(self, ctx, tns, op):
        return tns

    def unfurl(self, ctx, tns, ext, mode, proto):
        raise NotImplementedError("A scalar has no index to unfurl")


class Scalar(OverrideTensor):
    """
    A 0-dimensional tensor. Its fill is apart from its value: a static fill
    when given as a `StaticFill`, and otherwise a dynamic fill, which defaults
    to the value.
    """

    def __init__(self, val: Any, fill_value: Any = None, device=None):
        elem_t = ftype(val)
        if fill_value is None:
            fill_value = val
        if not isinstance(fill_value, AbstractFill):
            fill_value = DynamicFillFType(elem_t)(elem_t(fill_value))
        self.val = val
        self._fill_value = fill_value
        self._device = normalize_device(device)

    @property
    def ftype(self):
        elem_t = ftype(self.val)
        if is_static(self._fill_value):
            return ScalarFType(elem_t, StaticFillFType(self.fill_value), self._device)
        return ScalarFType(elem_t, DynamicFillFType(elem_t), self._device)

    @property
    def shape(self):
        return ()

    @property
    def fill_value(self) -> Any:
        """Default value to fill the scalar."""
        return self._fill_value.value

    @property
    def fill(self) -> AbstractFill:
        return self._fill_value

    @property
    def device(self):
        return self._device

    def with_fill(self, fill_value: AbstractFill) -> Scalar:
        return Scalar(self.val, fill_value=fill_value, device=self._device)

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

    def declare(self, init, op, shape):
        init = self.element_type(init)
        if is_dynamic(self._fill_value):
            self._fill_value = DynamicFillFType(self.element_type)(init)
        elif not bool(ffuncs.same(init, self.fill_value)):
            raise ValueError(
                f"Declaring with {init!r} would change the static fill "
                f"{self.fill_value!r}"
            )
        self.val = init
        return self

    def freeze(self, op):
        return self

    def thaw(self, op):
        return self

    def access(self, indices, op):
        if indices:
            raise IndexError("Too many indices for scalar tensor.")
        return ScalarAccessor(self, op)

    def __str__(self):
        return str(self.val)

    def to_numpy(self):
        return self.val

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")


class ScalarAccessor:
    """The element of a scalar, which kernels read or update with op."""

    def __init__(self, tns: Scalar, op=None):
        self.tns = tns
        self.op = op

    def unwrap(self):
        return self.tns.val

    def increment(self, val):
        if self.op is None:
            raise ValueError("No operation defined for increment.")
        self.tns.val = self.tns.element_type(self.op(self.tns.val, val))
        return self


class ConstantScalar(Scalar):
    """
    A scalar whose value is treated as a compile-time constant. Kernels may
    specialize on the value (and recompile per distinct value), unlike plain
    `Scalar`s, whose values are bound at call time.
    """

    def __init__(self, val: Any, device=None):
        super().__init__(val, fill_value=StaticFill(val), device=device)

    @property
    def ftype(self):
        return ScalarFType(
            ftype(self.val), StaticFillFType(self.val), self._device, constant=True
        )

    def with_fill(self, fill_value: AbstractFill) -> Scalar:
        if is_static(fill_value) and bool(ffuncs.same(fill_value.value, self.val)):
            return self
        return Scalar(self.val, fill_value=fill_value, device=self._device)

    def declare(self, init, op, shape):
        raise ValueError(f"Cannot declare the constant scalar {self}")
