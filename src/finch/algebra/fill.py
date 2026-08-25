"""
Static/Dynamic fill values.

Every tensor has a fill value: the background value of a sparse tensor, and the
value a newly constructed tensor is filled with. A fill is always a real value,
but a tensor's *ftype* additionally records whether a kernel may specialize on
that value:

* `StaticFill` -- compile against this value. It may be folded into generated
  code.
* `DynamicFill` -- do not compile against this value. The value is may change across
  invocations, so one compiled kernel serves every fill of the same dtype.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .ftypes import FType, FTyped, ftype

if TYPE_CHECKING:
    from .algebra import FinchOperator


class DynamicFillError(Exception):
    """
    Raised when a fill value must be specialized on at compile time but the fill
    is Dynamic. Callers may catch this to fall back to value-specialized
    compilation.
    """


class AbstractFill(FTyped, ABC):
    """A tensor's fill value, and whether kernels may specialize on it."""

    @property
    @abstractmethod
    def ftype(self) -> FType:
        """The dtype of the fill value, always known."""
        ...

    @property
    @abstractmethod
    def value(self) -> Any:
        """The fill value itself, always known."""
        ...

    def as_dynamic(self) -> DynamicFill:
        """This fill, marked so that kernels will not specialize on its value."""
        return DynamicFill(self.value, self.ftype)


class StaticFill(AbstractFill):
    """
    A fill value which kernels may specialize on. Equality and hashing are by
    value, so ftypes carrying different static fills are distinct and get
    distinct kernels.
    """

    def __init__(self, value: Any):
        if isinstance(value, AbstractFill):
            self._value = value.value
        else:
            self._value = value

    @property
    def ftype(self) -> FType:
        return ftype(self._value)

    @property
    def value(self) -> Any:
        return self._value

    def __eq__(self, other):
        # Values are compared with `same` rather than `==` because of NaN.
        # ftypes filled with NaN must compare equal or they get separate
        # kernels and fail each other's type checks. Imported lazily: `ffuncs`
        # depends on this module.
        from .ffuncs import same

        if not isinstance(other, StaticFill):
            return False
        return bool(np.all(same(self._value, other._value)))

    def __hash__(self):
        from .ffuncs import samehash

        return hash((StaticFill, samehash(self._value)))

    def __same__(self, other):
        return self == other

    def __rsame__(self, other):
        return self == other

    def __samehash__(self):
        return self

    def __repr__(self):
        return f"StaticFill({self._value!r})"


class DynamicFill(AbstractFill):
    """
    A fill value which kernels must not specialize on. The value is known and is
    bound to the kernel at call time.

    Equality and hashing are by dtype *only*, i.e. two equal `DynamicFill`s may have
    different `.value`'s.
    """

    def __init__(self, value: Any, dtype: Any = None):
        if isinstance(value, AbstractFill):
            self._value = value.value
        else:
            self._value = value
        self._dtype = ftype(self._value) if dtype is None else ftype(dtype)

    @property
    def ftype(self) -> FType:
        return self._dtype

    @property
    def value(self) -> Any:
        return self._value

    def as_dynamic(self) -> DynamicFill:
        return self

    def __eq__(self, other):
        return isinstance(other, DynamicFill) and self._dtype == other._dtype

    def __hash__(self):
        return hash((DynamicFill, self._dtype))

    def __same__(self, other):
        return self == other

    def __rsame__(self, other):
        return self == other

    def __samehash__(self):
        return self

    def __repr__(self):
        return f"DynamicFill({self._value!r}, {self._dtype!r})"


def as_fill(fill: Any) -> AbstractFill:
    """Normalize a raw value to a `StaticFill`, passing an `AbstractFill` through."""
    if isinstance(fill, AbstractFill):
        return fill
    return StaticFill(fill)


def is_dynamic(fill: Any) -> bool:
    return isinstance(fill, DynamicFill)


def apply_fill(op: FinchOperator, *fills: Any) -> AbstractFill:
    """
    Compute the fill value of mapping `op` over tensors with fills `fills`.
    """
    fills = tuple(as_fill(f) for f in fills)
    values = [f.value for f in fills]
    if not any(is_dynamic(f) for f in fills):
        return StaticFill(op(*values))
    for f in fills:
        if not is_dynamic(f) and op.is_annihilator(f.value):
            return f
    return DynamicFill(op(*values), op.return_type(*(f.ftype for f in fills)))
