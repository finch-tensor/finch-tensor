"""
Static/Dynamic fills.

Every tensor has a fill value: the background value of a sparse tensor, and the
value a newly constructed tensor is filled with. The fill value belongs to the
tensor, and the tensor's format records whether a kernel may specialize on it:

* `StaticFillFType` -- compile against this value, which is part of the format.
  It may be folded into generated code. Its values are `StaticFill`s.
* `DynamicFillFType` -- do not compile against the value, so only its element
  type is part of the format. The value may change across invocations, so one
  compiled kernel serves every fill of the same element type. Its values are
  `DynamicFill`s.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, overload

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


class AbstractFillFType(FType, ABC):
    """
    The fill of a format: the element type of its fill, and whether kernels may
    specialize on the value of the fill.
    """

    @property
    @abstractmethod
    def element_type(self) -> FType:
        """The element type of the fill, always known."""
        ...


class StaticFillFType(AbstractFillFType):
    """
    The fill of a format which kernels may specialize on, so its value is part
    of the format. Equality and hashing are by value, so formats with different
    static fills are distinct and get distinct kernels.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    @property
    def element_type(self) -> FType:
        return ftype(self._value)

    def __call__(self, val: Any) -> StaticFill:
        fill = StaticFill(self.element_type(val))
        if fill.ftype != self:
            raise ValueError(f"{val!r} is not the static fill {self._value!r}")
        return fill

    def __eq__(self, other):
        # Values are compared with `same` rather than `==` because of NaN.
        # ftypes filled with NaN must compare equal or they get separate
        # kernels and fail each other's type checks. Imported lazily: `ffuncs`
        # depends on this module.
        from .ffuncs import same

        if not isinstance(other, StaticFillFType):
            return False
        return bool(np.all(same(self._value, other._value)))

    def __hash__(self):
        from .ffuncs import samehash

        return hash((StaticFillFType, samehash(self._value)))

    def __repr__(self):
        return f"StaticFillFType({self._value!r})"


class DynamicFillFType(AbstractFillFType):
    """
    The fill of a format which kernels must not specialize on, so only its
    element type is part of the format. Each tensor holds its own fill value,
    which is bound to the kernel at call time.
    """

    def __init__(self, element_type: Any) -> None:
        self._element_type = ftype(element_type)

    @property
    def element_type(self) -> FType:
        return self._element_type

    def __call__(self, val: Any) -> DynamicFill:
        return DynamicFill(self._element_type(val), self._element_type)

    def __eq__(self, other):
        return (
            isinstance(other, DynamicFillFType)
            and self._element_type == other._element_type
        )

    def __hash__(self):
        return hash((DynamicFillFType, self._element_type))

    def __repr__(self):
        return f"DynamicFillFType({self._element_type!r})"


class AbstractFill(FTyped[AbstractFillFType], ABC):
    """A tensor's fill value. Its ftype says whether kernels may specialize on it."""

    @property
    @abstractmethod
    def ftype(self) -> AbstractFillFType:
        """The fill of the format which this fill belongs to."""
        ...

    @property
    def element_type(self) -> FType:
        """The element type of the fill value, always known."""
        return self.ftype.element_type

    @property
    @abstractmethod
    def value(self) -> Any:
        """The fill value itself, always known."""
        ...

    def as_dynamic(self) -> DynamicFill:
        """This fill, marked so that kernels will not specialize on its value."""
        return DynamicFill(self.value, self.element_type)


class StaticFill(AbstractFill):
    """
    A fill value which kernels may specialize on. Equality and hashing are by
    value, matching `StaticFillFType`.
    """

    def __init__(self, value: Any) -> None:
        if isinstance(value, AbstractFill):
            self._value = value.value
        else:
            self._value = value

    @property
    def ftype(self) -> StaticFillFType:
        return StaticFillFType(self._value)

    @property
    def value(self) -> Any:
        return self._value

    def __eq__(self, other):
        return isinstance(other, StaticFill) and self.ftype == other.ftype

    def __hash__(self):
        return hash((StaticFill, self.ftype))

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

    Equality and hashing are by element type *only*, i.e. two equal
    `DynamicFill`s may have different `.value`'s.
    """

    def __init__(self, value: Any, element_type: Any | None = None) -> None:
        if isinstance(value, AbstractFill):
            self._value = value.value
        else:
            self._value = value
        self._element_type = (
            ftype(self._value) if element_type is None else ftype(element_type)
        )

    @property
    def ftype(self) -> DynamicFillFType:
        return DynamicFillFType(self._element_type)

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def value(self) -> Any:
        return self._value

    def as_dynamic(self) -> DynamicFill:
        return self

    def __eq__(self, other):
        return isinstance(other, DynamicFill) and self.ftype == other.ftype

    def __hash__(self):
        return hash((DynamicFill, self.ftype))

    def __same__(self, other):
        return self == other

    def __rsame__(self, other):
        return self == other

    def __samehash__(self):
        return self

    def __repr__(self):
        return f"DynamicFill({self._value!r}, {self._element_type!r})"


AF = TypeVar("AF", bound=AbstractFill)


@overload
def as_fill(fill: AF) -> AF: ...


@overload
def as_fill(fill: Any) -> StaticFill: ...


def as_fill(fill: Any) -> AbstractFill:
    """Normalize a raw value to a `StaticFill`, passing an `AbstractFill` through."""
    if isinstance(fill, AbstractFill):
        return fill
    return StaticFill(fill)


def is_dynamic(fill: Any) -> TypeGuard[DynamicFill]:
    return isinstance(fill, DynamicFill)


def apply_fill(op: FinchOperator, *fills: Any) -> AbstractFill:
    """
    Compute the fill value of mapping `op` over tensors with fills `fills`.

    A fill stays static only while its value is one the algebra can act on.
    Combining static fills computes an arbitrary value -- adding 1 repeatedly
    walks a fill through 1, 2, 3, ... -- and a static fill is compared by
    value, so keeping those static would key a kernel per distinct value and
    make a loop cost a compilation per iteration.
    """
    # Imported lazily: `algebra` depends on this module.
    from .algebra import is_specializable_value

    fills = tuple(as_fill(f) for f in fills)
    values = [f.value for f in fills]
    if not any(is_dynamic(f) for f in fills):
        result = op(*values)
        if is_specializable_value(result):
            return StaticFill(result)
        return DynamicFill(result)
    for f in fills:
        if not is_dynamic(f) and op.ftype.is_annihilator(f.value):
            result_type = op.ftype.return_type(*(g.element_type for g in fills))
            return StaticFill(result_type(f.value))
    element_type = op.ftype.return_type(*(f.element_type for f in fills))
    return DynamicFill(op(*values), element_type)
