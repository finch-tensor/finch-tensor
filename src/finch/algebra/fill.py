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
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, overload

import numpy as np

from .ftypes import FType, FTyped, ftype

if TYPE_CHECKING:
    from .algebra import FinchOperator

FT = TypeVar("FT", bound=FType)


class DynamicFillError(Exception):
    """
    Raised when a fill value must be specialized on at compile time but the fill
    is Dynamic. Callers may catch this to fall back to value-specialized
    compilation.
    """


class AbstractFill(FTyped[FT], ABC):
    """A tensor's fill value, and whether kernels may specialize on it."""

    @property
    @abstractmethod
    def ftype(self) -> FT:
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


class StaticFill(AbstractFill[FT]):
    """
    A fill value which kernels may specialize on. Equality and hashing are by
    value, so ftypes carrying different static fills are distinct and get
    distinct kernels.
    """

    @overload
    def __init__(self, value: FTyped[FT]) -> None: ...

    @overload
    def __init__(self, value: Any) -> None: ...

    def __init__(self, value) -> None:
        if isinstance(value, AbstractFill):
            self._value = value.value
        else:
            self._value = value

    @property
    def ftype(self) -> FT:
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


class DynamicFill(AbstractFill[FT]):
    """
    A fill value which kernels must not specialize on. The value is known and is
    bound to the kernel at call time.

    Equality and hashing are by dtype *only*, i.e. two equal `DynamicFill`s may have
    different `.value`'s.
    """

    @overload
    def __init__(self, value: AbstractFill[FT], dtype: None = None) -> None: ...

    @overload
    def __init__(self, value: Any, dtype: FT) -> None: ...

    @overload
    def __init__(self, value: Any, dtype: Any | None = None) -> None: ...

    def __init__(self, value: Any, dtype=None) -> None:
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


AF = TypeVar("AF", bound=AbstractFill)


@overload
def as_fill(fill: AF) -> AF: ...


@overload
def as_fill(fill: FTyped[FT]) -> StaticFill[FT]: ...


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
            result_type = op.ftype.return_type(*(g.ftype for g in fills))
            return StaticFill(result_type(f.value))
    return DynamicFill(op(*values), op.ftype.return_type(*(f.ftype for f in fills)))
