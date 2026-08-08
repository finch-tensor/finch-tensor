"""
Known/Dynamic fill values.

A fill value is "Known" when the value itself is compile-time data (any raw
value is implicitly Known). A fill is "Dynamic" when only its dtype is known
at compile time and the value arrives when the kernel is bound to arguments.
`DynamicFill` is the sentinel for the latter; kernels compiled against a
Dynamic fill are reusable across fill values of the same dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ftypes import FType, FTyped, ftype

if TYPE_CHECKING:
    from .algebra import FinchOperator


class DynamicFillError(Exception):
    """
    Raised when a concrete fill value is required at compile time but only a
    `DynamicFill` is available. Callers may catch this to fall back to
    value-specialized compilation.
    """


class DynamicFill(FTyped):
    """
    A fill value whose dtype is compile-time data but whose value is bound at
    kernel call time. Equality and hashing are dtype-based, so ftypes carrying
    a `DynamicFill` compare equal across distinct runtime fill values.
    """

    def __init__(self, dtype: Any):
        self._dtype = ftype(dtype)

    @property
    def ftype(self) -> FType:
        return self._dtype

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
        return f"DynamicFill({self._dtype!r})"


def is_dynamic(fill: Any) -> bool:
    return isinstance(fill, DynamicFill)


def apply_fill(op: FinchOperator, *fills: Any) -> Any:
    """
    Compute the fill value of mapping `op` over tensors with fills `fills`.

    All-Known fills fold eagerly to `op(*fills)`. A Known fill that
    annihilates `op` determines the result regardless of the Dynamic args.
    Otherwise the result is Dynamic with `op`'s return dtype.
    """
    if not any(is_dynamic(f) for f in fills):
        return op(*fills)
    for f in fills:
        if not is_dynamic(f) and op.is_annihilator(f):
            return f
    return DynamicFill(op.return_type(*(ftype(f) for f in fills)))
