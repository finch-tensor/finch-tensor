"""Algebraic interfaces and helpers used by Finch operators."""

import math
from abc import ABC, abstractmethod
from typing import Any

from .fill import AbstractFill, is_dynamic
from .ftypes import (
    CallableFType,
    FDTypeOrdered,
    FType,
    FTyped,
    ImmutableStructFType,
    ftype,
)


class FinchOperator(FTyped, ABC):
    @property
    @abstractmethod
    def ftype(self) -> "FinchOperatorFType": ...

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    def __qual_str__(self) -> str:
        return repr(self)


class FinchOperatorFType(CallableFType, ABC):
    is_associative: bool = False
    is_commutative: bool = False
    is_idempotent: bool = False
    arity: int | float = 2

    def is_distributive(self, other: "FinchOperatorFType") -> bool:
        return False

    def is_identity(self, val: Any) -> bool:
        return False

    def is_annihilator(self, val: Any) -> bool:
        return False

    def init_value(self, type_: FType) -> Any:
        raise AttributeError(f"{self} has no init_value")

    def repeat_operator(self) -> Any:
        if self.is_idempotent:
            return None
        raise AttributeError(f"{self} has no repeat_operator")


class SingletonOperatorFType(ImmutableStructFType, FinchOperatorFType):
    """A stateless callable type with one canonical operator value."""

    @property
    @abstractmethod
    def operator(self) -> FinchOperator: ...

    @property
    def struct_name(self):
        return type(self).__name__.removeprefix("_").removesuffix("FType")

    @property
    def struct_fields(self):
        return []

    def from_fields(self):
        return self.operator

    def __call__(self, value):
        if ftype(value) == self:
            return value
        raise TypeError(f"Expected an operator of type {self}")


def arity(op: FinchOperatorFType) -> int | float:
    return op.arity


def is_associative(op: FinchOperatorFType) -> bool:
    return op.is_associative


def is_commutative(op: FinchOperatorFType) -> bool:
    return op.is_commutative


def is_idempotent(op: FinchOperatorFType) -> bool:
    return op.is_idempotent


def _specializable(val: Any) -> tuple[bool, Any]:
    if isinstance(val, AbstractFill):
        return not is_dynamic(val), val.value
    return True, val


def is_identity(op: FinchOperatorFType, val: Any) -> bool:
    ok, value = _specializable(val)
    return ok and op.is_identity(value)


def is_annihilator(op: FinchOperatorFType, val: Any) -> bool:
    ok, value = _specializable(val)
    return ok and op.is_annihilator(value)


SPECIALIZABLE_VALUES = (0, 1, -1, math.inf, -math.inf)


def is_specializable_value(val: Any) -> bool:
    """
    A constant is only worth specializing on if some operator's identity or
    annihilator rule can fire against it; see `SPECIALIZABLE_VALUES`.
    """
    try:
        return any(bool(val == candidate) for candidate in SPECIALIZABLE_VALUES)
    except (TypeError, ValueError):
        return False


def is_distributive(op: FinchOperatorFType, other_op: FinchOperatorFType) -> bool:
    return op.is_distributive(other_op)


def return_type(op: CallableFType, *args: FType) -> FType:
    return op.return_type(*args)


def init_value(op: FinchOperatorFType, arg: FType) -> Any:
    return op.init_value(arg)


def fixpoint_type(op: CallableFType, z: Any, t: FType) -> FType:
    """
    Determines the fixpoint type after repeated calling the given operation.

    Args:
        op: The operation to evaluate.
        z: The initial value.
        t: The type to evaluate against.

    Returns:
        The fixpoint type.
    """
    s = set()
    z_type = ftype(z)
    r = z_type
    while r not in s:
        s.add(r)
        r = return_type(op, z_type, t)
    return r


def type_min(type_: FDTypeOrdered) -> Any:
    """
    Returns the minimum value of the given type.

    Args:
        type_: The type to determine the minimum value for.

    Returns:
        The minimum value of the given type.

    Raises:
        AttributeError: If the minimum value is not implemented for the given type.
    """
    return type_.type_min


def type_max(type_: FDTypeOrdered) -> Any:
    """
    Returns the maximum value of the given type.

    Args:
        type_: The type to determine the maximum value for.

    Returns:
        The maximum value of the given type.

    Raises:
        AttributeError: If the maximum value is not implemented for the given type.
    """
    return type_.type_max


def repeat_operator(op: FinchOperatorFType):
    """
    If there exists an operator g such that
    f(x, x, ..., x)  (n times)  is equal to g(x, n),
    then return g.
    """
    return op.repeat_operator()


def cansplitpush(op: FinchOperatorFType) -> bool:
    """Whether a reduction can split-push through the same operator expression."""
    return repeat_operator(op) is not None and is_commutative(op) and is_associative(op)
