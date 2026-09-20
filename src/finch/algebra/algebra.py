"""Algebraic interfaces and helpers used by Finch operators."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    def ftype(self) -> "FinchOperatorFType":
        return SingletonOperatorFType(self)

    is_associative: bool = False
    is_commutative: bool = False
    is_idempotent: bool = False
    arity: int | float = 2

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def return_type(self, *args: FType) -> FType:
        pass

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return False

    def is_identity(self, val: Any) -> bool:
        return False

    def is_annihilator(self, val: Any) -> bool:
        return False

    def init_value(self, type_: FType) -> Any:
        raise AttributeError(f"{type(self)} has no init_value")

    def repeat_operator(self) -> Any:
        if self.is_idempotent:
            return None
        raise AttributeError(f"{type(self)} has no repeat_operator")

    def __qual_str__(self) -> str:
        """Return qualified string for printing/display purposes."""
        # Display as just the lowercase name
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


@dataclass(unsafe_hash=True)
class SingletonOperatorFType(ImmutableStructFType, FinchOperatorFType):
    """A callable type specializing on a particular operator value."""

    operator: FinchOperator

    @property
    def struct_name(self):
        return "SingletonOperator"

    @property
    def struct_fields(self):
        return []

    def from_fields(self):
        return self.operator

    def __call__(self, value):
        if value == self.operator:
            return value
        raise TypeError(f"Expected {self.operator}")

    @property
    def arity(self):
        return self.operator.arity

    @property
    def is_associative(self):
        return self.operator.is_associative

    @property
    def is_commutative(self):
        return self.operator.is_commutative

    @property
    def is_idempotent(self):
        return self.operator.is_idempotent

    def is_distributive(self, other):
        match other:
            case SingletonOperatorFType(operator):
                return self.operator.is_distributive(operator)
            case _:
                return False

    def is_identity(self, val):
        return self.operator.is_identity(val)

    def is_annihilator(self, val):
        return self.operator.is_annihilator(val)

    def init_value(self, type_):
        return self.operator.init_value(type_)

    def repeat_operator(self):
        return self.operator.repeat_operator()

    def return_type(self, *args):
        return self.operator.return_type(*args)


def _operator_type(op: FinchOperator | FinchOperatorFType) -> FinchOperatorFType:
    result = ftype(op)
    if not isinstance(result, FinchOperatorFType):
        raise TypeError(f"Expected a Finch operator type, got {result}")
    return result


def arity(op: FinchOperator | FinchOperatorFType) -> int | float:
    return _operator_type(op).arity


def is_associative(op: FinchOperator | FinchOperatorFType) -> bool:
    return _operator_type(op).is_associative


def is_commutative(op: FinchOperator | FinchOperatorFType) -> bool:
    return _operator_type(op).is_commutative


def is_idempotent(op: FinchOperator | FinchOperatorFType) -> bool:
    return _operator_type(op).is_idempotent


def _specializable(val: Any) -> tuple[bool, Any]:
    if isinstance(val, AbstractFill):
        return not is_dynamic(val), val.value
    return True, val


def is_identity(op: FinchOperator | FinchOperatorFType, val: Any) -> bool:
    ok, value = _specializable(val)
    return ok and _operator_type(op).is_identity(value)


def is_annihilator(op: FinchOperator | FinchOperatorFType, val: Any) -> bool:
    ok, value = _specializable(val)
    return ok and _operator_type(op).is_annihilator(value)


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


def is_distributive(
    op: FinchOperator | FinchOperatorFType, other_op: FinchOperator | FinchOperatorFType
) -> bool:
    return _operator_type(op).is_distributive(_operator_type(other_op))


def return_type(op: FinchOperator | FType, *args: FType) -> FType:
    arg_types = tuple(ftype(arg) for arg in args)
    op_type = ftype(op)
    if not isinstance(op_type, CallableFType):
        raise TypeError(f"Expected a callable type, got {op_type}")
    return op_type.return_type(*arg_types)


def init_value(op: FinchOperator | FinchOperatorFType, arg: Any) -> Any:
    return _operator_type(op).init_value(arg)


def fixpoint_type(op: FinchOperator | FinchOperatorFType, z: Any, t: FType) -> FType:
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


def repeat_operator(op: FinchOperator | FinchOperatorFType):
    """
    If there exists an operator g such that
    f(x, x, ..., x)  (n times)  is equal to g(x, n),
    then return g.
    """
    return _operator_type(op).repeat_operator()


def cansplitpush(
    x: FinchOperator | FinchOperatorFType, y: FinchOperator | FinchOperatorFType
):
    """
    Return True if a reduction with operator `x` can be 'split-pushed' through
    a pointwise operator `y`.

    We allow split-push when:
      - x has a known repeat operator (repeat_operator(x) is not None),
      - x and y are the same operator,
      - and x is both commutative and associative.
    """
    if not callable(x) or not callable(y):
        raise TypeError("Can't check splitpush of non-callable operators!")

    return (
        repeat_operator(x) is not None
        and ftype(x) == ftype(y)
        and is_commutative(x)
        and is_associative(x)
    )
