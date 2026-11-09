from __future__ import annotations

import math
import numbers
import weakref
from collections.abc import Hashable
from typing import Any, Self, TypeVar

"""
The table holds weak references, so an object is dropped from it as soon as
nothing else refers to it.

To intern a class, inherit from `HashCons`, which
returns the fields that identify an instance. Dataclasses must use `eq=False`
so that the generated structural `__eq__`/`__hash__` do not replace the
pointer-based ones defined here.

We recommend not defining `__init__` at all on classes inheriting from `HashCons`,
as it will run on every invocation rather than just the first one.

Please override `__new__` and override your logic there instead if needed.
"""

T = TypeVar("T")

_table: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _nan_to_none(x: float) -> float | None:
    return x if not math.isnan(x) else None


class HashCons:
    def __new__(cls, *args: Any) -> Self:
        hash_key = cls.hash_key(*args)
        if (cached := _table.get(hash_key)) is not None:
            return cached
        ret = super().__new__(cls)
        _table[hash_key] = ret
        return ret

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        return self

    @staticmethod
    def canonicalize(a: T, /) -> tuple[type[T], Hashable]:
        """
        Canonicalize an instance of this class into a (type, hashable) pair.

        Generic version provided, but one can override for speed.
        """
        if isinstance(a, numbers.Real):
            return type(a), _nan_to_none(float(a))

        if isinstance(a, numbers.Complex):
            return type(a), (_nan_to_none(float(a.real)), _nan_to_none(float(a.imag)))

        if isinstance(a, Hashable):
            return type(a), a

        return type(a), id(a)

    @classmethod
    def hash_key(cls, *args: Any) -> tuple[Hashable, ...]:
        """
        The hash key for this class, when it is constructed with `*args`.

        Generic version provided, but one can specialize for speed.
        """
        return (cls, *(cls.canonicalize(a) for a in args))
