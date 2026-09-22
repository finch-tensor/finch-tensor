from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import Any, Self

"""
Hash consing: constructing an object returns the canonical instance for its
`__hash_keys__`, so structurally equal objects are the same object and equality
is pointer equality.

The table holds weak references, so an object is dropped from it as soon as
nothing else refers to it.
"""

_table: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class HashCons(ABC):
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        # Fill the fields so that __hash_keys__ can read them. Python calls
        # __init__ once more on the object we return; for a frozen dataclass
        # that only re-assigns equal field values.
        obj.__init__(*args, **kwargs)
        return _table.setdefault((cls, *obj.__hash_keys__()), obj)

    @abstractmethod
    def __hash_keys__(self) -> tuple:
        """Return the tuple of values which identify this object."""
        ...

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        return self
