from __future__ import annotations

import weakref
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, fields
from inspect import isbuiltin, isclass, isfunction
from typing import Any, Self

"""
This module contains definitions for common functions that are useful for symbolic
expression manipulation. Its purpose is to provide a shared interface between various
symbolic programming in finch.

Classes:
    Term (ABC): An abstract base class representing a symbolic term. It provides methods
    to access the head of the term, its children, and to construct a new term with a
    similar structure.

Notes:
    Although TermTree implements `children`, `make_term` is defined under Term.
    The reason for this is to enable writing IR-agnostic passes that operate on any kind
    of term, whether it is a leaf or an internal node. For example, a function that
    constructs or transforms a tree term may only have access to a leaf node, but still
    needs to call `make_term` on it.

    For example:

        def insert_wrapper(node: Term, pattern, wrap_head):
            if matches(node, pattern):
                return node.make_term(wrap_head, node)
            elseif isinstance(node, TermTree):
                def recurse(node: Term) -> Term:
                    return insert_wrapper(node, pattern, wrap_head)
                return node.make_term(node.head(), *(
                    recurse(child) for child in node.children
                ))
            else:
                return node

    This function would not be able to wrap leaf nodes if Term didn't define
    `make_term`.

    Also, `make_term` is not meant to be written differently for different
    members of Term.  Instead of overriding `make_term` in subclasses, introduce
    your own method to override, and call that from make_term.
"""


_ATOMIC_KEY_TYPES = frozenset({int, float, complex, bool, str, bytes, type(None)})

_hash_cons_types: set[type] = set()


def hash_key_value(val: Any) -> Any:
    t = type(val)
    if t in _hash_cons_types:
        return id(val)
    if t in _ATOMIC_KEY_TYPES:
        return (t, val)
    if t is tuple or t is list:
        return tuple([hash_key_value(v) for v in val])
    try:
        hash(val)
    except TypeError:
        return (t, id(val))
    return (t, val)


class HashConsMeta(ABCMeta):
    _intern_table: dict[Any, weakref.ref]

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = super().__call__(*args, **kwargs)
        key = obj.__hash_key__()
        table = cls._intern_table
        wr = table.get(key)
        if wr is not None:
            cached = wr()
            if cached is not None:
                return cached

        def remove(r: Any, table: dict = table, key: Any = key) -> None:
            if table.get(key) is r:
                del table[key]

        new_ref = weakref.ref(obj, remove)
        existing = table.setdefault(key, new_ref)
        if existing is not new_ref:
            cached = existing()
            if cached is not None:
                return cached
            table[key] = new_ref
        return obj


class HashCons(metaclass=HashConsMeta):
    """
    A base class for interned ("hash consed") objects. Construction returns a
    canonical instance for each distinct `__hash_key__`, so structurally equal
    objects are the same object and equality is identity (object's default
    id-based __eq__ and __hash__).
    """

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._intern_table = {}
        _hash_cons_types.add(cls)

    @abstractmethod
    def __hash_key__(self) -> Any:
        """Return a hashable key of the fields which identify this object."""
        ...

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        return self

    def __reduce__(self):
        return (
            type(self),
            tuple(getattr(self, f.name) for f in fields(self)),  # type: ignore[arg-type]
        )


class Term(HashCons):
    @abstractmethod
    def head(self) -> Any:
        """Return the head type of the S-expression."""
        ...

    @classmethod
    @abstractmethod
    def make_term(cls, head: Any, *children: Term) -> Self:
        """
        Construct a new term in the same family of terms with the given head type and
        children. This function should satisfy
        `x == x.make_term(x.head(), *x.children)`
        """
        ...


@dataclass(frozen=True, eq=False)
class TermTree(Term, ABC):
    @property
    @abstractmethod
    def children(self) -> list[Term]:
        """Return the children (AKA tail) of the S-expression."""
        ...

    def __hash_key__(self) -> Any:
        return tuple(
            [
                id(c) if type(c) in _hash_cons_types else hash_key_value(c)
                for c in self.children
            ]
        )


class LiteralTerm(Term, ABC):
    """
    A leaf term which wraps the constant `val`.
    """

    val: Any

    def __hash_key__(self) -> Any:
        return hash_key_value(self.val)


class CallTerm(TermTree, ABC):
    """
    A tree term which applies the operator held by the literal `op` to `args`.
    """

    op: LiteralTerm
    args: tuple[Term, ...]


def _get_repr(val: Any) -> str:
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return f"{val.__module__}.{val.__qualname__}"
    return repr(val)


def literal_repr(name: str, fields: dict[str, Any]) -> str:
    return (
        name + "(" + ", ".join([f"{k}={_get_repr(v)}" for k, v in fields.items()]) + ")"
    )
