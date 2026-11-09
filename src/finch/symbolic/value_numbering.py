"""
hash-consed values and the partition lattice for global value numbering.

a `Value` says how a value is computed: a `Leaf` is what a location (a variable,
a memory cell, ...) holds, a `Constant` is a literal and an `Apply` is a pure
operator applied to values. values are hash-consed, so equal values are the same
object.

a `ValueUniverse` is the finite set of values a program computes. a state is a
partition of the universe, where values in one class are equal at that program
point. it is a tuple giving each value the index of the first value in its class,
so equal partitions are equal tuples, and the class is the value number.

states are congruences: if the args of two applies are equal, so are the applies.
`meet` keeps the equalities two states share and `assign` updates a state for a
write. a write to a leaf changes every value that reads it, and the leaf joins
the class of the written value through a member the write didn't change. there
are finitely many partitions, so a dataflow analysis over them terminates, loops
included, without naming merged values (phis).
"""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hashcons import HashCons


class Value(HashCons):
    """how a value is computed"""


@dataclass(eq=False, frozen=True)
class Leaf(Value):
    """the value a location holds, e.g. a variable"""

    name: Hashable


@dataclass(eq=False, frozen=True)
class Constant(Value):
    """a literal, interned by its exact type and bits, so `0.0` and `-0.0` differ"""

    val: Any

    @staticmethod
    def canonicalize(a: Any, /) -> tuple[type, Hashable]:
        match a:
            case bool() | int():
                return type(a), a
            case float() | complex() | np.number() | np.bool_():
                return type(a), np.asarray(a).tobytes()
        return HashCons.canonicalize(a)


@dataclass(eq=False, frozen=True)
class Apply(Value):
    """a pure operator applied to values"""

    op: Hashable
    args: tuple[Value, ...]


class ValueUniverse:
    """
    a finite set of values, closed under taking args. a state is a tuple giving
    each value the index of the first value in its class.
    """

    def __init__(self, values: Iterable[Value]):
        self.values: list[Value] = []
        self.indices: dict[Value, int] = {}
        self.applies: list[tuple[int, Hashable, tuple[int, ...]]] = []
        self._reads: list[frozenset[Leaf]] = []
        for value in values:
            self._add(value)
        # the values a write to each leaf changes, the leaf first
        self.kills: dict[Leaf, list[int]] = {}
        for i, reads in enumerate(self._reads):
            for leaf in reads:
                self.kills.setdefault(leaf, []).append(i)

    def _add(self, value: Value) -> int:
        index = self.indices.get(value)
        if index is not None:
            return index
        reads: frozenset[Leaf] = frozenset()
        match value:
            case Apply(op, args):
                arg_ids = tuple(self._add(arg) for arg in args)
                reads = reads.union(*(self._reads[i] for i in arg_ids))
                self.applies.append((len(self.values), op, arg_ids))
            case Leaf():
                reads = frozenset((value,))
        index = self.indices[value] = len(self.values)
        self.values.append(value)
        self._reads.append(reads)
        return index

    def empty(self) -> tuple[int, ...]:
        """the state with no equalities"""
        return tuple(range(len(self.values)))

    def number(self, state: tuple[int, ...], value: Value) -> int | None:
        """the class of `value` in `state`, or None if it isn't in the universe"""
        index = self.indices.get(value)
        return None if index is None else state[index]

    def meet(
        self, state_1: tuple[int, ...], state_2: tuple[int, ...]
    ) -> tuple[int, ...]:
        """the equalities both states have"""
        # two congruences meet in a congruence, so there's nothing to close
        first: dict[tuple[int, int], int] = {}
        return tuple(
            first.setdefault(pair, i)
            for i, pair in enumerate(zip(state_1, state_2, strict=True))
        )

    def assign(
        self, state: tuple[int, ...], leaf: Leaf, value: Value | None = None
    ) -> tuple[int, ...]:
        """the state after writing `value` to `leaf`, or an unknown value if None"""
        killed = self.kills.get(leaf)
        if killed is None:
            return state
        target = self.indices[leaf]
        source = None if value is None else self.indices.get(value)
        if source == target:
            return state
        labels = list(state)
        for i in killed:
            labels[i] = len(labels) + i
        # members of the class of `value` that the write didn't change still
        # hold it, e.g. `t` in `t = add(x, 1); x = add(x, 1)`
        if source is not None and state[source] in labels:
            labels[target] = state[source]
        # only values reading the leaf can become congruent to something
        return self._close(labels, congruence=len(killed) > 1)

    def _close(self, labels: list[int], congruence: bool) -> tuple[int, ...]:
        """the state with classes given by `labels`, closed under congruence"""
        parent = list(range(len(labels)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        first: dict[int, int] = {}
        for i, label in enumerate(labels):
            parent[i] = first.setdefault(label, i)
        changed = congruence
        while changed:
            changed = False
            signatures: dict[tuple, int] = {}
            for i, op, args in self.applies:
                other = signatures.setdefault((op, *(find(j) for j in args)), i)
                root, other_root = find(i), find(other)
                if root != other_root:
                    # the smaller index stays the root, so roots are canonical
                    parent[max(root, other_root)] = min(root, other_root)
                    changed = True
        return tuple(find(i) for i in range(len(parent)))
