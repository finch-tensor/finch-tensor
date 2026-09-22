import copy
import gc
from dataclasses import dataclass
from typing import Any

import numpy as np

from finch.symbolic import HashCons
from finch.symbolic.hashcons import _table


@dataclass(eq=False, frozen=True)
class Leaf(HashCons):
    name: str

    def __hash_keys__(self) -> tuple:
        return (self.name,)


@dataclass(eq=False, frozen=True)
class OtherLeaf(HashCons):
    name: str

    def __hash_keys__(self) -> tuple:
        return (self.name,)


@dataclass(eq=False, frozen=True)
class Node(HashCons):
    left: Any
    right: Any

    def __hash_keys__(self) -> tuple:
        return (self.left, self.right)


@dataclass(eq=False, frozen=True)
class Box(HashCons):
    val: Any

    def __hash_keys__(self) -> tuple:
        try:
            hash(self.val)
        except TypeError:
            return (type(self.val), id(self.val))
        return (type(self.val), self.val)


def test_interned_leaves():
    assert Leaf("a") is Leaf("a")
    assert Leaf("a") is Leaf(name="a")
    assert Leaf("a") is not Leaf("b")
    assert Leaf("a") is not OtherLeaf("a")


def test_interned_trees():
    def build():
        return Node(Node(Leaf("a"), Leaf("b")), Leaf("c"))

    t1, t2 = build(), build()
    assert t1 is t2
    assert t1 == t2
    assert hash(t1) == hash(t2)
    assert t1.left is t2.left


def test_equality_is_identity():
    assert Leaf("a") == Leaf("a")
    assert Leaf("a") != Leaf("b")
    assert Leaf("a") != OtherLeaf("a")
    assert len({Leaf("a"), Leaf("a"), Leaf("b")}) == 2


def test_key_rule_decides_identity():
    assert Box(1) is Box(1)
    assert Box(1) is not Box(1.0)
    assert Box(1) is not Box(True)
    assert Box(1) is not Box(np.int64(1))


def test_unhashable_payload_is_interned_by_pointer():
    a = np.ones(3)
    assert Box(a) is Box(a)
    assert Box(a) is not Box(np.ones(3))


def test_copy_returns_self():
    t = Node(Leaf("a"), Leaf("b"))
    assert copy.copy(t) is t
    assert copy.deepcopy(t) is t


def test_table_is_weak():
    t = Leaf("garbage_collected_leaf")
    key = (Leaf, *t.__hash_keys__())
    assert _table[key] is t
    del t
    gc.collect()
    assert key not in _table
