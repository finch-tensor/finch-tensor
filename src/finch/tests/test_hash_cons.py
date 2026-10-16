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


@dataclass(eq=False, frozen=True)
class OtherLeaf(HashCons):
    name: str


@dataclass(eq=False, frozen=True)
class Node(HashCons):
    left: Any
    right: Any


@dataclass(eq=False, frozen=True)
class Box(HashCons):
    val: Any


def test_interned_leaves():
    assert Leaf("a") is Leaf("a")
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


def test_copy_returns_self():
    t = Node(Leaf("a"), Leaf("b"))
    assert copy.copy(t) is t
    assert copy.deepcopy(t) is t


def test_non_hashable():
    arr = np.zeros(())
    arr2 = np.zeros(())
    assert Box(arr) is Box(arr)
    assert Box(arr) is not Box(arr2)


def test_nan():
    assert Box(np.nan) is Box(np.nan)
    assert Box(np.complex128(np.nan, 5.0)) is Box(np.complex128(np.nan, 5.0))
    assert Box(np.complex128(5.0, np.nan)) is Box(np.complex128(5.0, np.nan))
    assert Box(np.complex128(np.nan, np.nan)) is Box(np.complex128(np.nan, np.nan))
    assert Box(np.complex128(5.0, np.nan)) is not Box(np.complex128(np.nan, 5.0))
    assert Box(np.complex128(5.0, np.nan)) is not Box(np.complex128(np.nan, np.nan))


def test_table_is_weak():
    t = Leaf("garbage_collected_leaf")
    key = Leaf.hash_key("garbage_collected_leaf")
    assert _table[key] is t
    del t
    gc.collect()
    assert key not in _table
