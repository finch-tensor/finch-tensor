import copy
import gc
import operator

import numpy as np

import finch.finch_assembly as asm
import finch.finch_notation as ntn
from finch.finch_logic import Alias, Field, Literal, MapJoin, Table


def test_interned_leaves():
    assert Alias("A") is Alias("A")
    assert Field("i") is Field("i")
    assert Alias("A") is not Field("A")
    assert Literal(1) is Literal(1)


def test_interned_trees():
    def build():
        return MapJoin(
            Literal(operator.add),
            (
                Table(Alias("B"), (Field("i"),)),
                Table(Alias("C"), (Field("i"),)),
            ),
        )

    m1 = build()
    m2 = build()
    assert m1 is m2
    assert m1 == m2
    assert hash(m1) == hash(m2)


def test_literal_type_tag():
    assert Literal(0) is not Literal(0.0)
    assert Literal(True) is not Literal(1)
    assert Literal(1) is not Literal(np.int64(1))


def test_unhashable_literal():
    a = np.ones(3)
    assert Literal(a) is Literal(a)
    assert Literal(a) is not Literal(np.ones(3))


def test_make_term_round_trip():
    t = Table(Alias("A"), (Field("i"), Field("j")))
    assert t.make_term(t.head(), *t.children) is t


def test_copy_returns_self():
    t = Table(Alias("A"), (Field("i"),))
    assert copy.copy(t) is t
    assert copy.deepcopy(t) is t


def test_interning_is_weak():
    t = Table(Alias("garbage_collected_alias"), (Field("i"),))
    table = type(t)._intern_table
    key = t.__hash_key__()
    assert table[key]() is t
    del t
    gc.collect()
    assert key not in table


def test_interned_across_hierarchies():
    assert asm.Variable("x", np.int64) is asm.Variable("x", np.int64)
    assert asm.Variable("x", np.int64) is not asm.Variable("x", np.float64)
    assert ntn.Literal(2) is ntn.Literal(2)
    assert ntn.Literal(2) is not asm.Literal(2)
