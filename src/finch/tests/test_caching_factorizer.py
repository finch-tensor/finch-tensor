import numpy as np

import finch as ft
from finch import ffuncs
from finch.autoschedule import INTERPRET_NOTATION, LogicNormalizer
from finch.autoschedule.factorizer.caching import CachingFactorizer, query_key
from finch.finch_logic import (
    Aggregate,
    Field,
    Literal,
    LogicEvaluator,
    MapJoin,
    Table,
)
from finch.symbolic import UnvalidatedForm


class CountingEvaluator(UnvalidatedForm, LogicEvaluator):
    def __init__(self, ctx: LogicEvaluator):
        self.ctx = ctx
        self.calls = 0

    def lower(self, term, bindings=None):
        self.calls += 1
        return self.ctx(term, bindings)


def _matmul(x, y, idx):
    return Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (x, y)),
        (idx,),
    )


def _inputs():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    rng = np.random.default_rng(0)
    A, B, C = rng.random((3, 4)), rng.random((4, 5)), rng.random((5, 2))
    a = Table(Literal(ft.asarray(A)), (i, j))
    b = Table(Literal(ft.asarray(B)), (j, k))
    c = Table(Literal(ft.asarray(C)), (k, m))
    return (i, j, k, m), (A, B, C), (a, b, c)


def test_caching_factorizer_computes_programs():
    (_, j, k, _), (A, B, C), (a, b, c) = _inputs()
    scheduler = LogicNormalizer(CachingFactorizer(INTERPRET_NOTATION))
    result = scheduler(_matmul(_matmul(a, b, j), c, k))
    assert np.allclose(result.tns.to_numpy(), A @ B @ C)


def test_caching_factorizer_reuses_queries():
    (i, j, k, _), (A, B, _), (a, b, _) = _inputs()
    D = np.ones((3, 5))
    ab = _matmul(a, b, j)
    # A sum under an add stays where it is, so Galley computes A @ B on its own.
    plus = MapJoin(Literal(ffuncs.add), (ab, Table(Literal(ft.asarray(D)), (i, k))))

    alone = CountingEvaluator(INTERPRET_NOTATION)
    scheduler = LogicNormalizer(CachingFactorizer(alone))
    assert np.allclose(scheduler(plus).tns.to_numpy(), A @ B + D)
    calls = alone.calls
    scheduler(plus)
    assert alone.calls == calls

    shared = CountingEvaluator(INTERPRET_NOTATION)
    scheduler = LogicNormalizer(CachingFactorizer(shared))
    scheduler(ab)
    before = shared.calls
    assert np.allclose(scheduler(plus).tns.to_numpy(), A @ B + D)
    assert shared.calls - before < calls


def _tables(*idxs):
    return tuple(Table(Literal(ft.asarray(np.eye(2))), idx) for idx in idxs)


def test_query_key_is_cnf():
    i, j = Field("i"), Field("j")
    a, b, c = _tables((i, j), (i, j), (i, j))
    and_, or_ = Literal(ffuncs.and_), Literal(ffuncs.or_)
    dnf = MapJoin(or_, (a, MapJoin(and_, (b, c))))
    cnf = MapJoin(and_, (MapJoin(or_, (a, b)), MapJoin(or_, (a, c))))
    assert query_key(dnf, (i, j)) == query_key(cnf, (i, j))


def test_query_key_distributes_products_over_sums():
    i, j = Field("i"), Field("j")
    a, b, c = _tables((i, j), (i, j), (i, j))
    add, mul = Literal(ffuncs.add), Literal(ffuncs.mul)
    product = MapJoin(mul, (MapJoin(add, (a, b)), c))
    total = MapJoin(add, (MapJoin(mul, (a, c)), MapJoin(mul, (b, c))))
    assert query_key(product, (i, j)) == query_key(total, (i, j))


def test_query_key_keeps_the_fields_of_annihilated_operands():
    i, j = Field("i"), Field("j")
    (a,) = _tables((i, j))
    key = query_key(MapJoin(Literal(ffuncs.mul), (a, Literal(0.0))), (i, j))
    assert len(key.fields()) == 2


def test_query_key_ignores_names_of_fields():
    i, j, k, m, n, p = (Field(c) for c in "ijkmnp")
    A, B = (Literal(ft.asarray(np.eye(2))) for _ in range(2))

    def matmul(row, idx, col):
        return Aggregate(
            Literal(ffuncs.add),
            Literal(0.0),
            MapJoin(Literal(ffuncs.mul), (Table(A, (row, idx)), Table(B, (idx, col)))),
            (idx,),
        )

    assert query_key(matmul(i, j, k), (i, k)) == query_key(matmul(n, m, p), (n, p))
    assert query_key(matmul(i, j, k), (i, k)) != query_key(matmul(i, j, k), (k, i))
