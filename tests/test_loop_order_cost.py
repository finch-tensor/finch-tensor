import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule import DefaultLogicOptimizer
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_order_cost import (
    cost_of_reformat,
    get_conjunctive_and_disjunctive_inputs,
    get_loop_lookups,
    get_reformat_set,
    loop_order_cost,
)
from finch.autoschedule.loop_order_greedy import (
    GreedyLoopOrderer,
    connected_loop_candidates,
    greedy_loop_order,
    transpose_penalty,
)
from finch.autoschedule.normalize import LogicNormalizer
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.finch_logic import Alias, Field, Literal, MapJoin, Table
from finch.finch_notation.interpreter import NotationInterpreter


def test_empty_input_ordered_first_is_cheaper():
    sf = DCStatsFactory()
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    bindings = {
        a: sf(fl.asarray(np.ones((4, 4))), (i, j)),
        b: sf(fl.asarray(np.zeros((4, 4))), (j, k)),
    }

    dense_first = loop_order_cost(expr, (i, j, k), sf, bindings)
    empty_first = loop_order_cost(expr, (j, k, i), sf, bindings)

    assert empty_first < dense_first


def test_empty_relation():
    sf = DCStatsFactory()
    # l_ is l, precommit throws bad name error otehrwise
    i, j, k, l_, m = (Field(name) for name in "ijklm")
    a, b, c, d = (Alias(name) for name in "ABCD")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(a, (i, j)),
            Table(b, (j, k)),
            Table(c, (k, l_)),
            Table(d, (l_, m)),
        ),
    )
    bindings = OrderedDict(
        {
            a: sf(fl.asarray(np.ones((2, 2))), (i, j)),
            b: sf(fl.asarray(np.ones((2, 2))), (j, k)),
            c: sf(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: sf(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    forward = loop_order_cost(expr, (i, j, k, l_, m), sf, bindings)
    reverse = loop_order_cost(expr, (m, l_, k, j, i), sf, bindings)
    assert forward > reverse


def greedy_scheduler():
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicOptimizer(
                GreedyLoopOrderer(
                    DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
                )
            )
        )
    )


def test_greedy_orderer_multi_query_plan():
    """A plan with more than one aggregate query reads an alias produced by an
    earlier query; those intermediates must be bound before they can be costed.
    """
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(20, dtype=np.float64).reshape(4, 5)
    c = np.arange(30, dtype=np.float64).reshape(5, 6)
    fa, fb, fc = (fl.asarray(x) for x in (a, b, c))

    with fl.with_default_scheduler(greedy_scheduler()):
        result = fl.compute(fl.matmul(fl.matmul(fl.lazy(fa), fl.lazy(fb)), fl.lazy(fc)))

    assert np.allclose(np.asarray(result), a @ b @ c)


def test_greedy_orderer_matches_reference_results():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(20, dtype=np.float64).reshape(4, 5)
    d = np.arange(12, dtype=np.float64).reshape(4, 3)
    fa, fb, fd = (fl.asarray(x) for x in (a, b, d))
    scheduler = greedy_scheduler()

    cases = [
        (lambda: fl.matmul(fl.lazy(fa), fl.lazy(fb)), a @ b),
        (lambda: fl.sum(fl.multiply(fl.lazy(fa), fl.lazy(fa))), (a * a).sum()),
        (lambda: fl.sum(fl.matmul(fl.lazy(fa), fl.lazy(fb)), axis=0), (a @ b).sum(0)),
        (
            lambda: fl.multiply(fl.permute_dims(fl.lazy(fa), (1, 0)), fl.lazy(fd)),
            a.T * d,
        ),
    ]
    for build, expected in cases:
        with fl.with_default_scheduler(scheduler):
            result = fl.compute(build())
        assert np.allclose(np.asarray(result), expected)


def test_candidates_preserve_remaining_order():
    """Candidates must stay ordered: iterating a set would make tie-breaking
    depend on ``Field`` hashes, which differ between processes.
    """
    sf = DCStatsFactory()
    i, j, k = Field("i"), Field("j"), Field("k")
    ones = fl.asarray(np.ones((4, 4)))
    stats = [sf(ones, (i, j)), sf(ones, (j, k))]

    remaining = [k, j, i]
    assert connected_loop_candidates((), remaining, stats, []) == remaining

    # j touches both tensors, so i and k are both connected candidates and the
    # result must follow the order they were given in.
    candidates = connected_loop_candidates((j,), [k, i], stats, [])
    assert isinstance(candidates, list)
    assert candidates == [k, i]
    assert connected_loop_candidates((j,), [i, k], stats, []) == [i, k]

    # i only touches the first tensor, so k is unreachable from it.
    assert connected_loop_candidates((i,), [k, j], stats, []) == [j]


def test_greedy_order_breaks_ties_by_field_order():
    """On a symmetric chain every candidate costs the same, so the order is
    decided purely by the tie-break and must equal the expression's field order.
    """
    sf = DCStatsFactory()
    i, j, k, m = (Field(name) for name in "ijkm")
    a, b, c = Alias("A"), Alias("B"), Alias("C")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (Table(a, (i, j)), Table(b, (j, k)), Table(c, (k, m))),
    )
    ones = fl.asarray(np.ones((4, 4)))
    bindings = {a: sf(ones, (i, j)), b: sf(ones, (j, k)), c: sf(ones, (k, m))}

    order = greedy_loop_order(expr, sf, bindings)
    assert order == tuple(dict.fromkeys(expr.fields()))
    # Repeated calls must agree.
    assert greedy_loop_order(expr, sf, bindings) == order


def _dedup_stats(expr, sf, bindings):
    conjuncts, disjuncts = get_conjunctive_and_disjunctive_inputs(
        expr, sf, dict(bindings), {}
    )
    input_stats = []
    for stat in conjuncts + disjuncts:
        if not any(stat is seen for seen in input_stats):
            input_stats.append(stat)
    return conjuncts, disjuncts, input_stats


def test_greedy_avoids_transposing_an_input():
    """``B`` is stored ``(i, j)``, so looping ``j`` before ``i`` forces a
    reformat. Lookups alone tie, and the expression's field order puts ``j``
    first, so only the transpose cost can steer greedy to ``(i, j)``.
    """
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (j,)), Table(b, (i, j))))
    bindings = {
        a: sf(fl.asarray(np.ones(8)), (j,)),
        b: sf(fl.asarray(np.ones((8, 8))), (i, j)),
    }

    # Without the transpose cost the tie-break would follow the field order.
    assert tuple(dict.fromkeys(expr.fields())) == (j, i)
    conjuncts, disjuncts, input_stats = _dedup_stats(expr, sf, bindings)
    assert get_loop_lookups((j,), conjuncts, disjuncts, sf) == get_loop_lookups(
        (i,), conjuncts, disjuncts, sf
    )

    # Only (j, i) forces a reformat, so it must be the more expensive one.
    assert transpose_penalty(input_stats, (j,), frozenset()) > 0.0
    assert transpose_penalty(input_stats, (i,), frozenset()) == 0.0

    assert greedy_loop_order(expr, sf, bindings) == (i, j)


def test_transpose_penalty_totals_match_loop_order_cost():
    """The penalty is charged incrementally as each loop is appended; summing it
    over a full order must equal the reformat cost ``loop_order_cost`` charges.
    """
    sf = DCStatsFactory()
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    ones = fl.asarray(np.ones((3, 3)))
    bindings = {a: sf(ones, (i, j)), b: sf(ones, (j, k))}
    _, _, input_stats = _dedup_stats(expr, sf, bindings)

    saw_nonzero = False
    for order in itertools.permutations((i, j, k)):
        charged: frozenset[int] = frozenset()
        incremental = 0.0
        for n in range(1, len(order) + 1):
            incremental += transpose_penalty(input_stats, order[:n], charged)
            charged = get_reformat_set(input_stats, order[:n])

        expected = sum(
            cost_of_reformat(input_stats[x])
            for x in get_reformat_set(input_stats, order)
        )
        assert incremental == pytest.approx(expected)
        saw_nonzero |= expected > 0.0

    assert saw_nonzero, "expected at least one order to force a reformat"
