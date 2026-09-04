import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule import DefaultLogicOptimizer, loop_order_greedy
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_order_cost import (
    cost_of_reformat,
    get_conjunctive_and_disjunctive_inputs,
    get_prefix_cost,
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
from finch.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    Table,
)
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

    candidates = connected_loop_candidates((j,), [k, i], stats, [])
    assert isinstance(candidates, list)
    assert candidates == [k, i]
    assert connected_loop_candidates((j,), [i, k], stats, []) == [i, k]
    assert connected_loop_candidates((i,), [k, j], stats, []) == [k, j]


def test_greedy_order_breaks_ties_by_field_order():
    """The two ends of a symmetric chain cost exactly the same, so which one
    greedy starts from is decided purely by the tie-break. Mirroring the chain
    must mirror the answer -- if candidates were held in a set, the winner would
    instead depend on ``Field`` hashes, which vary between processes.
    """
    sf = DCStatsFactory()
    i, j, k, m = (Field(name) for name in "ijkm")
    a, b, c = Alias("A"), Alias("B"), Alias("C")
    ones = fl.asarray(np.ones((4, 4)))

    forward = MapJoin(
        Literal(ffuncs.mul),
        (Table(a, (i, j)), Table(b, (j, k)), Table(c, (k, m))),
    )
    forward_bindings = {a: sf(ones, (i, j)), b: sf(ones, (j, k)), c: sf(ones, (k, m))}
    conjuncts, disjuncts, _ = _dedup_stats(forward, sf, forward_bindings)

    # The chain's endpoints tie, so only the tie-break separates them.
    assert get_prefix_cost((i,), conjuncts, disjuncts, sf) == get_prefix_cost(
        (m,), conjuncts, disjuncts, sf
    )

    order = greedy_loop_order(forward, sf, forward_bindings)
    assert order[0] == i
    assert set(order) == {i, j, k, m}
    assert greedy_loop_order(forward, sf, forward_bindings) == order

    # Same chain written back-to-front: the tie must now break the other way.
    reverse = MapJoin(
        Literal(ffuncs.mul),
        (Table(c, (m, k)), Table(b, (k, j)), Table(a, (j, i))),
    )
    reverse_bindings = {c: sf(ones, (m, k)), b: sf(ones, (k, j)), a: sf(ones, (j, i))}
    reverse_order = greedy_loop_order(reverse, sf, reverse_bindings)
    assert reverse_order[0] == m
    assert set(reverse_order) == {i, j, k, m}
    assert greedy_loop_order(reverse, sf, reverse_bindings) == reverse_order


def _dedup_stats(expr, sf, bindings):
    conjuncts, disjuncts = get_conjunctive_and_disjunctive_inputs(
        expr, sf, dict(bindings), {}
    )
    input_stats = []
    for stat in conjuncts + disjuncts:
        if not any(stat is seen for seen in input_stats):
            input_stats.append(stat)
    return conjuncts, disjuncts, input_stats


def test_greedy_avoids_transposing_a_sparse_input(monkeypatch):
    """``B`` is a sparse diagonal stored ``(i, j)``, so reformatting it costs far
    more than reformatting the dense ``A`` stored ``(j, i)``. Read and write
    costs tie, and the expression's field order puts ``j`` first, so only the
    transpose cost can steer greedy to the cheaper ``(i, j)``.
    """
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, b = Alias("A"), Alias("B")
    n = 16
    diagonal = np.zeros((n, n))
    diagonal[np.arange(n), np.arange(n)] = 1.0

    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (j, i)), Table(b, (i, j))))
    bindings = {
        a: sf(fl.asarray(np.ones((n, n))), (j, i)),
        b: sf(fl.asarray(diagonal), (i, j)),
    }
    assert tuple(dict.fromkeys(expr.fields())) == (j, i)

    conjuncts, disjuncts, input_stats = _dedup_stats(expr, sf, bindings)
    # Reads and writes alone cannot separate the two, so the field-order
    # tie-break would pick j; transposing the sparse B is what makes that order
    # the expensive one.
    assert get_prefix_cost((j,), conjuncts, disjuncts, sf) == get_prefix_cost(
        (i,), conjuncts, disjuncts, sf
    )
    assert transpose_penalty(input_stats, (j,), frozenset()) > transpose_penalty(
        input_stats, (i,), frozenset()
    )
    assert loop_order_cost(expr, (i, j), sf, dict(bindings)) < loop_order_cost(
        expr, (j, i), sf, dict(bindings)
    )

    assert greedy_loop_order(expr, sf, bindings) == (i, j)

    # Drop the transpose term and greedy settles for the more expensive order.
    monkeypatch.setattr(loop_order_greedy, "transpose_penalty", lambda *args: 0.0)
    assert greedy_loop_order(expr, sf, bindings) == (j, i)


def test_transpose_penalty_totals_match_loop_order_cost():
    """Greedy scores one prefix at a time, charging each reformat at the step
    where it becomes unavoidable. Summed over a full order, that must reproduce
    the order's ``loop_order_cost`` exactly -- otherwise greedy is minimising
    something other than the cost model it claims to use.
    """
    sf = DCStatsFactory()
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    ones = fl.asarray(np.ones((3, 3)))
    bindings = {a: sf(ones, (i, j)), b: sf(ones, (j, k))}
    conjuncts, disjuncts, input_stats = _dedup_stats(expr, sf, bindings)

    saw_reformat = False
    for output_vars in (None, (i, k), (k, i)):
        for order in itertools.permutations((i, j, k)):
            charged: frozenset[int] = frozenset()
            greedy_score = 0.0
            reformats = 0.0
            for n in range(1, len(order) + 1):
                prefix = order[:n]
                greedy_score += get_prefix_cost(
                    prefix, conjuncts, disjuncts, sf, output_vars
                )
                penalty = transpose_penalty(input_stats, prefix, charged)
                greedy_score += penalty
                reformats += penalty
                charged = get_reformat_set(input_stats, prefix)

            assert greedy_score == pytest.approx(
                loop_order_cost(expr, order, sf, dict(bindings), output_vars)
            )
            # The incremental reformat charges add up to the one-shot total.
            assert reformats == pytest.approx(
                sum(
                    cost_of_reformat(input_stats[x])
                    for x in get_reformat_set(input_stats, order)
                )
            )
            saw_reformat |= reformats > 0.0

    assert saw_reformat, "expected at least one order to force a reformat"
