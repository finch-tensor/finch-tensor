import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule import DefaultLogicFactorizer
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.cost_models import (
    PigeonCostModel,
    Prefix,
    Sum,
    TransposeCostModel,
)
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer.loop_order_greedy import (
    GreedyLoopOrderer,
    connected_loop_candidates,
    greedy_loop_order,
)
from finch.autoschedule.normalize import LogicNormalizer
from finch.autoschedule.tensor_stats import DCStats, DCStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Query,
    Table,
)
from finch.finch_notation.interpreter import NotationInterpreter


def test_empty_input_cost_matches_model():
    sf = DCStatsFactory()
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = HardAlias("A"), HardAlias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    bindings: dict[Alias, DCStats] = {
        a: sf(fl.asarray(np.ones((4, 4))), (i, j)),
        b: sf(fl.asarray(np.zeros((4, 4))), (j, k)),
    }

    model = Sum(PigeonCostModel(), TransposeCostModel())
    machine = StatsMachine(sf, dict(bindings))
    for order in ((i, j, k), (j, k, i)):
        query = Query(
            Table(HardAlias("out"), (i, j, k)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), expr, ()),
        )
        assert sum(
            model.predict_component_cost(Prefix(order[:n], expr), machine)
            for n in range(len(order) + 1)
        ) == model.predict_cost(fuse_loop_order(query, order), {}, bindings, sf)


def test_empty_relation_cost_matches_model():
    sf = DCStatsFactory()
    # l_ is l, precommit throws bad name error otehrwise
    i, j, k, l_, m = (Field(name) for name in "ijklm")
    a, b, c, d = (HardAlias(name) for name in "ABCD")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(a, (i, j)),
            Table(b, (j, k)),
            Table(c, (k, l_)),
            Table(d, (l_, m)),
        ),
    )
    bindings: OrderedDict[Alias, DCStats] = OrderedDict(
        {
            a: sf(fl.asarray(np.ones((2, 2))), (i, j)),
            b: sf(fl.asarray(np.ones((2, 2))), (j, k)),
            c: sf(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: sf(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    model = Sum(PigeonCostModel(), TransposeCostModel())
    machine = StatsMachine(sf, dict(bindings))
    for order in ((i, j, k, l_, m), (m, l_, k, j, i)):
        query = Query(
            Table(HardAlias("out"), (i, j, k, l_, m)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), expr, ()),
        )
        assert sum(
            model.predict_component_cost(Prefix(order[:n], expr), machine)
            for n in range(len(order) + 1)
        ) == model.predict_cost(fuse_loop_order(query, order), {}, bindings, sf)


def greedy_scheduler():
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(
                GreedyLoopOrderer(
                    DefaultLogicFormatter(
                        CompilerFormLowerer(LogicCompiler(NotationInterpreter()))
                    )
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
        result = fl.compute(
            fl.matmul(fl.matmul(fl.defer(fa), fl.defer(fb)), fl.defer(fc))
        )

    assert np.allclose(np.asarray(result), a @ b @ c)


def test_greedy_orderer_matches_reference_results():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(20, dtype=np.float64).reshape(4, 5)
    d = np.arange(12, dtype=np.float64).reshape(4, 3)
    fa, fb, fd = (fl.asarray(x) for x in (a, b, d))
    scheduler = greedy_scheduler()

    cases = [
        (lambda: fl.matmul(fl.defer(fa), fl.defer(fb)), a @ b),
        (lambda: fl.sum(fl.multiply(fl.defer(fa), fl.defer(fa))), (a * a).sum()),
        (lambda: fl.sum(fl.matmul(fl.defer(fa), fl.defer(fb)), axis=0), (a @ b).sum(0)),
        (
            lambda: fl.multiply(fl.permute_dims(fl.defer(fa), (1, 0)), fl.defer(fd)),
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
    i, j, k = Field("i"), Field("j"), Field("k")

    remaining = [k, j, i]
    assert connected_loop_candidates((), remaining) == remaining

    candidates = connected_loop_candidates((j,), [k, i])
    assert isinstance(candidates, list)
    assert candidates == [k, i]
    assert connected_loop_candidates((j,), [i, k]) == [i, k]
    assert connected_loop_candidates((i,), [k, j]) == [k, j]


def test_greedy_order_breaks_ties_by_field_order():
    """The two ends of a symmetric chain cost exactly the same, so which one
    greedy starts from is decided purely by the tie-break. Mirroring the chain
    must mirror the answer -- if candidates were held in a set, the winner would
    instead depend on ``Field`` hashes, which vary between processes.
    """
    sf = DCStatsFactory()
    i, j, k, m = (Field(name) for name in "ijkm")
    a, b, c = HardAlias("A"), HardAlias("B"), HardAlias("C")
    ones = fl.asarray(np.ones((4, 4)))

    forward = MapJoin(
        Literal(ffuncs.mul),
        (Table(a, (i, j)), Table(b, (j, k)), Table(c, (k, m))),
    )
    forward_bindings: dict[Alias, DCStats] = {
        a: sf(ones, (i, j)),
        b: sf(ones, (j, k)),
        c: sf(ones, (k, m)),
    }
    pigeon = PigeonCostModel()
    machine = StatsMachine(sf, dict(forward_bindings))

    # The chain's endpoints tie, so only the tie-break separates them.
    assert pigeon.predict_component_cost(
        Prefix((i,), forward), machine
    ) == pigeon.predict_component_cost(Prefix((m,), forward), machine)

    order = greedy_loop_order(forward, sf, forward_bindings)
    assert order[0] == i
    assert set(order) == {i, j, k, m}
    assert greedy_loop_order(forward, sf, forward_bindings) == order

    # Same chain written back-to-front: the tie must now break the other way.
    reverse = MapJoin(
        Literal(ffuncs.mul),
        (Table(c, (m, k)), Table(b, (k, j)), Table(a, (j, i))),
    )
    reverse_bindings: dict[Alias, DCStats] = {
        c: sf(ones, (m, k)),
        b: sf(ones, (k, j)),
        a: sf(ones, (j, i)),
    }
    reverse_order = greedy_loop_order(reverse, sf, reverse_bindings)
    assert reverse_order[0] == m
    assert set(reverse_order) == {i, j, k, m}
    assert greedy_loop_order(reverse, sf, reverse_bindings) == reverse_order


def test_greedy_avoids_transposing_a_sparse_input():
    """``B`` is a sparse diagonal stored ``(i, j)``, so reformatting it costs far
    more than reformatting the dense ``A`` stored ``(j, i)``. Read and write
    costs tie, and the expression's field order puts ``j`` first, so only the
    transpose cost can steer greedy to the cheaper ``(i, j)``.
    """
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, b = HardAlias("A"), HardAlias("B")
    n = 16
    diagonal = np.zeros((n, n))
    diagonal[np.arange(n), np.arange(n)] = 1.0

    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (j, i)), Table(b, (i, j))))
    bindings: dict[Alias, DCStats] = {
        a: sf(fl.asarray(np.ones((n, n))), (j, i)),
        b: sf(fl.asarray(diagonal), (i, j)),
    }
    assert tuple(dict.fromkeys(expr.fields())) == (j, i)

    pigeon = PigeonCostModel()
    transpose = TransposeCostModel()
    model = Sum(pigeon, transpose)
    machine = StatsMachine(sf, dict(bindings))
    # Reads and writes alone cannot separate the two, so the field-order
    # tie-break would pick j; transposing the sparse B is what makes that order
    # the expensive one.
    assert pigeon.predict_component_cost(
        Prefix((j,), expr), machine
    ) == pigeon.predict_component_cost(Prefix((i,), expr), machine)
    assert transpose.predict_component_cost(
        Prefix((j,), expr), machine
    ) > transpose.predict_component_cost(Prefix((i,), expr), machine)
    costs = [
        sum(
            model.predict_component_cost(Prefix(order[:n], expr), machine)
            for n in range(len(order) + 1)
        )
        for order in ((i, j), (j, i))
    ]
    assert costs[0] < costs[1]

    assert greedy_loop_order(expr, sf, bindings) == (i, j)


def test_prefix_totals_match_program_cost():
    sf = DCStatsFactory()
    model = Sum(PigeonCostModel(), TransposeCostModel())
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = HardAlias("A"), HardAlias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    ones = fl.asarray(np.ones((3, 3)))
    bindings: dict[Alias, DCStats] = {a: sf(ones, (i, j)), b: sf(ones, (j, k))}
    machine = StatsMachine(sf, dict(bindings))

    for output_vars in (None, (), (i, k), (k, i)):
        out_idxs = expr.fields() if output_vars is None else output_vars
        reduced = tuple(idx for idx in expr.fields() if idx not in out_idxs)
        for order in itertools.permutations((i, j, k)):
            prefix_total = sum(
                model.predict_component_cost(
                    Prefix(order[:n], expr, output_vars), machine
                )
                for n in range(len(order) + 1)
            )
            query = Query(
                Table(HardAlias("out"), out_idxs),
                Aggregate(Literal(ffuncs.add), Literal(0.0), expr, reduced),
            )
            assert prefix_total == pytest.approx(
                model.predict_cost(fuse_loop_order(query, order), {}, bindings, sf)
            )
