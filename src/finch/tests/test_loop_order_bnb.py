import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.cost_models import (
    Concat,
    LinearPredictor,
    LoopCostModel,
    PigeonCostModel,
    Prefix,
    QuadraticPredictor,
    Sum,
    TransposeCostModel,
)
from finch.autoschedule.loop_orderer.loop_order_bnb import (
    BFSLoopOrderer,
    BruteForceLoopOrderer,
    DFSLoopOrderer,
    loop_order_bfs,
    loop_order_brute_force,
    loop_order_dfs,
)
from finch.autoschedule.loop_orderer.loop_order_greedy import (
    GreedyLoopOrderer,
    greedy_loop_order,
)
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats import DCStatsFactory, UniformStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    Table,
)


def test_bfs_and_dfs_are_no_worse_than_greedy():
    stats_factory = DCStatsFactory()
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
    stats = OrderedDict(
        {
            a: stats_factory(fl.asarray(np.ones((2, 2))), (i, j)),
            b: stats_factory(fl.asarray(np.ones((2, 2))), (j, k)),
            c: stats_factory(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: stats_factory(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    greedy = greedy_loop_order(expr, stats_factory, stats)
    bfs = loop_order_bfs(expr, stats_factory, stats)
    dfs = loop_order_dfs(expr, stats_factory, stats)

    model = BFSLoopOrderer().cost_model

    def cost(order):
        query = Query(
            Table(Alias("out"), expr.fields()),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Reorder(expr, order), ()),
        )
        return model.predict_cost(query, {}, stats, stats_factory)

    greedy_cost, bfs_cost, dfs_cost = map(cost, (greedy, bfs, dfs))

    assert set(bfs) == set(expr.fields())
    assert set(dfs) == set(expr.fields())
    assert bfs_cost <= greedy_cost
    assert dfs_cost <= greedy_cost
    assert bfs_cost == pytest.approx(dfs_cost)


def test_brute_force_is_no_worse_than_heuristics():
    stats_factory = DCStatsFactory()
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
    stats = OrderedDict(
        {
            a: stats_factory(fl.asarray(np.ones((2, 2))), (i, j)),
            b: stats_factory(fl.asarray(np.ones((2, 2))), (j, k)),
            c: stats_factory(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: stats_factory(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    greedy = greedy_loop_order(expr, stats_factory, stats)
    bfs = loop_order_bfs(expr, stats_factory, stats)
    dfs = loop_order_dfs(expr, stats_factory, stats)
    brute = loop_order_brute_force(expr, stats_factory, stats)

    model = BruteForceLoopOrderer().cost_model

    def cost(order):
        query = Query(
            Table(Alias("out"), expr.fields()),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Reorder(expr, order), ()),
        )
        return model.predict_cost(query, {}, stats, stats_factory)

    greedy_cost, bfs_cost, dfs_cost, brute_cost = map(cost, (greedy, bfs, dfs, brute))

    assert set(brute) == set(expr.fields())
    assert brute_cost <= greedy_cost
    assert brute_cost <= bfs_cost
    assert brute_cost <= dfs_cost
    assert brute_cost == pytest.approx(
        min(cost(order) for order in itertools.permutations(tuple(expr.fields())))
    )


@pytest.mark.parametrize(
    "search", [loop_order_bfs, loop_order_dfs, loop_order_brute_force]
)
def test_loop_search_minimizes_nonlinear_component_cost(search):
    sf = DCStatsFactory()
    i, j, k = (Field(name) for name in "ijk")
    a, b, out = (Alias(name) for name in ("a", "b", "out"))
    stats = {
        a: sf(fl.asarray(np.ones((2, 3))), (i, j)),
        b: sf(fl.asarray(np.ones((3, 4))), (j, k)),
    }
    arg = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (j,))
    model = Concat(
        PigeonCostModel(),
        TransposeCostModel(),
        predictor=QuadraticPredictor(np.ones(25)),
    )

    def cost(order):
        query = Query(
            Table(out, (i, k)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Reorder(arg, order), (j,)),
        )
        features = model.get_features(query, {}, stats, sf)
        return sum(row.sum() ** 2 for row in features)

    chosen = search(expr, sf, stats, (i, k), cost_model=model)
    expected = min(
        cost(order)
        for order in itertools.permutations((i, j, k))
        if order.index(i) < order.index(k)
    )
    assert cost(chosen) == expected


class PreferFieldCostModel(LoopCostModel, LoopOrderedForm):
    @property
    def n_features(self) -> int:
        return 1

    def __init__(self, field):
        super().__init__(LinearPredictor(np.array([1.0])))
        self.field = field

    def get_component_features(self, component, machine):
        prefix = component.prefix
        if not prefix:
            return np.array([[7.0]])
        return np.array(
            [[10.0 if len(prefix) == 1 and prefix[0] != self.field else 0.0]]
        )


@pytest.mark.parametrize(
    "search, orderer",
    [
        (greedy_loop_order, GreedyLoopOrderer),
        (loop_order_bfs, BFSLoopOrderer),
        (loop_order_dfs, DFSLoopOrderer),
        (loop_order_brute_force, BruteForceLoopOrderer),
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_loop_orderers_use_selected_model(search, orderer, inplace):
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    expr = Table(a, (i, j))
    stats = {a: sf(fl.asarray(np.ones((16, 16))), (i, j))}
    model = PreferFieldCostModel(j)
    assert search(expr, sf, stats, cost_model=model) == (j, i)
    if inplace:
        stats[out] = sf(fl.asarray(np.ones(())), ())
        query = QueryInto(Table(out, ()), Literal(ffuncs.add), expr)
    else:
        query = Query(
            Table(out, ()), Aggregate(Literal(ffuncs.add), Literal(0.0), expr, (i, j))
        )
    plan = Plan((query, Produces((out,))))
    scheduler = orderer(cost_model=model)
    ordered = scheduler.set_loop_orders(plan, stats, sf)
    assert scheduler.cost_model is model
    assert model.predict_cost(ordered, {}, stats, sf) == 7
    match ordered.bodies[0]:
        case Query(_, Aggregate(_, _, Reorder(_, order), _)) | QueryInto(
            _, _, Reorder(_, order)
        ):
            assert order == (j, i)
        case _:
            pytest.fail("Expected an ordered query")
    assert isinstance(orderer().cost_model, Sum)


@pytest.mark.parametrize(
    "search",
    [greedy_loop_order, loop_order_bfs, loop_order_dfs, loop_order_brute_force],
)
def test_loop_orderers_respect_output_order(search):
    sf = UniformStatsFactory()
    i, j, unit = Field("i"), Field("j"), Field("unit")
    a = Alias("a")
    stats = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), Table(a, (i, j)), (j,))
    order = search(expr, sf, stats, (unit, i), cost_model=PreferFieldCostModel(j))
    assert order == (j, unit, i)


@pytest.mark.parametrize(
    "op, init", [(ffuncs.add, 0.0), (ffuncs.mul, 1.0), (ffuncs.max, -float("inf"))]
)
def test_prefix_cost_uses_reduction_statistics(op, init):
    sf = UniformStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    stats = {a: sf(fl.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), (i, j))}
    table = Table(a, (i, j))
    expr = Aggregate(Literal(op), Literal(init), table, (j,))
    model = BFSLoopOrderer().cost_model
    machine = StatsMachine(sf, dict(stats))
    for order in ((i, j), (j, i)):
        query = Query(
            Table(out, (i,)),
            Aggregate(Literal(op), Literal(init), Reorder(table, order), (j,)),
        )
        assert sum(
            model.predict_component_cost(Prefix(order[:n], expr, (i,)), machine)
            for n in range(len(order) + 1)
        ) == pytest.approx(model.predict_cost(query, {}, stats, sf))
