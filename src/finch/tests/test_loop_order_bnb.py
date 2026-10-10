import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.loop_orderer import loop_order_bnb, loop_order_greedy
from finch.autoschedule.loop_orderer.loop_order_bnb import (
    BFSLoopOrderer,
    BruteForceLoopOrderer,
    DFSLoopOrderer,
    loop_order_bfs,
    loop_order_brute_force,
    loop_order_dfs,
)
from finch.autoschedule.loop_orderer.loop_order_cost import loop_order_cost
from finch.autoschedule.loop_orderer.loop_order_greedy import (
    GreedyLoopOrderer,
    greedy_loop_order,
)
from finch.autoschedule.tensor_stats import DCStats, DCStatsFactory
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)


def test_bfs_and_dfs_are_no_worse_than_greedy():
    stats_factory = DCStatsFactory()
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
    stats: OrderedDict[Alias, DCStats] = OrderedDict(
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

    greedy_cost = loop_order_cost(expr, greedy, stats_factory, stats)
    bfs_cost = loop_order_cost(expr, bfs, stats_factory, stats)
    dfs_cost = loop_order_cost(expr, dfs, stats_factory, stats)

    assert set(bfs) == set(expr.fields())
    assert set(dfs) == set(expr.fields())
    assert bfs_cost <= greedy_cost
    assert dfs_cost <= greedy_cost
    assert bfs_cost == pytest.approx(dfs_cost)


def test_brute_force_is_no_worse_than_heuristics():
    stats_factory = DCStatsFactory()
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
    stats: OrderedDict[Alias, DCStats] = OrderedDict(
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

    greedy_cost = loop_order_cost(expr, greedy, stats_factory, stats)
    bfs_cost = loop_order_cost(expr, bfs, stats_factory, stats)
    dfs_cost = loop_order_cost(expr, dfs, stats_factory, stats)
    brute_cost = loop_order_cost(expr, brute, stats_factory, stats)

    assert set(brute) == set(expr.fields())
    assert brute_cost <= greedy_cost
    assert brute_cost <= bfs_cost
    assert brute_cost <= dfs_cost
    assert brute_cost == pytest.approx(
        min(
            loop_order_cost(expr, order, stats_factory, stats)
            for order in itertools.permutations(tuple(expr.fields()))
        )
    )


@pytest.mark.parametrize(
    "orderer,module,search",
    [
        (GreedyLoopOrderer, loop_order_greedy, "greedy_loop_order"),
        (BFSLoopOrderer, loop_order_bnb, "loop_order_bfs"),
        (DFSLoopOrderer, loop_order_bnb, "loop_order_dfs"),
        (BruteForceLoopOrderer, loop_order_bnb, "loop_order_brute_force"),
    ],
)
def test_loop_orderers_refresh_stats_of_redefined_aliases(
    monkeypatch, orderer, module, search
):
    # The second copy is the same node as the first, but scratch has been
    # redefined in between, so its statistics must not come from a cache.
    i = Field("i")
    a, b, scratch, out, result = map(
        HardAlias, ("a", "b", "scratch", "out", "result")
    )
    copy = Query(Table(out, (i,)), Table(scratch, (i,)))
    total = Aggregate(Literal(ffuncs.add), Literal(0.0), Table(out, (i,)), (i,))
    plan = Plan(
        (
            Query(Table(scratch, (i,)), Table(a, (i,))),
            copy,
            Query(Table(scratch, (i,)), Table(b, (i,))),
            copy,
            Query(Table(result, ()), total),
            Produces((result,)),
        )
    )
    seen = []

    def choose(expr, stats_factory, stats, output_vars, **kwargs):
        seen.append(stats[out].estimate_non_fill_values())
        return expr.fields()

    monkeypatch.setattr(module, search, choose)
    factory = DCStatsFactory()
    sparse = fl.asarray(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    dense = fl.asarray(np.ones(5))
    orderer().set_loop_orders(
        plan, {a: factory(sparse, (i,)), b: factory(dense, (i,))}, factory
    )
    assert seen == [5]
