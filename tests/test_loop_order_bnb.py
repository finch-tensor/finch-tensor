from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.loop_order_bnb import loop_order_bfs, loop_order_dfs
from finch.autoschedule.loop_order_cost import loop_order_cost
from finch.autoschedule.loop_order_greedy import greedy_loop_order
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.finch_logic import Alias, Field, Literal, MapJoin, Table


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

    greedy_cost = loop_order_cost(expr, greedy, stats_factory, stats)
    bfs_cost = loop_order_cost(expr, bfs, stats_factory, stats)
    dfs_cost = loop_order_cost(expr, dfs, stats_factory, stats)

    assert set(bfs) == set(expr.fields())
    assert set(dfs) == set(expr.fields())
    assert bfs_cost <= greedy_cost
    assert dfs_cost <= greedy_cost
    assert bfs_cost == pytest.approx(dfs_cost)

