import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch.autoschedule.galley.logical_optimizer import insert_statistics
from finch.autoschedule.smart_formatter import (
    IterCostFormatter,
    StorageCostFormatter,
    total_tree_cost,
)
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.finch_logic import Field, Literal, Table


def get_stats(matrix, fields):
    factory = DCStatsFactory()
    node = Table(Literal(fl.asarray(matrix)), fields)
    stats = insert_statistics(
        stats_factory=factory,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )

    return stats, factory


def brute_force_best(
    fields, shape_type, stats, factory, fill_value, candidates, leaf_cost_fn
):
    fill_ftype = fl.ftype(fill_value)
    leaf_lvl = fl.element(fill_value, fill_ftype)
    best_cost = None
    best_fmt = None
    for combination in itertools.product(candidates, repeat=len(fields)):
        lvl = leaf_lvl
        for option, dim_type in zip(
            reversed(combination), reversed(shape_type), strict=True
        ):
            lvl = option.build(lvl, dim_type)
        cost = total_tree_cost(
            lvl, fields, stats, factory, 1.0, 0, candidates, leaf_cost_fn
        )
        if best_cost is None or cost < best_cost:
            best_cost, best_fmt = cost, lvl
    return best_cost, best_fmt


@pytest.fixture(
    params=[
        np.array(
            [
                [0, 3, 0, 0],
                [0, 0, 0, 0],
                [5, 0, 2, 0],
                [0, 0, 0, 7],
            ],
            dtype=np.float64,
        ),
        np.ones((4, 4), dtype=np.float64),
        np.zeros((6, 5), dtype=np.float64),
        np.eye(5, dtype=np.float64),
    ]
)
def matrix(request):
    return request.param


@pytest.fixture
def fields_2d():
    return (Field("i"), Field("j"))


def test_storage_dp_and_brute(matrix, fields_2d):
    stats, factory = get_stats(matrix, fields_2d)
    shape_type = tuple(fl.ftype(np.intp) for _ in fields_2d)
    formatter = StorageCostFormatter()
    formatter._stats_factory = factory

    brute_cost, brute_fmt = brute_force_best(
        fields_2d,
        shape_type,
        stats,
        factory,
        stats.fill_value,
        formatter.candidates,
        formatter.leaf_cost_fn,
    )

    dp_fmt = formatter.get_tensor_ftype(stats.fill_value, shape_type, stats).lvl_t
    dp_cost = total_tree_cost(
        dp_fmt,
        fields_2d,
        stats,
        factory,
        1,
        0,
        formatter.candidates,
        formatter.leaf_cost_fn,
    )
    # print(f"Matrix : {matrix}, \nFormat by brute : {
    # brute_fmt}, \nFormat by dp : {dp_fmt}")
    assert dp_cost == pytest.approx(brute_cost)
    assert dp_fmt == brute_fmt


def test_iter_dp_and_brute(matrix, fields_2d):
    stats, factory = get_stats(matrix, fields_2d)
    shape_type = tuple(fl.ftype(np.intp) for _ in fields_2d)
    formatter = IterCostFormatter()
    formatter._stats_factory = factory

    brute_cost, brute_fmt = brute_force_best(
        fields_2d,
        shape_type,
        stats,
        factory,
        stats.fill_value,
        formatter.candidates,
        formatter.leaf_cost_fn,
    )

    dp_fmt = formatter.get_tensor_ftype(stats.fill_value, shape_type, stats).lvl_t
    dp_cost = total_tree_cost(
        dp_fmt,
        fields_2d,
        stats,
        factory,
        1,
        0,
        formatter.candidates,
        formatter.leaf_cost_fn,
    )

    # print(f"Matrix : {matrix}, \nFormat by brute : {
    # brute_fmt}, \nFormat by dp : {dp_fmt}")
    assert dp_cost == pytest.approx(brute_cost)
    assert dp_fmt == brute_fmt
