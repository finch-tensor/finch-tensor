import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch.autoschedule.cost_models import ElementLevelOption
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_BUILDERS,
    IterCostFormatter,
    StorageCostFormatter,
)
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import Alias, Field, Plan, Produces, Query, Table


def get_stats(matrix, fields):
    factory = DCStatsFactory()
    M = Alias("M")
    query = Query(Table(Alias("A"), fields), Table(M, fields))
    machine = StatsMachine(
        factory, OrderedDict({M: factory(fl.asarray(matrix), fields)})
    )
    stats = machine(query)
    return stats, query, machine


def format_cost(cost_model, query, options, machine):
    prgm = Plan((query, Produces((query.lhs.tns,))))
    M = query.rhs.tns
    return cost_model.predict_cost(
        prgm,
        {query.lhs.tns: list(options)},
        {M: machine.bindings[M]},
        machine.stats_factory,
    )


def brute_force_best(formatter, query, shape_type, machine, fill_type):
    best_cost = None
    best_options = ()
    for options in itertools.product(formatter.options, repeat=len(shape_type)):
        cost = format_cost(
            formatter.cost_model, query, (*options, ElementLevelOption()), machine
        )
        if best_cost is None or cost < best_cost:
            best_cost, best_options = cost, options
    lvl = fl.element(fill_type, fill_type.element_type)
    for option, dim_type in zip(
        reversed(best_options), reversed(shape_type), strict=True
    ):
        lvl = LEVEL_BUILDERS[option.kind](lvl, dim_type)
    return lvl


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
        np.array(3.0),
        np.array([0.0, 1.0, 0.0, 2.0]),
        np.arange(24, dtype=np.float64).reshape(2, 3, 4) % 5,
    ]
)
def matrix(request):
    return request.param


@pytest.mark.parametrize("formatter_cls", [StorageCostFormatter, IterCostFormatter])
def test_search_and_brute(formatter_cls, matrix):
    fields = tuple(Field(f"i{n}") for n in range(matrix.ndim))
    stats, query, machine = get_stats(matrix, fields)
    shape_type = tuple(fl.ftype(np.intp) for _ in fields)
    formatter = formatter_cls()

    brute_fmt = brute_force_best(formatter, query, shape_type, machine, stats.fill_type)

    search_fmt = formatter.get_tensor_ftype(
        query.lhs.tns, query, machine, stats.fill_type, shape_type
    ).lvl_t
    assert search_fmt == brute_fmt
