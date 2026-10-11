import pytest

import numpy as np

import finch as ft
from finch.algebra import DynamicFillFType, StaticFillFType, ffuncs, ftype
from finch.autoschedule.tensor_stats import (
    DCStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
)
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.logic_to_stats import insert_statistics
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import (
    StatsInterpreter,
    calculate_estimated_error,
)
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    FillTable,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)


@pytest.mark.parametrize("use_interpreter", [False, True])
@pytest.mark.parametrize(
    "factory_cls,expected_nnz",
    [
        (DCStatsFactory, 1),  # Degree bounds conservatively count any scalar as one.
        (UniformStatsFactory, 0),
        (ExactStatsFactory, 0),
        (SamplingStatsFactory, 0),
    ],
)
@pytest.mark.parametrize(
    "fill_type",
    [
        StaticFillFType(np.float64(3)),
        DynamicFillFType(ftype(np.float64)),
        StaticFillFType((np.float64(0), np.intp(0))),
        DynamicFillFType(ftype((np.float64(0), np.intp(0)))),
    ],
)
def test_fill_table_statistics(use_interpreter, factory_cls, expected_nnz, fill_type):
    factory = factory_cls()
    fields = (Field("i"), Field("j"))
    alias = HardAlias("a")
    source = factory(ft.asarray(np.ones((2, 3))), fields)
    source.fill_type = fill_type
    bindings: dict[Alias, NumericStats] = {alias: source}
    node = FillTable(alias)
    if use_interpreter:
        result = StatsInterpreter(factory)(node, bindings)
    else:
        cache = {}
        result = insert_statistics(factory, node, bindings, False, cache)
        assert cache[node] is result

    assert isinstance(result, NumericStats)
    assert result.index_order == ()
    assert result.dim_sizes == {}
    assert result.fill_type == fill_type
    assert result.estimate_non_fill_values() == expected_nnz
    assert source.index_order == fields
    assert source.fill_type == fill_type
    assert source.estimate_non_fill_values() == 6


@pytest.mark.parametrize("use_interpreter", [False, True])
def test_fill_table_statistics_requires_binding(use_interpreter):
    factory = DCStatsFactory()
    node = FillTable(HardAlias("missing"))
    with pytest.raises(ValueError, match="alias"):
        if use_interpreter:
            StatsInterpreter(factory)(node, {})
        else:
            insert_statistics(factory, node, {}, False, {})


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 2), (2, 2)),
        ((2, 3), (3, 4)),
    ],
)
def test_stats_matrix_multiplication(shape_a, shape_b):
    a = ft.asarray(np.ones(shape_a))
    b = ft.asarray(np.ones(shape_b))

    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
        (
            Query(Table(HardAlias("A"), (i, k)), Table(Literal(a), (i, k))),
            Query(Table(HardAlias("B"), (k, j)), Table(Literal(b), (k, j))),
            Query(
                Table(HardAlias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    interpreter = StatsInterpreter(stats_factory=DCStatsFactory())
    result = interpreter(p, {})
    assert isinstance(result, tuple)
    result_stats = result[0]

    expected_rows = shape_a[0]
    expected_cols = shape_b[1]

    assert result_stats.dim_sizes[i] == expected_rows
    assert result_stats.dim_sizes[j] == expected_cols
    assert tuple(f.name for f in result_stats.index_order) == ("i", "j")


def test_stats_matmul_error():
    a_val = ft.asarray(np.ones((20, 30)))
    b_val = ft.asarray(np.ones((30, 20)))

    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
        (
            Query(Table(HardAlias("A"), (i, k)), Table(Literal(a_val), (i, k))),
            Query(Table(HardAlias("B"), (k, j)), Table(Literal(b_val), (k, j))),
            Query(
                Table(HardAlias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    errors = calculate_estimated_error(
        node=p,
        stats_factory=DCStatsFactory(),
        logic_bindings={},
        stats_bindings={},
    )

    assert errors[0] == 0.0


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 2), (2, 2)),
        ((2, 3), (3, 4)),
    ],
)
def test_vp_stats_matrix_multiplication(shape_a, shape_b):
    a = ft.asarray(np.ones(shape_a))
    b = ft.asarray(np.ones(shape_b))

    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
        (
            Query(Table(HardAlias("A"), (i, k)), Table(Literal(a), (i, k))),
            Query(Table(HardAlias("B"), (k, j)), Table(Literal(b), (k, j))),
            Query(
                Table(HardAlias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    interpreter = StatsInterpreter(stats_factory=VPStatsFactory())
    result = interpreter(p, {})
    assert isinstance(result, tuple)
    result_stats = result[0]

    expected_rows = shape_a[0]
    expected_cols = shape_b[1]

    assert result_stats.dim_sizes[i] == expected_rows
    assert result_stats.dim_sizes[j] == expected_cols
    assert tuple(f.name for f in result_stats.index_order) == ("i", "j")


def test_vp_stats_matmul_error():
    a_val = ft.asarray(np.ones((20, 30)))
    b_val = ft.asarray(np.ones((30, 20)))

    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
        (
            Query(Table(HardAlias("A"), (i, k)), Table(Literal(a_val), (i, k))),
            Query(Table(HardAlias("B"), (k, j)), Table(Literal(b_val), (k, j))),
            Query(
                Table(HardAlias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    errors = calculate_estimated_error(
        node=p,
        stats_factory=VPStatsFactory(),
        logic_bindings={},
        stats_bindings={},
    )

    assert errors[0] == 0.0
