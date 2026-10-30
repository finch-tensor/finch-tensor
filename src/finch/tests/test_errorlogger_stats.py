import pytest

import numpy as np

import finch as ft
from finch.algebra import StaticFillFType, ffuncs
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.tensor_stats.errorlogger_stats import ErrorLoggerStatsFactory
from finch.finch_logic import Aggregate, Field, MapJoin


@pytest.fixture
def errorlogger():
    factory = ErrorLoggerStatsFactory(DenseStatsFactory())
    data = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    stats = factory(ft.asarray(data), (Field("i"), Field("j")))
    return factory, stats


def test_errorlogger_preserves_tensor_metadata(errorlogger):
    _, stats = errorlogger
    i, j = stats.index_order

    assert stats.idxs == (i, j)
    assert stats.dim_sizes == {i: 2.0, j: 3.0}
    assert stats.fill_type == stats.estimate.fill_type == stats.exact.fill_type
    assert stats.shape_type == stats.estimate.shape_type == stats.exact.shape_type
    assert stats.element_type == ft.float64
    np.testing.assert_allclose(stats.get_embedding(), np.log2([2, 3]))


@pytest.mark.parametrize(
    "fixed_axes, expected", [((), 6), ((0,), 3), ((1,), 2), ((0, 1), 1)]
)
def test_errorlogger_estimates_slices(errorlogger, fixed_axes, expected):
    _, stats = errorlogger
    fields = (stats.index_order[axis] for axis in fixed_axes)

    assert stats.estimate_non_fill_values(over=fields) == expected


def test_errorlogger_records_join_and_reduction(errorlogger):
    factory, stats = errorlogger
    i, j = stats.index_order

    joined = factory.mapjoin(ffuncs.mul, stats, stats)
    reduced = factory.aggregate(ffuncs.add, 0, (j,), joined)

    assert joined.index_order == (i, j)
    assert reduced.index_order == (i,)
    assert reduced.estimate_non_fill_values() == 2
    assert factory.error_log.samples == {
        f"{MapJoin.__name__}({ffuncs.mul})": [(3.0, 6.0)],
        f"{Aggregate.__name__}({ffuncs.add})": [(2.0, 2.0)],
    }


def test_errorlogger_copy_preserves_independent_stats(errorlogger):
    factory, stats = errorlogger
    copied = factory.copy(stats)
    i, _ = stats.index_order

    assert copied is not stats
    assert copied.exact is not stats.exact
    assert copied.estimate is not stats.estimate
    assert copied.fill_type == stats.fill_type
    copied.dim_sizes[i] = 5
    copied.estimate.dim_sizes[i] = 7
    copied.exact.dim_sizes[i] = 9
    for original in (stats, stats.estimate, stats.exact):
        assert original.dim_sizes[i] == 2
    assert factory.error_log.samples == {}


@pytest.mark.parametrize("operation", ["relabel", "reorder"])
def test_errorlogger_transforms_both_stats(errorlogger, operation):
    factory, stats = errorlogger
    i, j = stats.index_order
    match operation:
        case "relabel":
            fields = (Field("row"), Field("column"))
            transformed = factory.relabel(stats, fields)
            sizes = (2, 3)
        case "reorder":
            fields = (j, i)
            transformed = factory.reorder(stats, fields)
            sizes = (3, 2)

    for result in (transformed, transformed.exact, transformed.estimate):
        assert result.index_order == fields
        assert tuple(result.dim_sizes[field] for field in fields) == sizes
        assert result.fill_type == stats.fill_type
    assert transformed.exact.estimate_non_fill_values() == 3
    assert transformed.estimate_non_fill_values() == 6
    assert factory.error_log.samples == {}


def test_errorlogger_support_updates_both_stats():
    factory = ErrorLoggerStatsFactory(DenseStatsFactory())
    i, j = Field("i"), Field("j")
    tensor = ft.BufferizedNDArray.from_numpy(
        np.array([[5.0, 1.0, 5.0], [5.0, 5.0, 5.0]]), fill_value=np.float64(5.0)
    )
    stats = factory(tensor, (i, j))

    support = factory.support(stats)
    for result in (support, support.exact, support.estimate):
        assert result.fill_type == StaticFillFType(False)
    assert stats.fill_type == StaticFillFType(np.float64(5.0))

    projected = factory.aggregate(ffuncs.or_, False, (j,), support)
    assert projected.exact.estimate_non_fill_values() == 1
    assert projected.estimate_non_fill_values() == 2
