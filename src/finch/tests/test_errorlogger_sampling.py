from unittest.mock import Mock

import pytest

import numpy as np

import finch as ft
from finch.algebra import StaticFillFType, ffuncs
from finch.autoschedule.tensor_stats import sampling_stats
from finch.autoschedule.tensor_stats.dense_stats import DenseStatsFactory
from finch.autoschedule.tensor_stats.errorlogger_stats import ErrorLoggerStatsFactory
from finch.autoschedule.tensor_stats.sampling_stats import (
    SamplingStats,
    SamplingStatsFactory,
)
from finch.finch_logic import Field, Literal, Table


@pytest.fixture
def sketch_spy(monkeypatch):
    spy = Mock(wraps=sampling_stats.compute_sketch)
    monkeypatch.setattr(sampling_stats, "compute_sketch", spy)
    return spy


def sampling_estimate(stats, factory) -> SamplingStats:
    match stats.estimates[factory]:
        case SamplingStats() as estimate:
            return estimate
        case _:
            pytest.fail("Expected sampling statistics")


@pytest.mark.parametrize("sample_budgets", [(10, 10), (10, 20)])
def test_errorlogger_sampling_factories_share_seeds(sample_budgets, sketch_spy):
    first = SamplingStatsFactory(sample_budgets[0], estimator="uj1")
    second = SamplingStatsFactory(sample_budgets[1], estimator="sh3")
    factories = (first, DenseStatsFactory(), second)
    logger = ErrorLoggerStatsFactory(*factories)
    data = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.intp)

    stats = logger(ft.asarray(data), (Field("i"), Field("j")))

    assert logger.factories == factories
    assert tuple(stats.estimates) == factories
    assert sketch_spy.call_count == 2
    # Equal budgets share seeds, so both estimators see the same sketch.
    assert (first._seeds is second._seeds) == (sample_budgets[0] == sample_budgets[1])
    first_stats = sampling_estimate(stats, first)
    second_stats = sampling_estimate(stats, second)
    assert first_stats is not second_stats
    assert first_stats.estimator == "uj1"
    assert second_stats.estimator == "sh3"
    assert not logger.log.samples
    assert stats.estimate_non_fill_values() == 3
    assert tuple(logger.log.samples) == factories
    assert logger.log.samples[first] == (3, 3)
    assert logger.log.samples[factories[1]] == (3, 6)
    assert logger.log.samples[second] == (3, 3)


@pytest.mark.parametrize(
    "operation", ["copy", "support", "mapjoin", "aggregate", "relabel", "reorder"]
)
def test_errorlogger_sampling_transformations(operation, sketch_spy):
    first = SamplingStatsFactory(10, estimator="uj1")
    second = SamplingStatsFactory(10, estimator="sh3")
    logger = ErrorLoggerStatsFactory(first, second)
    i, j = Field("i"), Field("j")
    data = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.intp)
    source = logger(ft.asarray(data), (i, j))
    assert not logger.log.samples
    sketch_spy.reset_mock()
    expected = data
    fields = (i, j)

    match operation:
        case "copy":
            stats = logger.copy(source)
        case "support":
            stats = logger.support(source)
            assert stats.fill_type == StaticFillFType(False)
        case "mapjoin":
            stats = logger.mapjoin(ffuncs.mul, source, source)
        case "aggregate":
            stats = logger.aggregate(ffuncs.add, 0, (j,), source)
            expected = data.sum(axis=1)
            fields = (i,)
        case "relabel":
            fields = (Field("row"), Field("col"))
            stats = logger.relabel(source, fields)
        case "reorder":
            fields = (j, i)
            stats = logger.reorder(source, fields)
            expected = data.T

    assert sketch_spy.call_count == (0 if operation in ("copy", "support") else 2)
    assert tuple(stats.estimates) == (first, second)
    first_stats = sampling_estimate(stats, first)
    second_stats = sampling_estimate(stats, second)
    assert first_stats is not second_stats
    assert first_stats.estimator == "uj1"
    assert second_stats.estimator == "sh3"
    assert stats.index_order == fields
    for estimate in (first_stats, second_stats):
        match estimate.sketch:
            case Table(Literal(tensor), indices):
                assert indices == fields
                np.testing.assert_array_equal(np.asarray(tensor), expected)
            case _:
                pytest.fail("Sampling statistics must contain a materialized sketch")
    count = float(np.count_nonzero(expected))
    assert not logger.log.samples
    assert stats.estimate_non_fill_values(exact_count=count) == count
    assert tuple(logger.log.samples) == (first, second)
    assert list(logger.log.samples.values()) == [(count, count), (count, count)]
    assert source.index_order == (i, j)


def test_errorlogger_sampling_records_latest_counts(sketch_spy):
    first = SamplingStatsFactory(10, estimator="uj1")
    second = SamplingStatsFactory(10, estimator="sh3")
    logger = ErrorLoggerStatsFactory(first, second)
    fields = (Field("i"), Field("j"))
    dense = logger(ft.asarray(np.ones((2, 3), dtype=np.intp)), fields)
    sparse = logger(ft.asarray(np.eye(2, 3, dtype=np.intp)), fields)

    assert sketch_spy.call_count == 4
    assert not logger.log.samples
    assert dense.estimate_non_fill_values(exact_count=6) == 6
    assert list(logger.log.samples.values()) == [(6, 6), (6, 6)]
    assert sparse.estimate_non_fill_values(exact_count=2) == 2
    assert list(logger.log.samples.values()) == [(2, 2), (2, 2)]


def test_errorlogger_sampling_subclasses(sketch_spy):
    class CustomSamplingStatsFactory(SamplingStatsFactory):
        pass

    first = SamplingStatsFactory(10, estimator="uj1")
    second = CustomSamplingStatsFactory(10, estimator="sh3")
    logger = ErrorLoggerStatsFactory(first, second)

    stats = logger(ft.asarray(np.ones((2, 3), dtype=np.intp)), (Field("i"), Field("j")))

    assert sketch_spy.call_count == 2
    assert sampling_estimate(stats, first).estimator == "uj1"
    assert sampling_estimate(stats, second).estimator == "sh3"
    assert not logger.log.samples


def test_errorlogger_without_sampling_factories(sketch_spy):
    factories = (DenseStatsFactory(), DenseStatsFactory())
    logger = ErrorLoggerStatsFactory(*factories)
    stats = logger(ft.asarray(np.ones((2, 3), dtype=np.intp)), (Field("i"), Field("j")))

    assert sketch_spy.call_count == 0
    assert tuple(stats.estimates) == factories
    assert stats.estimates[factories[0]] is not stats.estimates[factories[1]]
    assert not logger.log.samples
    assert stats.estimate_non_fill_values(exact_count=6) == 6
    assert tuple(logger.log.samples) == factories
    assert list(logger.log.samples.values()) == [(6, 6), (6, 6)]
