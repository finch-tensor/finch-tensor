import math

import pytest

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.autoschedule.tensor_stats import BaseTensorStats
from finch.autoschedule.tensor_stats.sampling_stats import (
    SAMPLING_ESTIMATORS,
    SamplingStats,
    SamplingStatsFactory,
    _dgood1,
    _dsh,
    _dsh2,
    _dsh3,
    _dsilly,
    _dsj1,
    _duj1,
    _duj2,
)
from finch.compile_jl.julia import julia_available
from finch.finch_logic import Field, Literal, Table
from finch.tests.stats_cases import SAMPLE_NNZ, make_models


def estimate(name, d, f1, frequencies, q, n, N):
    match name:
        case "uj1":
            return _duj1(d, f1, q, n)
        case "sj1":
            return _dsj1(d, q, N)
        case "uj2":
            return _duj2(d, f1, frequencies, q, n, N)
        case "schlosser":
            return _dsh(d, f1, frequencies, q, n)
        case "sh2":
            return _dsh2(d, f1, frequencies, q, n, N)
        case "sh3":
            return _dsh3(d, f1, frequencies, q, n)
        case "good1":
            return _dgood1(d, frequencies, n, N)
        case "silly":
            return _dsilly(d, q)
        case _:
            raise ValueError(name)


def test_goodman_accumulates_alternating_corrections():
    # Goodman: d + (N-n)/n*f1 - (N-n)(N-n+1)/(n(n-1))*f2.
    assert _dgood1(6, {1: 5, 2: 1}, 7, 14) == pytest.approx(29 / 3)


@pytest.mark.parametrize("name", SAMPLING_ESTIMATORS)
def test_sampling_estimators_census_and_empty(name):
    assert estimate(name, 17, 10, {1: 10, 2: 5, 3: 2}, 1.0, 26, 26) == 17
    assert estimate(name, 0, 0, None, 0.0, 0, 100) == 0


@pytest.mark.parametrize("name", SAMPLING_ESTIMATORS)
@pytest.mark.parametrize("q", [0.25, 1e-20])
def test_sampling_estimators_singletons(name, q):
    assert estimate(name, 10, 10, {1: 10}, q, 10, 10 / q) == pytest.approx(10 / q)


def test_smoothed_jackknife_solves_occupancy_equation():
    d = 100 * (1 - 0.9**10)
    assert _dsj1(d, 0.1, 1000) == pytest.approx(100)


@pytest.mark.parametrize("name", ["schlosser", "sh2", "sh3"])
def test_schlosser_large_frequencies_do_not_overflow(name):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = estimate(name, 2, 1, {1: 1, 10000: 1}, 0.5, 10001, 20002)
    assert math.isfinite(result)
    assert result >= 2


@pytest.mark.parametrize(
    "name,expected",
    [
        ("uj1", 23.89189189189189),
        ("uj2", 23.980150818429326),
        ("sj1", 23.717990245990222),
        ("schlosser", 38.377245508982035),
        ("sh2", 23.880115843368404),
        ("sh3", 33.52105882450244),
    ],
)
def test_sampling_estimators_reference_formulas(name, expected):
    # Haas and Stokes: https://people.cs.umass.edu/~phaas/files/jasa3rj.pdf
    assert estimate(name, 17, 10, {1: 10, 2: 5, 3: 2}, 0.25, 26, 104) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("name", SAMPLING_ESTIMATORS)
def test_sampling_output_and_reduction_probabilities_are_separate(name):
    i, j = Field("i"), Field("j")
    factory = SamplingStatsFactory(estimator=name)
    stats = factory(ft.asarray(np.ones((100, 4))), (i, j), [0.5, 0.25])
    reduced = factory.aggregate(ft.ffuncs.add, 0, (j,), stats)
    reduced.scan_cache = (8.0, 4.0, 2.0, {1: 2, 3: 2})
    conditional = (
        4.0 if name == "silly" else estimate(name, 4, 2, {1: 2, 3: 2}, 0.25, 8, 32)
    )
    assert reduced.estimate_non_fill_values() == pytest.approx(
        min(100, conditional / 0.5)
    )


@pytest.mark.parametrize("name", SAMPLING_ESTIMATORS)
def test_sampling_output_only_sampling_scales_repeated_counts(name):
    i, j = Field("i"), Field("j")
    factory = SamplingStatsFactory(estimator=name)
    stats = factory(ft.asarray(np.ones((100, 4))), (i, j), [0.25, 1.0])
    reduced = factory.aggregate(ft.ffuncs.add, 0, (j,), stats)
    reduced.scan_cache = (40.0, 10.0, 0.0, None)
    assert reduced.estimate_non_fill_values() == 40.0
    reduced.scan_cache = (120.0, 30.0, 0.0, None)
    assert reduced.estimate_non_fill_values() == 100.0


def test_accuracy_includes_all_sampling_estimators():
    models = make_models()
    sampling = {
        name: model for name, model in models.items() if name.startswith("Sampling_")
    }
    assert set(sampling) == {f"Sampling_{name}" for name in SAMPLING_ESTIMATORS}
    assert SAMPLE_NNZ == SamplingStatsFactory().sample_nnz == 10_000
    seeds = []
    for model in sampling.values():
        assert model.sample_nnz == 10_000
        seeds.append(model._rng.integers(0, 1 << 64, dtype=np.uint64))
    assert len(set(seeds)) == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_sampling_histogram_reads_stored_counts(monkeypatch, sparse):
    from finch.autoschedule import default_schedulers

    if sparse and not julia_available():
        pytest.skip("sparse histogram scan requires the Julia backend")
    data = np.array([[0, 1, 1, 10000]], dtype=np.intp)
    tensor = ft.asarray(sps.csr_array(data) if sparse else data)
    fields = (Field("i"), Field("j"))
    stats = SamplingStatsFactory()(tensor, fields)
    stats.sketch = Table(Literal(tensor), fields)
    scheduler = default_schedulers.NON_RECURSIVE_SCHEDULER
    calls = []

    def counted_scheduler(plan):
        calls.append(plan)
        return scheduler(plan)

    monkeypatch.setattr(
        default_schedulers, "NON_RECURSIVE_SCHEDULER", counted_scheduler
    )
    assert stats.scan(needs_freq=True) == (10002, 3, 2, {1: 2, 10000: 1})
    assert not calls


def test_sampling_scan_keeps_large_sketch_sparse():
    data = sps.coo_array(
        ([1, 1, 1], ([0, 500_000, 999_999], [3, 500_001, 999_998])),
        shape=(1_000_000, 1_000_000),
    )
    tensor = ft.asarray(data)
    fields = (Field("i"), Field("j"))
    stats = SamplingStats(
        BaseTensorStats(fields, dict(zip(fields, tensor.shape, strict=True)), 0),
        sketch=Table(Literal(tensor), fields),
        sample_probs=[1.0, 1.0],
    )
    assert stats.scan(needs_freq=True) == (3, 3, 3, {1: 3})
