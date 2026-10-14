import pytest

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.autoschedule.tensor_stats.sampling_stats import (
    SAMPLING_ESTIMATORS,
    SamplingStatsFactory,
)
from finch.tests.stats_cases import (
    DATASETS,
    KERNELS,
    SAMPLE_NNZ,
    SEED,
    make_banded,
    make_diagonal,
    make_kernel_estimator,
    make_kernel_stats,
    make_tridiagonal,
)


@pytest.mark.parametrize("n", [0, 1, 7])
@pytest.mark.parametrize(
    "make_matrix,width", [(make_diagonal, 0), (make_tridiagonal, 1), (make_banded, 5)]
)
def test_sparse_synthetic_patterns(n, make_matrix, width):
    matrix = make_matrix(n)
    rows, cols = np.indices((n, n))
    assert matrix.format == "csr"
    np.testing.assert_array_equal(matrix.toarray(), abs(rows - cols) <= width)


@pytest.mark.parametrize(
    "dataset",
    ["Diagonal", "Tridiagonal", "Banded", "Triangular", "Striped", "Uniform Random"],
)
def test_synthetic_inputs_exceed_sampling_budget(dataset):
    matrix = sps.csr_array(DATASETS[dataset]())
    assert matrix.count_nonzero() > SAMPLE_NNZ


def test_sampling_estimators_can_share_kernel_sketches():
    tensor = ft.asarray((np.random.default_rng(0).random((8, 8)) < 0.4).astype(float))

    def make_factory(estimator):
        factory = SamplingStatsFactory(sample_nnz=8, estimator=estimator)
        factory._rng = np.random.default_rng(SEED)
        return factory

    factory = make_factory("uj1")
    kernel_stats = make_kernel_stats(factory, tensor)
    shared = {kernel: kernel_stats(kernel) for kernel in KERNELS}
    for stats in shared.values():
        stats.scan(needs_freq=True)
    for estimator in SAMPLING_ESTIMATORS:
        independent = make_kernel_estimator(make_factory(estimator), tensor)
        for kernel, cached_stats in shared.items():
            stats = factory.copy(cached_stats)
            stats.estimator = estimator
            assert stats.estimate_non_fill_values() == pytest.approx(
                independent(kernel)
            )
