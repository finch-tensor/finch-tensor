"""Compare full-size estimator accuracy against one array reference per estimator.

Regenerate with::

    pixi run -e test-julia pytest src/finch/tests/test_accuracy.py --regen-all

Bundled Matrix Market inputs retain their full dimensions. Input values are
normalized to their nonzero pattern to avoid signed cancellation. The uniform
random matrix and sampling factories are seeded. Array comparisons allow
floating-point roundoff across platforms while keeping counts and metadata exact.
Synthetic inputs exceed the sampling budget. Sampling estimators share sketches.
"""

from __future__ import annotations

import logging
import math

import pytest

import numpy as np
import scipy.sparse as sps
from numpy.typing import ArrayLike

import finch as ft
from finch.autoschedule.tensor_stats.sampling_stats import (
    SamplingStats,
    SamplingStatsFactory,
)
from finch.compile_jl.julia import julia_available
from finch.tests.stats_cases import (
    BLOCK_COUNT,
    DATASETS,
    DENSE_MATRIX_SIZE,
    KERNELS,
    RANDOM_DENSITY,
    RANDOM_MATRIX_SIZE,
    SAMPLE_NNZ,
    SEED,
    SPARSE_MATRIX_SIZE,
    make_kernel_estimator,
    make_kernel_stats,
    make_models,
)

GEOMEAN_DATASETS = (
    "ct20stif",
    "bcsstk39",
    "ca-GrQc",
    "ca-HepTh",
    "web-NotreDame",
    "roadNet-PA",
    "soc-sign-epinions",
)


def actual_nonzeros(matrix, batch_rows=128):
    # Boolean products count reachability without large path counts. Row batches
    # bound intermediate storage even when powers of a sparse graph become dense.
    pattern = matrix.astype(bool)
    actual = dict.fromkeys(KERNELS, 0)
    actual["Hadamard"] = int(pattern.count_nonzero())
    for start in range(0, pattern.shape[0], batch_rows):
        rows = pattern[start : start + batch_rows]
        squared = rows @ pattern
        actual["SpGEMM"] += int(squared.count_nonzero())
        actual["SpGEMM2"] += int((squared @ pattern).count_nonzero())
        actual["Triangle Counting"] += int(squared.multiply(rows).count_nonzero())
    return actual


@pytest.fixture(scope="module")
def accuracy_results():
    results: dict[str, dict[str, ArrayLike]] = {
        model: {
            "kernels": KERNELS,
            "dense_matrix_size": DENSE_MATRIX_SIZE,
            "sparse_matrix_size": SPARSE_MATRIX_SIZE,
            "random_matrix_size": RANDOM_MATRIX_SIZE,
            "random_density": RANDOM_DENSITY,
            "block_count": BLOCK_COUNT,
            "seed": SEED,
            "sample_nnz": SAMPLE_NNZ,
            "geomean_datasets": GEOMEAN_DATASETS,
        }
        for model in make_models()
    }
    q_errors = {model: [] for model in results}
    for dataset, make_matrix in DATASETS.items():
        a = sps.csr_array(make_matrix())
        tensor = ft.asarray(a)
        logging.getLogger(__name__).info("Computing reference counts for %s", dataset)
        actual = actual_nonzeros(a)
        sampling_stats: dict[str, SamplingStats] = {}
        # Fresh factories isolate sampling masks between datasets.
        for model, factory in make_models().items():
            logging.getLogger(__name__).info("Estimating %s with %s", dataset, model)
            match factory:
                case SamplingStatsFactory():
                    # Sampling variants use identical budgets and seeds; only
                    # the estimator applied to the materialized sketch differs.
                    if not sampling_stats:
                        kernel_stats = make_kernel_stats(factory, tensor)
                        sampling_stats = {
                            kernel: kernel_stats(kernel) for kernel in KERNELS
                        }
                        for kernel, stats in sampling_stats.items():
                            _, nnz, _, _ = stats.scan(needs_freq=True)
                            assert nnz <= SAMPLE_NNZ
                            assert (
                                math.prod(stats.sample_probs) * stats.remainder_prob < 1
                            )
                            logging.getLogger(__name__).info(
                                "%s / %s: %s retained nonzeros", dataset, kernel, nnz
                            )
                    estimates = {}
                    for kernel, cached_stats in sampling_stats.items():
                        stats = factory.copy(cached_stats)
                        stats.estimator = factory.estimator
                        estimates[kernel] = stats.estimate_non_fill_values()
                case _:
                    estimate_kernel = make_kernel_estimator(factory, tensor)
                    estimates = {kernel: estimate_kernel(kernel) for kernel in KERNELS}
            actual_counts = np.array([actual[kernel] for kernel in KERNELS])
            estimated_counts = np.array(
                [estimates[kernel] for kernel in KERNELS], dtype=np.float64
            )
            assert np.all(np.isfinite(estimated_counts) & (estimated_counts >= 0)), (
                dataset,
                model,
                estimated_counts,
            )
            # Clamp only for ratios, retaining the raw counts in the reference.
            ratios = np.maximum(estimated_counts, 1) / np.maximum(actual_counts, 1)
            errors = np.maximum(ratios, 1 / ratios)
            if dataset in GEOMEAN_DATASETS:
                q_errors[model].extend(errors)
            results[model].update(
                {
                    f"{dataset}/shape": a.shape,
                    f"{dataset}/nnz": int(a.nnz),
                    f"{dataset}/actual_nnz": actual_counts,
                    f"{dataset}/estimated_nnz": estimated_counts,
                    f"{dataset}/ratio": ratios,
                    f"{dataset}/q_error": errors,
                }
            )
    for model, errors in q_errors.items():
        results[model]["geomean_q_error"] = float(np.exp(np.mean(np.log(errors))))
    return results


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(), reason="accuracy regression requires the Julia backend"
)
@pytest.mark.parametrize("model", make_models())
def test_statistics_accuracy(model, accuracy_results, ndarrays_regression):
    ndarrays_regression.check(
        accuracy_results[model], default_tolerance={"rtol": 1e-11, "atol": 1e-6}
    )
