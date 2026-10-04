"""Shared full-size datasets and estimator cases for regression and benchmarks."""

from functools import cache, partial
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.io import mmread

from finch.algebra import ffuncs
from finch.autoschedule.tensor_stats import (
    BlockedUniformStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
)
from finch.finch_logic import Field

i, j, k, ell = (Field(name) for name in "ijkl")
N = 100
RANDOM_MATRIX_SIZE = 10_000
RANDOM_DENSITY = 0.001
BLOCK_COUNT = 5
SEED = 42
KERNELS = ("Hadamard", "SpGEMM", "SpGEMM2", "Triangle Counting")


def make_models():
    sampling = SamplingStatsFactory(sample_prob=0.5)
    sampling._rng = np.random.default_rng(SEED)
    return {
        "Dense": DenseStatsFactory(),
        "Uniform": UniformStatsFactory(),
        "DC": DCStatsFactory(),
        "LP": LPStatsFactory(),
        "Sampling_0.5": sampling,
        "Blocked-Uniform": BlockedUniformStatsFactory(block_count=BLOCK_COUNT),
    }


def make_diagonal(n):
    return np.eye(n, dtype=np.float64)


def make_tridiagonal(n):
    A = np.eye(n, k=0) + np.eye(n, k=1) + np.eye(n, k=-1)
    return (A > 0).astype(np.float64)


def make_banded(n, bw=5):
    r, c = np.indices((n, n))
    return (np.abs(r - c) <= bw).astype(np.float64)


def make_triangular(n):
    return np.triu(np.ones((n, n), dtype=np.float64))


def make_striped(n):
    A = np.zeros((n, n), dtype=np.float64)
    A[:, ::5] = 1
    return A


def load_matrix_market(filename):
    path = Path(__file__).parent / "data" / filename
    matrix = sps.csr_array(mmread(path), dtype=np.float64)
    # Match the synthetic inputs: measure sparsity, without signed cancellation.
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data.fill(1.0)
    return matrix


def make_uniform_random():
    return sps.csr_array(
        sps.random(
            RANDOM_MATRIX_SIZE,
            RANDOM_MATRIX_SIZE,
            density=RANDOM_DENSITY,
            format="csr",
            random_state=np.random.default_rng(SEED),
            data_rvs=np.ones,
        )
    )


DATASETS = {
    "Diagonal": partial(make_diagonal, N),
    "Tridiagonal": partial(make_tridiagonal, N),
    "Banded": partial(make_banded, N),
    "Triangular": partial(make_triangular, N),
    "Striped": partial(make_striped, N),
    "Uniform Random": make_uniform_random,
    "ct20stif": partial(load_matrix_market, "ct20stif.mtx"),
    "roadNet-PA": partial(load_matrix_market, "roadNet-PA.mtx"),
    "soc-sign-epinions": partial(load_matrix_market, "soc-sign-epinions.mtx"),
}


def make_kernel_estimator(factory, tensor):
    @cache
    def stats(fields):
        return factory(tensor, fields)

    @cache
    def matmul(left, right, reduced):
        return factory.aggregate(
            ffuncs.add,
            0.0,
            (reduced,),
            factory.mapjoin(ffuncs.mul, stats(left), stats(right)),
        )

    def estimate(kernel):
        match kernel:
            case "Hadamard":
                result = factory.mapjoin(ffuncs.mul, stats((i, j)), stats((i, j)))
            case "SpGEMM":
                result = matmul((i, k), (k, j), k)
            case "SpGEMM2":
                result = factory.aggregate(
                    ffuncs.add,
                    0.0,
                    (k,),
                    factory.mapjoin(
                        ffuncs.mul, matmul((i, ell), (ell, k), ell), stats((k, j))
                    ),
                )
            case "Triangle Counting":
                result = factory.mapjoin(
                    ffuncs.mul, matmul((i, k), (k, j), k), stats((i, j))
                )
            case _:
                raise ValueError(f"Unknown statistics kernel: {kernel}")
        return result.estimate_non_fill_values()

    return estimate
