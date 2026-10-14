"""
Test four common operations:
 - 1. D[i] = A[i] + B[i] + C[i]
 - 2. D[i] = A[i] *  B[i] * C[i]
               with `*`: hadamard product aka
                     elementwise multiplication
 - 3. D = sum_ijk A[i, j] A[j, k] A[i, k]
 - 4. D[j, i] = A[i, j]
"""

import pytest

import numpy as np
import scipy.sparse as sp

import finch as ft

NUMPY_RANDOM_DEFAULT_RNG_SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(NUMPY_RANDOM_DEFAULT_RNG_SEED)


def compute_tr3(a):
    return ft.einsum("ij,jk,ik->", a, a, a)


def create_random_fill_matrix(nrow, ncol, sparsity_proportion, rng, random_fill):
    """
    Create a sparse matrix with a given sparsity proportion.
    The rng function should be a np.random.default_rng (or be an object
    which implements an in place shuffle on numpy arrays). random_fill
    should generate numbers to fill the non-fill values.
    """

    num_entries = int((nrow * ncol) * (1 - sparsity_proportion))

    row, col = np.meshgrid(
        np.asarray(np.arange(nrow)), np.asarray(np.arange(ncol)), indexing="ij"
    )

    row = np.ravel(row)
    col = np.ravel(col)

    rng.shuffle(row)
    rng.shuffle(col)

    row = row[0:num_entries]
    col = col[0:num_entries]

    values = random_fill(num_entries)

    assert values.shape[0] == row.shape[0]

    return ft.FiberTensor.from_scipy_csr(
        sp.csr_array((values, (row, col)), shape=(nrow, ncol))
    )


def create_random_fill_vector(nrow, sparsity_proportion, rng, random_fill):
    """
    Create a vector with a provided cardinality (nrow),
    sparsity proportion. The rng function should be a
    `np.random.default_rng(<seed>)`, though _technically_
    any object with an in place shuffle algorithm for numpy
    arrays will work. The random_fill function will
    be called once and it should provide arbitrary values
    for vector.
    """
    row_entries = np.asarray(np.arange(nrow))
    rng.shuffle(row_entries)

    row = row_entries[0 : int(nrow * (1 - sparsity_proportion))]
    col = np.zeros(row.shape[0])

    values = random_fill(row.shape[0])

    return ft.FiberTensor.from_scipy_csr(
        sp.csr_array((values, (row, col)), shape=(nrow, 1))
    )


@pytest.mark.parametrize("matrix_dimension", [2000])
@pytest.mark.parametrize("sparsity", [0.95])
def test_compute_tr3(scheduler, benchmark, rng, matrix_dimension, sparsity):
    filled_matrix = create_random_fill_matrix(
        matrix_dimension,
        matrix_dimension,
        sparsity,
        rng,
        lambda x: rng.integers(1, 10 + 1, x),
    )

    benchmark(compute_tr3, filled_matrix)


@pytest.fixture
def large_sparse_vector_triplet(rng):
    """
    Create a constructor that returns three sparse vectors, given
    a dimension and a sparsity.
    """

    # basically use the fact that (a -> b -> c) is equivalent to
    # a -> (b -> c), to produce a way to create large sparse vectors
    # that both abstracts away the randomness and presents the dimensions
    # and sparsity instead.
    def _inner(dimension, sparsity):
        sparse_vector_a = create_random_fill_vector(
            dimension, sparsity, rng, lambda x: rng.integers(1, 10, x)
        )

        sparse_vector_b = create_random_fill_vector(
            dimension, sparsity, rng, lambda x: rng.integers(1, 10, x)
        )

        sparse_vector_c = create_random_fill_vector(
            dimension, sparsity, rng, lambda x: rng.integers(1, 10, x)
        )

        return (sparse_vector_a, sparse_vector_b, sparse_vector_c)

    return _inner


@pytest.mark.parametrize("dimension", [10**7])
@pytest.mark.parametrize("sparsity", [0.95])
def test_addition(
    large_sparse_vector_triplet, scheduler, benchmark, dimension, sparsity
):
    (sparse_vector_a, sparse_vector_b, sparse_vector_c) = large_sparse_vector_triplet(
        dimension, sparsity
    )
    benchmark(
        lambda a, b, c: a + b + c, sparse_vector_a, sparse_vector_b, sparse_vector_c
    )


@pytest.mark.parametrize("dimension", [10**7])
@pytest.mark.parametrize("sparsity", [0.95])
def test_multiplication(
    large_sparse_vector_triplet, scheduler, benchmark, dimension, sparsity
):
    (sparse_vector_a, sparse_vector_b, sparse_vector_c) = large_sparse_vector_triplet(
        dimension, sparsity
    )

    benchmark(
        lambda a, b, c: a * b * c, sparse_vector_a, sparse_vector_b, sparse_vector_c
    )


@pytest.mark.parametrize("dimension", [100])
@pytest.mark.parametrize("sparsity", [0.80])
def test_compute_transpose(scheduler, benchmark, dimension, sparsity, rng):
    rng = np.random.default_rng(42)
    large_sparse_array = create_random_fill_matrix(
        dimension, dimension, sparsity, rng, lambda x: rng.integers(1, 10, x)
    )

    benchmark(ft.matrix_transpose, large_sparse_array)
