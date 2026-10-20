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
VECTOR_SPARSITY_RUNS = [0.80, 0.99]

rng = np.random.default_rng(NUMPY_RANDOM_DEFAULT_RNG_SEED)


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


def create_random_fill_vector(
    element_count, number_of_non_fill_values, rng: np.random.Generator, fill_value=0
) -> ft.FiberTensor:
    random_values = rng.integers(1, 10, number_of_non_fill_values)
    indicies = np.arange(0, element_count)
    rng.shuffle(indicies)
    indicies = indicies[:number_of_non_fill_values]

    return ft.FiberTensor(
        ft.SparseListLevel(
            ft.ElementLevel(
                _format=ft.element(
                    ft.ftype(random_values.dtype)(fill_value),
                    ft.ftype(random_values.dtype),
                    np.int64,
                ),
                _val=ft.NumpyBuffer(random_values),
            ),
            dimension=ft.int64(element_count),
            ptr=ft.NumpyBuffer(
                np.array([0, number_of_non_fill_values], dtype=np.int64)
            ),
            idx=ft.NumpyBuffer(indicies),
        )
    )


@pytest.mark.parametrize("matrix_dimension", [2000])
@pytest.mark.parametrize("sparsity", [0.95])
def test_compute_tr3(scheduler, benchmark, matrix_dimension, sparsity):
    filled_matrix = create_random_fill_matrix(
        matrix_dimension,
        matrix_dimension,
        sparsity,
        rng,
        lambda x: rng.integers(1, 10 + 1, x),
    )

    benchmark(compute_tr3, filled_matrix)


def sparse_vector_gen(dimension, sparsity):
    nnf = int(dimension * (1 - sparsity))
    print(f"{nnf=}")

    while True:
        yield create_random_fill_vector(dimension, nnf, rng)


@pytest.mark.parametrize("dimension", [10**7])
@pytest.mark.parametrize("sparsity", VECTOR_SPARSITY_RUNS)
def test_addition(scheduler, benchmark, dimension, sparsity):
    vgen = sparse_vector_gen(dimension, sparsity)
    benchmark(lambda a, b, c: a + b + c, next(vgen), next(vgen), next(vgen))


@pytest.mark.parametrize("dimension", [10**7])
@pytest.mark.parametrize("sparsity", VECTOR_SPARSITY_RUNS)
def test_multiplication(scheduler, benchmark, dimension, sparsity):
    vgen = sparse_vector_gen(dimension, sparsity)

    benchmark(lambda a, b, c: a * b * c, next(vgen), next(vgen), next(vgen))


@pytest.mark.parametrize("dimension", [500])
# ~ one in every 750 elements is sparse (so just under 1 nnf per row)
# ~ five entries of every 500 elements (row) is non-fill
# ~ four in every five entries is non-fill
@pytest.mark.parametrize(
    "sparsity",
    [
        1 - 1 / 750,
        1 - 5 / 500,
        1 - 4 / 5,
    ],
)
def test_compute_transpose(scheduler, benchmark, dimension, sparsity):
    rng = np.random.default_rng(42)
    large_sparse_array = create_random_fill_matrix(
        dimension, dimension, sparsity, rng, lambda x: rng.integers(1, 10, x)
    )

    benchmark(ft.matrix_transpose, large_sparse_array)
