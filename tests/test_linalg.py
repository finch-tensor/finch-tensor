import pytest

import numpy as np
from numpy.testing import assert_allclose

import finch

arr1d = np.array([1, -1, 2, 3])
arr2d = np.array([[1, 2, 0, 4, 0], [0, -2, 1, 0, 1]])


@pytest.mark.parametrize("arr", [arr1d, arr2d])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize(
    "ord",
    [
        0,
        1,
        10,
        finch.inf,
        -finch.inf,
        pytest.param(
            2,
            marks=pytest.mark.skip(
                reason="https://github.com/finch-tensor/Finch.jl/pull/709"
            ),
        ),
    ],
)
def test_vector_norm(arr, keepdims, ord):
    tns = finch.asarray(arr)

    actual = finch.linalg.vector_norm(tns, keepdims=keepdims, ord=ord)
    expected = np.linalg.vector_norm(arr, keepdims=keepdims, ord=ord)

    assert_allclose(actual.todense(), expected)
