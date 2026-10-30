import pytest

import numpy as np
import scipy.sparse as sps

import finch
from finch.tensor import FiberTensor


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_issue_64():
    a = finch.defer(np.arange(1 * 2).reshape(1, 2, 1))
    b = finch.defer(np.arange(4 * 2 * 3).reshape(4, 2, 3))

    c = finch.multiply(a, b)
    result = finch.compute(c).shape
    expected = (4, 2, 3)
    assert result == expected, f"Expected shape {expected}, got {result}"


def test_issue_50():
    x = finch.defer(np.array([[2, 4, 6, 8], [1, 3, 5, 7]]))
    m = finch.defer(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    n = finch.defer(
        np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    )  # Int -> Float caused return_type error
    # If replaced above with below line, no error
    # n = finch.defer(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    o = finch.defer(np.array([[3, 3, 3, 3], [3, 3, 3, 3]]))
    finch.add(finch.add(finch.subtract(x, m), n), o)


@pytest.mark.parametrize(
    ("shapes", "density"),
    [
        ((2, 2, 2, 2), 0.01),  # the exact repro from the issue: all-empty matrices
        ((2, 2, 2, 2), 0.9),
        ((8, 5, 7, 6), 0.4),
        ((20, 30, 25, 15), 0.15),
    ],
)
@pytest.mark.usefixtures("numba_compiler")
def test_issue_620(rng, shapes, density):
    """Chained matmul over scipy CSR inputs, whose indices are int32."""
    n, m, k, ell = shapes
    a, b, c = (
        sps.random(i, j, density=density, format="csr", random_state=rng)
        for i, j in ((n, m), (m, k), (k, ell))
    )
    a_f, b_f, c_f = (FiberTensor.from_scipy_csr(x) for x in (a, b, c))

    result = finch.compute(
        finch.matmul(
            finch.matmul(finch.defer(a_f), finch.defer(b_f)),
            finch.defer(c_f),
        )
    )

    np.testing.assert_allclose(np.asarray(result), (a @ b @ c).toarray())
