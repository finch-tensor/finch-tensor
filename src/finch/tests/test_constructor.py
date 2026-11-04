import pytest

import numpy as np
import scipy.sparse as sps

from finch.interface.constructor import (
    from_numpy,
    from_parts_coo,
    from_parts_csr,
    from_pydata,
    from_scipy,
)

from .conftest import finch_assert_equal


@pytest.mark.parametrize(
    "a",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array(5.0),
    ],
)
def test_from_numpy_wraps_array(a):
    t = from_numpy(a)
    finch_assert_equal(t, a)


@pytest.mark.parametrize(
    "a",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_from_numpy_is_zero_copy(a):
    t = from_numpy(a)
    a.flat[0] = 99.0
    assert t.to_numpy().flat[0] == 99.0


@pytest.mark.parametrize(
    "a",
    [
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]),
    ],
)
def test_from_scipy_csr_wraps_matrix(a):
    mat = sps.csr_matrix(a)
    t = from_scipy(mat)
    np.testing.assert_array_equal(t.to_scipy().toarray(), mat.toarray())


@pytest.mark.parametrize(
    "data, indices, indptr, shape, expected",
    [
        (
            np.array([1.0, 2.0]),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            (1, 2),
            np.array([[1.0, 2.0]]),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 2, 1], dtype=np.int64),
            np.array([0, 2, 3], dtype=np.int64),
            (2, 3),
            np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]),
        ),
    ],
)
def test_from_parts_csr(data, indices, indptr, shape, expected):
    t = from_parts_csr(data, indices, indptr, shape=shape)
    np.testing.assert_array_equal(t.to_scipy().toarray(), expected)


def test_from_parts_csr_is_zero_copy():
    data = np.array([1.0, 2.0])
    indices = np.array([0, 1], dtype=np.int64)
    indptr = np.array([0, 2], dtype=np.int64)
    t = from_parts_csr(data, indices, indptr, shape=(1, 2))
    data[0] = 99.0
    assert t.to_scipy().toarray()[0, 0] == 99.0


@pytest.mark.parametrize(
    "data, indices, indptr, shape, match",
    [
        (
            np.array([1.0]),
            np.array([0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            (1, 1, 1),
            "2D",
        ),
        (
            np.array([1.0]),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            (1, 1),
            "indptr",
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            (1, 2),
            "indices",
        ),
    ],
)
def test_from_parts_csr_rejects_malformed(data, indices, indptr, shape, match):
    with pytest.raises(ValueError, match=match):
        from_parts_csr(data, indices, indptr, shape=shape)


def test_from_pydata_coo_roundtrip():
    sparse = pytest.importorskip("sparse")
    coo = sparse.COO(coords=[[0, 1], [0, 1]], data=[1.0, 2.0], shape=(2, 2))
    t = from_pydata(coo)
    np.testing.assert_array_equal(t.to_scipy().toarray(), coo.todense())


@pytest.mark.parametrize(
    "coords, data, shape, expected",
    [
        (
            np.array([[0, 1], [0, 1]], dtype=np.int64),
            np.array([1.0, 2.0]),
            (2, 2),
            np.array([[1.0, 0.0], [0.0, 2.0]]),
        ),
        (
            np.array([[0, 2], [2, 1]], dtype=np.int64),
            np.array([3.0, 5.0]),
            (3, 3),
            np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        ),
    ],
)
def test_from_parts_coo_roundtrip(coords, data, shape, expected):
    t = from_parts_coo(coords, data, shape=shape)
    np.testing.assert_array_equal(t.to_scipy().toarray(), expected)


def test_from_scipy_csr_is_zero_copy():
    source = sps.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    t = from_scipy(source, copy=False).to_scipy()

    np.testing.assert_array_equal(t.toarray(), source.toarray())
    assert np.shares_memory(t.data, source.data)
    assert np.shares_memory(t.indices, source.indices)
    assert np.shares_memory(t.indptr, source.indptr)


def test_from_pydata_returns_fiber_tensor():
    from finch.tensor import BufferizedNDArray, FiberTensor

    sparse = pytest.importorskip("sparse")
    coo = sparse.COO(coords=[[0, 1], [0, 1]], data=[1.0, 2.0], shape=(2, 2))
    t = from_pydata(coo)
    assert isinstance(t, FiberTensor)
    assert not isinstance(t, BufferizedNDArray)


def test_from_parts_csr_returns_fiber_tensor():
    from finch.tensor import BufferizedNDArray, FiberTensor

    data = np.array([1.0, 2.0])
    indices = np.array([0, 1], dtype=np.int64)
    indptr = np.array([0, 2], dtype=np.int64)
    t = from_parts_csr(data, indices, indptr, shape=(1, 2))
    assert isinstance(t, FiberTensor)
    assert not isinstance(t, BufferizedNDArray)


def test_from_parts_coo_returns_fiber_tensor():
    from finch.tensor import BufferizedNDArray, FiberTensor

    coords = np.array([[0, 1], [0, 1]], dtype=np.int64)
    data = np.array([1.0, 2.0])
    t = from_parts_coo(coords, data, shape=(2, 2))
    assert isinstance(t, FiberTensor)
    assert not isinstance(t, BufferizedNDArray)


@pytest.mark.parametrize(
    "coords, data, shape, match",
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([1.0, 2.0]),
            (2, 2),
            "2D",
        ),
        (
            np.array([[0, 1], [0, 1]], dtype=np.int64),
            np.array([1.0, 2.0]),
            (2, 2, 2),
            "shape",
        ),
        (
            np.array([[0, 1], [0, 1]], dtype=np.int64),
            np.array([1.0]),
            (2, 2),
            "second dimension",
        ),
    ],
)
def test_from_parts_coo_rejects_malformed(coords, data, shape, match):
    with pytest.raises(ValueError, match=match):
        from_parts_coo(coords, data, shape=shape)


@pytest.mark.parametrize(
    "a",
    [
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]),
    ],
)
def test_from_scipy_coo_wraps_matrix(a):
    mat = sps.coo_matrix(a)
    t = from_scipy(mat)
    np.testing.assert_array_equal(t.to_scipy().toarray(), mat.toarray())
