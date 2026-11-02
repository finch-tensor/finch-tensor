import logging

import pytest

import numpy as np

import finch as ft
from finch import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    dense,
    element,
    fiber_tensor,
    ftype,
)
from finch.symbolic.gensym import _sg

from .conftest import reset_name_counts


@pytest.mark.usefixtures("numba_compiler")
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_selected_ops(dtype):
    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))

    elem_ftype = element(dtype(0), ftype(dtype), ftype(np.intp(0)), NumpyBufferFType)

    a = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(elem_ftype, data),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    a_np = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], dtype=dtype)

    fmt = fiber_tensor(
        dense(
            dense(element(dtype(0), ftype(dtype), ftype(np.intp(0)), NumpyBufferFType))
        )
    )
    b_np = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]], dtype=dtype)
    b = ft.asarray(b_np, format=fmt)

    la = ft.lazy(a)
    lb = ft.lazy(b)

    plan = ft.sum(la, axis=0)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), np.sum(a_np, axis=0))

    plan = ft.sum(la, axis=1)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), np.sum(a_np, axis=1))

    plan = ft.multiply(la, lb)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), a_np * b_np)

    plan = ft.add(la, lb)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), a_np + b_np)


def test_asm_sparse_elemwise(file_regression, caplog, numba_compiler):
    _sg.counter = 0

    dtype = np.float64
    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))

    a = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(
                    element(
                        dtype(0), ft.ftype(dtype), ft.ftype(np.intp), NumpyBufferFType
                    ),
                    data,
                ),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    b = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(
                    element(
                        dtype(0), ft.ftype(dtype), ft.ftype(np.intp), NumpyBufferFType
                    ),
                    data,
                ),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    la = ft.lazy(a)
    lb = ft.lazy(b)

    class DummyHandler(logging.Handler):
        def __init__(self, level):
            super().__init__(level=level)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = DummyHandler(logging.DEBUG)
    log = logging.getLogger("finch.compile.lower")
    log.addHandler(handler)
    log.propagate = False

    with caplog.at_level(logging.DEBUG, logger="finch.compile.lower"):
        result = ft.multiply(la, lb)
        _ = ft.compute(result)

    log.propagate = True

    assert len(handler.records) == 1
    file_regression.check(
        reset_name_counts(str(handler.records[0].msg)), extension=".txt"
    )


def _sparse_list(arr: np.ndarray) -> SparseListLevel:
    """Store the last dimension of `arr` in a sparse list level, CSR style."""
    rows = arr.reshape(-1, arr.shape[-1])
    ptr, idx = [0], list[int]()
    for row in rows:
        idx.extend(np.flatnonzero(row))
        ptr.append(len(idx))
    elem_ftype = element(
        arr.dtype.type(0), ftype(arr.dtype.type), ftype(np.intp), NumpyBufferFType
    )
    return SparseListLevel(
        ElementLevel(elem_ftype, NumpyBuffer(rows[rows != 0])),
        np.intp(arr.shape[-1]),
        NumpyBuffer(np.array(ptr, dtype=np.intp)),
        NumpyBuffer(np.array(idx, dtype=np.intp)),
    )


def _sparse_tensor(arr: np.ndarray) -> FiberTensor:
    match arr.shape:
        case (_,):
            return FiberTensor(_sparse_list(arr))
        case (m, _):
            return FiberTensor(DenseLevel(_sparse_list(arr), np.intp(m)))
    raise ValueError(arr.shape)


@pytest.mark.usefixtures("numba_compiler")
@pytest.mark.parametrize(
    ("ft_op", "np_op"),
    [
        (ft.add, np.add),
        (ft.subtract, np.subtract),
        (ft.multiply, np.multiply),
    ],
)
def test_sparse_sparse_elementwise(ft_op, np_op, rng):
    cases = [
        # empty rows next to stored ones
        (np.array([[1, 2], [0, 0]]), np.array([[0, 0], [3, 0]])),
        (np.array([[0], [1], [0], [5], [8]]), np.zeros((5, 1), dtype=int)),
        # stepping past the last stored index of one of the operands
        (np.array([5, 0, 3, 0]), np.array([0, 0, 1, 0])),
        (np.array([5, 0, 3, 0]), np.array([6, 0, 8, 2])),
        (np.array([0, 0, 0, 0, 0, 5]), np.array([0, 2, 0, 7, 0, 0])),
    ]
    for _ in range(10):
        shape = tuple(rng.integers(1, 7, 2))
        a, b = (rng.integers(1, 9, shape) * (rng.random(shape) < 0.4) for _ in range(2))
        cases += [(a, b), (a[0], b[0])]

    for a, b in cases:
        res = ft_op(_sparse_tensor(a), _sparse_tensor(b))
        np.testing.assert_array_equal(res.to_numpy(), np_op(a, b))
