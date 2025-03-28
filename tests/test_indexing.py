import numpy as np
from numpy.testing import assert_equal
import pytest
import juliacall as jc

import finch


@pytest.mark.parametrize(
    "index",
    [
        ...,
        40,
        (32,),
        slice(None),
        slice(30, 60, 3),
        -10,
        slice(None, -10, -2),
        (None, slice(None)),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_indexing_1d(arr1d, index, order):
    arr = np.array(arr1d, order=order)
    arr_finch = finch.Tensor(arr)

    actual = arr_finch[index]
    expected = arr[index]

    if isinstance(actual, finch.Tensor):
        actual = actual.todense()

    assert_equal(actual, expected)


@pytest.mark.parametrize(
    "index",
    [
        ...,
        0,
        (2,),
        (2, 3),
        slice(None),
        (..., slice(0, 4, 2)),
        (-1, slice(-1, None, -1)),
        (None, slice(None), slice(None)),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_indexing_2d(arr2d, index, order):
    arr = np.array(arr2d, order=order)
    arr_finch = finch.Tensor(arr)

    actual = arr_finch[index]
    expected = arr[index]

    if isinstance(actual, finch.Tensor):
        actual = actual.todense()

    assert_equal(actual, expected)


@pytest.mark.parametrize(
    "index",
    [
        (0, 1, 2),
        (1, 0, 0),
        (0, 1),
        1,
        2,
        (2, slice(None), 3),
        (slice(None), 0),
        slice(None),
        (0, slice(None), slice(1, 4, 2)),
        (0, 1, ...),
        (..., 1),
        (0, ..., 1),
        ...,
        (..., slice(1, 4, 2)),
        (slice(None, None, -1), slice(None, None, -1), slice(None, None, -1)),
        (slice(None, -1, 1), slice(-1, None, -1), slice(4, 1, -1)),
        (-1, 0, 0),
        (0, -1, -2),
        ([1, 2], 0, slice(3, None, -1)),
        (0, slice(1, 0, -1), 0),
        (slice(None), None, slice(None), slice(None)),
        (slice(None), slice(None), slice(None), None),
    ],
)
@pytest.mark.parametrize(
    "levels_descr",
    [
        finch.Dense(finch.Dense(finch.Dense(finch.Element(0)))),
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0)))),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_indexing_3d(arr3d, index, levels_descr, order):
    arr = np.array(arr3d, order=order)
    storage = finch.Storage(levels_descr, order=order)
    arr_finch = finch.Tensor(arr).to_storage(storage)

    actual = arr_finch[index]
    expected = arr[index]

    if isinstance(actual, finch.Tensor):
        actual = actual.todense()

    assert_equal(actual, expected)


def test_lazy_none_ellipsis(arr3d):
    arr_finch = finch.lazy(finch.Tensor(arr3d))
    assert_equal(finch.compute(arr_finch[..., None]).todense(), arr3d[..., None])

    with pytest.raises(
        jc.JuliaError,
        match="Cannot index a lazy tensor with more or fewer `:` dims than it had original dims.",
    ):
        arr_finch[None, :]
