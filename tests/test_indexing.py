import pytest

from numpy.testing import assert_equal

from juliacall import Main as jl

from finch import FinchJLTensor


@pytest.mark.parametrize(
    "index",
    [
        40,
        (32,),
        slice(30, 60, 3),
        -10,
        slice(None, -10, -2),
        (None, slice(None)),
        # The following two tests are commented out since Finch.jl
        # returns errors for them
        #
        # ...,
        # slice(None),
    ],
)
def test_indexing_1d(arr1d, index):
    arr_finch = FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Element(0)), arr1d))

    actual = arr_finch[index]
    expected = arr1d[index]

    if isinstance(actual, FinchJLTensor):
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
def test_indexing_2d(arr2d, index):
    arr_finch = FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(0))), arr2d))

    actual = arr_finch[index]
    expected = arr2d[index]

    if isinstance(actual, FinchJLTensor):
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
def test_indexing_3d(arr3d, index):
    arr_finch = FinchJLTensor(
        jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Dense(jl.Element(0)))), arr3d)
    )

    actual = arr_finch[index]
    expected = arr3d[index]

    if isinstance(actual, FinchJLTensor):
        actual = actual.todense()

    assert_equal(actual, expected)
