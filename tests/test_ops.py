import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from functools import reduce
import juliacall as jc

import finch


arr1d = np.array([1, 1, 2, 3])
arr2d = np.array([[1, 2, 0, 0], [0, 1, 0, 1]])
arr3d = np.array(
    [
        [[0, 1, 0, 0], [1, 0, 0, 3]],
        [[4, 0, -1, 0], [2, 2, 0, 0]],
        [[0, 0, 0, 0], [1, 5, 0, 3]],
    ]
)


@pytest.fixture(
    scope="module",
    params=[finch.DefaultScheduler(), finch.GalleyScheduler()],
    ids=["default", "galley"],
)
def opt(request):
    finch.set_optimizer(request.param)
    yield request.param


def test_eager(arr3d, opt):
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)

    result = finch.multiply(A_finch, B_finch)

    assert_equal(result.todense(), np.multiply(arr3d, arr2d))


def test_lazy_mode(arr3d, opt):
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)
    C_finch = finch.Tensor(arr1d)

    @finch.compiled(opt=opt)
    def my_custom_fun(arr1, arr2, arr3):
        temp = finch.multiply(arr1, arr2)
        temp = finch.divide(temp, arr3)
        reduced = finch.sum(temp, axis=(0, 1))
        return finch.add(temp, reduced)

    result = my_custom_fun(A_finch, B_finch, C_finch)

    temp = np.divide(np.multiply(arr3d, arr2d), arr1d)
    expected = np.add(temp, np.sum(temp, axis=(0, 1)))
    assert_equal(result.todense(), expected)

    A_lazy = finch.lazy(A_finch)
    B_lazy = finch.lazy(B_finch)
    mul_lazy = finch.multiply(A_lazy, B_lazy)
    result = finch.compute(mul_lazy)

    assert_equal(result.todense(), np.multiply(arr3d, arr2d))


def test_lazy_mode_mult_output(opt):
    A_finch = finch.Tensor(arr1d)
    B_finch = finch.Tensor(arr2d)

    @finch.compiled(opt=opt)
    def mult_out_fun(arr1, arr2):
        out1 = finch.add(arr1, arr2)
        out2 = finch.multiply(arr1, arr2)
        out3 = arr2 ** finch.asarray(2)
        return out1, out2, out3

    res1, res2, res3 = mult_out_fun(A_finch, B_finch)

    assert_equal(res1.todense(), np.add(arr1d, arr2d))
    assert_equal(res2.todense(), np.multiply(arr1d, arr2d))
    assert_equal(res3.todense(), arr2d**2)


def test_lazy_mode_heterogenous_output():
    A_finch = finch.Tensor(arr1d)
    B_finch = finch.Tensor(arr2d)

    @finch.compiled()
    def heterogenous_fun(a: list[finch.Tensor], b: int):
        sum_a = reduce(lambda x1, x2: x1 + x2, a)
        b_squared = b**2
        return (a, sum_a, (b, "text"), {"key1": 12, "key2": b_squared})

    ret = heterogenous_fun([A_finch, B_finch], 3)

    assert type(ret) is tuple
    assert len(ret) == 4
    assert type(ret[0]) is list
    assert len(ret[0]) == 2
    assert_equal(ret[0][0].todense(), arr1d)
    assert_equal(ret[0][1].todense(), arr2d)
    assert_equal(ret[1].todense(), arr1d + arr2d)
    assert ret[2] == (3, "text")
    assert type(ret[3]) is dict
    assert ret[3] == {"key1": 12, "key2": 9}


@pytest.mark.parametrize(
    "func_name",
    [
        "log",
        "log10",
        "log1p",
        "log2",
        "sqrt",
        "sign",
        "round",
        "exp",
        "expm1",
        "floor",
        "ceil",
        "isnan",
        "isfinite",
        "isinf",
        "square",
        "trunc",
    ],
)
def test_elemwise_ops_1_arg(arr3d, func_name, opt):
    arr = arr3d + 1.6
    A_finch = finch.Tensor(arr)

    actual = getattr(finch, func_name)(A_finch)
    expected = getattr(np, func_name)(arr)

    assert_allclose(actual.todense(), expected)


@pytest.mark.parametrize("func_name", ["real", "imag", "conj"])
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.float64, np.int64])
def test_elemwise_complex_ops_1_arg(func_name, dtype, opt):
    arr = np.asarray([[1 + 1j, 2 + 2j], [3 + 3j, 4 - 4j], [-5 - 5j, -6 - 6j]]).astype(
        dtype
    )
    arr_finch = finch.asarray(arr)

    actual = getattr(finch, func_name)(arr_finch)
    expected = getattr(np, func_name)(arr)

    assert_allclose(actual.todense(), expected)
    assert actual.todense().dtype == expected.dtype


@pytest.mark.parametrize(
    "meth_name",
    ["__pos__", "__neg__", "__abs__", "__invert__"],
)
def test_elemwise_tensor_ops_1_arg(arr3d, meth_name, opt):
    A_finch = finch.Tensor(arr3d)

    actual = getattr(A_finch, meth_name)()
    expected = getattr(arr3d, meth_name)()

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize(
    "func_name",
    ["logaddexp", "logical_and", "logical_or", "logical_xor"],
)
def test_elemwise_ops_2_args(arr3d, func_name, opt):
    arr2d = np.array([[0, 3, 2, 0], [0, 0, 3, 2]])
    if func_name.startswith("logical"):
        arr3d = arr3d.astype(bool)
        arr2d = arr2d.astype(bool)
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)

    actual = getattr(finch, func_name)(A_finch, B_finch)
    expected = getattr(np, func_name)(arr3d, arr2d)

    assert_allclose(actual.todense(), expected)


@pytest.mark.parametrize(
    "meth_name",
    [
        "__add__",
        "__mul__",
        "__sub__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__and__",
        "__or__",
        "__xor__",
        "__lshift__",
        "__rshift__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__eq__",
        "__ne__",
    ],
)
def test_elemwise_tensor_ops_2_args(arr3d, meth_name, opt):
    arr2d = np.array([[2, 3, 2, 3], [3, 2, 3, 2]])
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)

    actual = getattr(A_finch, meth_name)(B_finch)
    expected = getattr(arr3d, meth_name)(arr2d)

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("func_name", ["sum", "prod", "max", "min", "any", "all", "mean", "std", "var"])
@pytest.mark.parametrize("axis", [None, -1, 1, (0, 1), (0, 1, 2)])
def test_reductions(arr3d, func_name, axis, opt):
    A_finch = finch.Tensor(arr3d)

    actual = getattr(finch, func_name)(A_finch, axis=axis)
    expected = getattr(np, func_name)(arr3d, axis=axis)

    assert_equal(actual.todense(), expected)

@pytest.mark.parametrize("func_name", ["argmax", "argmin"])
@pytest.mark.parametrize("axis", [None, -1, 1, 2, (0, 1, 2)])
def test_reductions(arr3d, func_name, axis, opt):
    A_finch = finch.Tensor(arr3d)

    actual = getattr(finch, func_name)(A_finch, axis=axis)
    expected = getattr(np, func_name)(arr3d, axis=axis)

    assert_equal(actual.todense(), expected)

@pytest.mark.parametrize("axis", [-1, 1, (0, 1), (0, 1, 2)])
def test_reductions(arr3d, axis, opt):
    A_finch = finch.Tensor(arr3d)

    actual = finch.expand_dims(A_finch, axis=axis)
    expected = np.expand_dims(arr3d, axis=axis)

    assert_equal(actual.todense(), expected)

    actual = finch.squeeze(actual, axis=axis)
    expected = np.squeeze(expected, axis=axis)

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("func_name", ["sum", "prod"])
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize(
    "in_dtype, dtype, expected_dtype",
    [
        (finch.int64, None, np.int64),
        (finch.int16, None, np.int64),
        (finch.uint8, None, np.uint64),
        (finch.int64, finch.float32, np.float32),
        (finch.float64, finch.complex128, np.complex128),
    ],
)
def test_sum_prod_dtype_arg(
    arr3d, func_name, axis, in_dtype, dtype, expected_dtype, opt
):
    arr_finch = finch.asarray(np.abs(arr3d), dtype=in_dtype)

    actual = getattr(finch, func_name)(arr_finch, axis=axis, dtype=dtype).todense()

    assert actual.dtype == expected_dtype


@pytest.mark.parametrize(
    "storage",
    [
        None,
        (
            finch.Storage(finch.SparseList(finch.Element(np.int64(0))), order="C"),
            finch.Storage(
                finch.Dense(finch.SparseList(finch.Element(np.int64(0)))), order="C"
            ),
            finch.Storage(
                finch.Dense(
                    finch.SparseList(finch.SparseList(finch.Element(np.int64(0))))
                ),
                order="C",
            ),
        ),
    ],
)
def test_tensordot(arr3d, storage, opt):
    A_finch = finch.Tensor(arr1d)
    B_finch = finch.Tensor(arr2d)
    C_finch = finch.Tensor(arr3d)
    if storage is not None:
        A_finch = A_finch.to_storage(storage[0])
        B_finch = B_finch.to_storage(storage[1])
        C_finch = C_finch.to_storage(storage[2])

    actual = finch.tensordot(B_finch, B_finch)
    expected = np.tensordot(arr2d, arr2d)
    assert_equal(actual.todense(), expected)

    actual = finch.tensordot(B_finch, B_finch, axes=(1, 1))
    expected = np.tensordot(arr2d, arr2d, axes=(1, 1))
    assert_equal(actual.todense(), expected)

    actual = finch.tensordot(
        C_finch, finch.permute_dims(C_finch, (2, 1, 0)), axes=((2, 0), (0, 2))
    )
    expected = np.tensordot(arr3d, arr3d.T, axes=((2, 0), (0, 2)))
    assert_equal(actual.todense(), expected)

    actual = finch.tensordot(C_finch, A_finch, axes=(2, 0))
    expected = np.tensordot(arr3d, arr1d, axes=(2, 0))
    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (arr2d, arr2d.mT),
        (arr2d, arr3d.mT),
        (arr2d.mT, arr3d),
        (arr3d, arr3d.mT),
        (arr1d, arr1d),
        (arr1d, arr2d.mT),
        (arr2d, arr1d),
        (arr1d, arr3d.mT),
        (arr3d, arr1d),
    ],
)
def test_matmul(opt, a: np.ndarray, b: np.ndarray):
    A_finch = finch.Tensor(a)
    B_finch = finch.Tensor(b)

    expected = a @ b
    actual = A_finch @ B_finch

    assert_equal(actual.todense(), expected)

    if a.ndim >= 2 and b.ndim >= 2:
        At_finch = A_finch.mT
        Bt_finch = B_finch.mT

        assert_equal((Bt_finch @ At_finch).todense(), expected.mT)


def test_matmul_dimension_mismatch(opt):
    A_finch = finch.Tensor(arr2d)
    B_finch = finch.Tensor(arr3d)

    with pytest.raises(jc.JuliaError, match="DimensionMismatch"):
        A_finch @ B_finch


def test_negative__mod__(opt):
    arr = np.array([-1, 0, 0, -2, -3, 0])
    arr_finch = finch.asarray(arr)

    actual = arr_finch % 5
    expected = arr % 5
    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("force_materialization", [False, True])
def test_recursive_compiled(
    opt, force_materialization: bool, arr3d: finch.Tensor
) -> None:
    decorator = finch.compiled(opt=opt, force_materialization=force_materialization)

    @decorator
    def my_custom_fun_inner(
        arr1: finch.Tensor, arr2: finch.Tensor, arr3: finch.Tensor
    ) -> finch.Tensor:
        temp = finch.multiply(arr1, arr2)
        temp = finch.divide(temp, arr3)
        reduced = finch.sum(temp, axis=(0, 1))
        return finch.add(temp, reduced)

    @decorator
    def my_custom_fun_outer(
        arr1: finch.Tensor, arr2: finch.Tensor, arr3: finch.Tensor
    ) -> finch.Tensor:
        arr = my_custom_fun_inner(arr1, arr2, arr3)
        assert arr.is_computed() == force_materialization
        return arr

    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)
    C_finch = finch.Tensor(arr1d)

    result = my_custom_fun_outer(A_finch, B_finch, C_finch)
    assert result.is_computed()
