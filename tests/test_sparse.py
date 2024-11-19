import numpy as np
from numpy.testing import assert_equal
import pytest
import sparse

import finch


@pytest.mark.parametrize(
    "dtype,jl_dtype",
    [
        (np.int64, finch.int64),
        (np.float64, finch.float64),
        (np.complex128, finch.complex128),
    ],
)
@pytest.mark.parametrize("order", ["C", "F", None])
def test_wrappers(dtype, jl_dtype, order):
    A = np.array([[0, 0, 4], [1, 0, 0], [2, 0, 5], [3, 0, 0]], dtype=dtype, order=order)
    B = np.array(np.stack([A, A], axis=2, dtype=dtype), order=order)

    B_finch = finch.Tensor(B)

    storage = finch.Storage(
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(dtype(0.0))))),
        order=order,
    )
    B_finch = B_finch.to_storage(storage)

    assert B_finch.shape == B.shape
    assert B_finch.dtype == jl_dtype
    assert_equal(B_finch.todense(), B)

    storage = finch.Storage(
        finch.Dense(finch.Dense(finch.Element(dtype(1.0)))), order=order
    )
    A_finch = finch.Tensor(A).to_storage(storage)

    assert A_finch.shape == A.shape
    assert A_finch.dtype == jl_dtype
    assert_equal(A_finch.todense(), A)
    assert A_finch.todense().dtype == A.dtype and B_finch.todense().dtype == B.dtype


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
@pytest.mark.parametrize("order", ["C", "F", None])
@pytest.mark.parametrize("copy", [True, False, None])
def test_copy_fully_dense(dtype, order, copy, arr3d):
    arr = np.array(arr3d, dtype=dtype, order=order)
    arr_finch = finch.Tensor(arr, copy=copy)
    arr_todense = arr_finch.todense()

    assert_equal(arr_todense, arr)
    if copy:
        assert not np.shares_memory(arr_todense, arr)
    else:
        assert np.shares_memory(arr_todense, arr)

def test_coo(rng):
    coords = (
        np.asarray([0, 1, 2, 3, 4], dtype=np.intp),
        np.asarray([0, 1, 2, 3, 4], dtype=np.intp),
    )
    data = rng.random(5)

    arr_pydata = sparse.COO(np.vstack(coords), data, shape=(5, 5))
    arr = arr_pydata.todense()
    arr_finch = finch.Tensor.construct_coo(coords, data, shape=(5, 5))

    assert_equal(arr_finch.todense(), arr)
    assert arr_finch.todense().dtype == data.dtype


@pytest.mark.parametrize(
    "classes",
    [
        (sparse._compressed.CSC, finch.Tensor.construct_csc),
        (sparse._compressed.CSR, finch.Tensor.construct_csr),
    ],
)
def test_compressed2d(rng, classes):
    sparse_class, finch_class = classes
    indices, indptr, data = np.arange(5), np.arange(6), rng.random(5)

    arr_pydata = sparse_class((data, indices, indptr), shape=(5, 5))
    arr = arr_pydata.todense()
    arr_finch = finch_class((data, indices, indptr), shape=(5, 5))

    assert_equal(arr_finch.todense(), arr)
    assert arr_finch.todense().dtype == data.dtype


def test_csf(arr3d):
    arr = arr3d
    dtype = np.int64

    data = np.array([4, 1, 2, 1, 1, 2, 5, -1, 3, 3], dtype=dtype)
    indices_list = [
        np.array([1, 0, 1, 2, 0, 1, 2, 1, 0, 2], dtype=dtype),
        np.array([0, 1, 0, 1, 0, 1], dtype=dtype),
    ]
    indptr_list = [
        np.array([0, 1, 4, 5, 7, 8, 10], dtype=dtype),
        np.array([0, 2, 4, 5, 6], dtype=dtype),
    ]

    arr_finch = finch.Tensor.construct_csf(
        (data, indices_list, indptr_list), shape=(3, 2, 4)
    )

    assert_equal(arr_finch.todense(), arr)
    assert arr_finch.todense().dtype == data.dtype


@pytest.mark.parametrize(
    "permutation", [(0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]
)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_permute_dims(arr3d, permutation, order, opt):
    finch.set_optimizer(opt)
    arr = np.array(arr3d, order=order)
    storage = finch.Storage(
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0)))), order=order
    )

    arr_finch = finch.Tensor(arr).to_storage(storage)

    actual_eager_mode = finch.permute_dims(arr_finch, permutation)
    actual_lazy_mode = finch.compute(
        finch.permute_dims(finch.lazy(arr_finch), permutation)
    )
    expected = np.transpose(arr, permutation)

    assert_equal(actual_eager_mode.todense(), expected)
    assert_equal(actual_lazy_mode.todense(), expected)

    actual_eager_mode = finch.permute_dims(actual_eager_mode, permutation)
    actual_lazy_mode = finch.compute(
        finch.permute_dims(finch.lazy(actual_lazy_mode), permutation)
    )
    expected = np.transpose(expected, permutation)

    assert_equal(actual_eager_mode.todense(), expected)
    assert_equal(actual_lazy_mode.todense(), expected)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_astype(arr3d, order, opt):
    finch.set_optimizer(opt)
    arr = np.array(arr3d, order=order, dtype=np.int64)
    storage = finch.Storage(
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(np.int64(0))))),
        order=order,
    )
    arr_finch = finch.Tensor(arr).to_storage(storage)

    result = finch.astype(arr_finch, finch.int64)
    assert not result is arr_finch
    result = result.todense()
    assert_equal(result, arr)
    assert result.dtype == arr.dtype

    result = finch.astype(arr_finch, finch.int64, copy=False)
    assert result is arr_finch
    result = result.todense()
    assert_equal(result, arr)
    assert result.dtype == arr.dtype

    result = finch.astype(arr_finch, finch.float32).todense()
    arr = arr.astype(np.float32)
    assert_equal(result, arr)
    assert result.dtype == arr.dtype

    with pytest.raises(
        ValueError, match="Unable to avoid a copy while casting in no-copy mode."
    ):
        finch.astype(arr_finch, finch.float64, copy=False)


@pytest.mark.parametrize("random_state", [42, np.random.default_rng(42)])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_random(random_state, opt):
    finch.set_optimizer(opt)
    result = finch.random((10, 20, 30), density=0.0, random_state=random_state)
    expected = sparse.random((10, 20, 30), density=0.0, random_state=random_state)

    assert_equal(result.todense(), expected.todense())

    # test reproducible runs
    run1 = finch.random((20, 20), density=0.8, random_state=0)
    run2 = finch.random((20, 20), density=0.8, random_state=0)
    run3 = finch.random((20, 20), density=0.8, random_state=0)
    assert_equal(run1.todense(), run2.todense())
    assert_equal(run1.todense(), run3.todense())


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("format", ["coo", "csr", "csc", "csf", "dense", None])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_asarray(arr2d, arr3d, order, format, opt):
    finch.set_optimizer(opt)
    arr = arr3d if format == "csf" else arr2d
    arr = np.array(arr, order=order)
    arr_finch = finch.Tensor(arr)

    result = finch.asarray(arr_finch, format=format)
    assert_equal(result.todense(), arr)


@pytest.mark.parametrize(
    "arr,new_shape",
    [
        (np.arange(10), (2, 5)),
        (np.ones((10, 10)), (100,)),
        (np.ones((3, 4, 5)), (5, 2, 2, 3)),
        (np.arange(1), (1, 1, 1, 1)),
        (np.zeros((10, 1, 2)), (1, 5, 4, 1)),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_reshape(arr, new_shape, order, opt):
    finch.set_optimizer(opt)
    arr = np.array(arr, order=order)
    arr_finch = finch.Tensor(arr)

    res = finch.reshape(arr_finch, new_shape)
    assert_equal(res.todense(), arr.reshape(new_shape))


@pytest.mark.parametrize("shape", [10, (3, 3), (2, 1, 5)])
@pytest.mark.parametrize("dtype_name", [None, "int64", "float64"])
@pytest.mark.parametrize("format", ["coo", "dense"])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_full_ones_zeros_empty(shape, dtype_name, format, opt):
    finch.set_optimizer(opt)
    jl_dtype = getattr(finch, dtype_name) if dtype_name is not None else None
    np_dtype = getattr(np, dtype_name) if dtype_name is not None else None

    res = finch.full(shape, 2.0, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.full(shape, 2.0, np_dtype))
    res = finch.full_like(res, 3.0, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.full(shape, 3.0, np_dtype))

    res = finch.ones(shape, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.ones(shape, np_dtype))
    res = finch.ones_like(res, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.ones(shape, np_dtype))

    res = finch.zeros(shape, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.zeros(shape, np_dtype))
    res = finch.zeros_like(res, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.zeros(shape, np_dtype))

    res = finch.empty(shape, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.empty(shape, np_dtype))
    res = finch.empty_like(res, dtype=jl_dtype, format=format)
    assert_equal(res.todense(), np.empty(shape, np_dtype))


@pytest.mark.parametrize("func,arg", [(finch.asarray, np.zeros(3)), (finch.zeros, 3)])
def test_device_keyword(func, arg):
    func(arg, device="cpu")

    with pytest.raises(
        ValueError, match="Device not understood. Only \"cpu\" is allowed, but received: cuda"
    ):
        func(arg, device="cuda")


@pytest.mark.parametrize(
    "order_and_format",
    [("C", None), ("F", None), ("C", "coo"), ("F", "coo"), ("F", "csc")],
)
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_where(order_and_format, opt):
    finch.set_optimizer(opt)
    order, format = order_and_format
    cond = np.array(
        [
            [True, False, False, False],
            [False, True, True, False],
            [True, False, True, True],
        ],
        order=order,
    )
    arr1 = np.array([[0, 0, 0, 1], [0, 2, 0, 3], [1, 0, 0, 5]], order=order)
    arr2 = np.array([10, 20, 30, 40], order=order)

    tns_cond = finch.asarray(cond, format=format)
    arr1_cond = finch.asarray(arr1, format=format)
    arr2_cond = finch.asarray(arr2)

    actual = finch.where(tns_cond, arr1_cond, arr2_cond)
    expected = np.where(cond, arr1, arr2)

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "format_shape",
    [
        ("coo", (80,)),
        ("coo", (10, 5, 8)),
        ("csf", (10, 5, 8)),
        ("csr", (5, 10)),
        ("csc", (5, 10)),
    ],
)
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_nonzero(order, format_shape, opt):
    finch.set_optimizer(opt)
    format, shape = format_shape
    rng = np.random.default_rng(0)
    arr = rng.random(shape)
    arr = np.array(arr, order=order)
    mask = arr < 0.8
    arr[mask] = 0.0

    tns = finch.asarray(arr, format=format)

    actual = finch.nonzero(tns)
    expected = np.nonzero(arr)
    for actual_i, expected_i in zip(actual, expected):
        assert_equal(actual_i.todense(), expected_i)


@pytest.mark.parametrize("dtype_name", ["int64", "float64", "complex128"])
@pytest.mark.parametrize("k", [0, -1, 1, -2, 2])
@pytest.mark.parametrize("format", ["coo", "dense"])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_eye(dtype_name, k, format, opt):
    finch.set_optimizer(opt)
    result = finch.eye(3, 4, k=k, dtype=getattr(finch, dtype_name), format=format)
    expected = np.eye(3, 4, k=k, dtype=getattr(np, dtype_name))

    assert_equal(result.todense(), expected)


@pytest.mark.parametrize("opt", ["default", "galley"])
def test_to_scalar(opt):
    finch.set_optimizer(opt)
    for obj, meth_name in [
        (True, "__bool__"), (1, "__int__"), (1.0, "__float__"), (1, "__index__"), (1+1j, "__complex__")
    ]:
        tns = finch.asarray(np.asarray(obj))
        assert getattr(tns, meth_name)() == obj

    tns = finch.asarray(np.ones((2, 2)))
    with pytest.raises(
        ValueError, match="<class 'int'> can be computed for one-element tensors only."
    ):
        tns.__int__()


@pytest.mark.parametrize("dtype_name", [None, "int16", "float64"])
@pytest.mark.parametrize("opt", ["default", "galley"])
def test_arange_linspace(dtype_name, opt):
    finch.set_optimizer(opt)
    if dtype_name is not None:
        finch_dtype = getattr(finch, dtype_name)
        np_dtype = getattr(np, dtype_name)
    else:
        finch_dtype = np_dtype = None

    result = finch.arange(10, 100, 5, dtype=finch_dtype)
    expected = np.arange(10, 100, 5, dtype=np_dtype)
    assert_equal(result.todense(), expected)

    result = finch.linspace(20, 80, 10, dtype=finch_dtype)
    expected = np.linspace(20, 80, 10, dtype=np_dtype)
    assert_equal(result.todense(), expected)
