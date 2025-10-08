import pytest

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_equal

import finch
from finch.tensor import _eq_scalars


def test_scipy_coo(arr2d):
    sp_arr = sp.coo_matrix(arr2d, dtype=np.int64)
    finch_arr = finch.Tensor(sp_arr)
    lvl = finch_arr._obj.body.lvl

    assert np.shares_memory(sp_arr.row, lvl.tbl[1].data)
    assert np.shares_memory(sp_arr.col, lvl.tbl[0].data)
    assert np.shares_memory(sp_arr.data, lvl.lvl.val)

    assert_equal(finch_arr.todense(), sp_arr.todense())
    new_arr = finch.permute_dims(finch_arr, (1, 0))
    assert_equal(new_arr.todense(), sp_arr.todense().transpose())


@pytest.mark.parametrize("cls", [sp.csc_matrix, sp.csr_matrix])
def test_scipy_compressed2d(arr2d, cls):
    sp_arr = cls(arr2d, dtype=np.int64)
    finch_arr = finch.Tensor(sp_arr)
    lvl = finch_arr._obj.body.lvl.lvl

    assert np.shares_memory(sp_arr.indices, lvl.idx.data)
    assert np.shares_memory(sp_arr.indptr, lvl.ptr.data)
    assert np.shares_memory(sp_arr.data, lvl.lvl.val)

    assert_equal(finch_arr.todense(), sp_arr.todense())
    new_arr = finch.permute_dims(finch_arr, (1, 0))
    assert_equal(new_arr.todense(), sp_arr.todense().transpose())


@pytest.mark.parametrize(
    "format_with_cls_with_order",
    [
        ("coo", sp.coo_matrix, "C"),
        ("coo", sp.coo_matrix, "F"),
        ("csc", sp.csc_matrix, "F"),
        ("csr", sp.csr_matrix, "C"),
    ],
)
@pytest.mark.parametrize("fill_value_in", [0, finch.inf, finch.nan, 5, None])
@pytest.mark.parametrize("fill_value_out", [0, finch.inf, finch.nan, 5, None])
def test_to_scipy_sparse(format_with_cls_with_order, fill_value_in, fill_value_out):
    format, sp_class, order = format_with_cls_with_order
    np_arr = np.random.default_rng(0).random((4, 5))
    np_arr = np.array(np_arr, order=order)

    finch_arr = finch.asarray(np_arr, format=format, fill_value=fill_value_in)

    if not (
        fill_value_in in {0, None} and fill_value_out in {0, None}
    ) and not _eq_scalars(fill_value_in, fill_value_out):
        match_fill_value_out = 0 if fill_value_out is None else fill_value_out
        with pytest.raises(
            ValueError,
            match=rf"Can only convert arrays with \[{match_fill_value_out}\] fill-values "
            "to a Scipy sparse matrix.",
        ):
            finch_arr.to_scipy_sparse(accept_fv=fill_value_out)
        return

    actual = finch_arr.to_scipy_sparse(accept_fv=fill_value_out)

    assert isinstance(actual, sp_class)
    assert_equal(actual.todense(), np_arr)


def test_to_scipy_sparse_invalid_input():
    finch_arr = finch.asarray(np.ones((3, 3, 3)), format="dense")

    with pytest.raises(ValueError, match="Can only convert a 2-dimensional array"):
        finch_arr.to_scipy_sparse()

    finch_arr = finch.asarray(np.ones((3, 4)), format="dense")

    with pytest.raises(
        ValueError, match="Tensor can't be converted to scipy.sparse object"
    ):
        finch_arr.to_scipy_sparse()


@pytest.mark.parametrize(
    "format_with_pattern",
    [
        ("coo", "SparseCOO"),
        ("csr", "SparseList"),
        ("csc", "SparseList"),
        ("bsr", "SparseCOO"),
        ("dok", "SparseCOO"),
    ],
)
@pytest.mark.parametrize("fill_value", [0, finch.inf, finch.nan, 5, None])
def test_from_scipy_sparse(format_with_pattern, fill_value):
    format, pattern = format_with_pattern
    sp_arr = sp.random(10, 5, density=0.1, format=format)

    result = finch.Tensor.from_scipy_sparse(sp_arr, fill_value=fill_value)
    assert pattern in str(result)
    fill_value = 0 if fill_value is None else fill_value
    assert _eq_scalars(result.fill_value, fill_value)


@pytest.mark.parametrize("format", ["coo", "bsr"])
def test_non_canonical_format(format):
    sp_arr = sp.random(3, 4, density=0.5, format=format)

    with pytest.raises(ValueError, match="Unable to avoid copy while creating an array"):
        finch.asarray(sp_arr, copy=False)

    finch_arr = finch.asarray(sp_arr)
    assert_equal(finch_arr.todense(), sp_arr.toarray())
