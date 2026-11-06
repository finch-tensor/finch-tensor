"""Regressions found running the Sparse Array Programming Suite (SAPS) through
finch with the Julia backend and the Galley scheduler, as SAPS's adapter does.
"""

import pytest

import numpy as np

import finch as ft
from finch.algebra import cansplitpush, ffuncs, repeat_operator
from finch.autoschedule import COMPILE_JULIA_GALLEY, with_default_scheduler
from finch.compile_jl.julia import julia_available

pytestmark = pytest.mark.skipif(
    not julia_available(),
    reason="the julia extra (juliapkg, juliacall) is not installed",
)


@pytest.fixture(autouse=True)
def _galley_scheduler():
    with with_default_scheduler(COMPILE_JULIA_GALLEY):
        yield


def _dense(x):
    return np.asarray(ft.to_numpy(ft.asarray(x)))


def test_index_julia_owned_result_with_dynamic_fill():
    # cg: `xp.sqrt(xp.vecdot(b, b))[()]`. The result's fill is Dynamic, so its
    # ftype has no value, but the Julia storage holds one.
    b = ft.asarray(np.array([3.0, 4.0]))
    norm = ft.sqrt(ft.vecdot(b, b))
    assert float(norm[()]) == 5.0


def test_reshape_tensor_without_its_own_reshape():
    # rqc_statevector, tictactoe_minimax: FiberTensors and Julia-owned results
    # have no reshape method, so eager reshape must go through lazy reshape.
    x = ft.asarray(np.arange(6.0)) + 1.0
    np.testing.assert_array_equal(
        _dense(ft.reshape(x, (2, 3))), np.arange(1.0, 7.0).reshape(2, 3)
    )


def test_einsum_logical_not():
    # model_counting: `not` is unary, but the einsum interpreter only listed
    # logical_not among the n-ary operators.
    x = ft.asarray(np.array([True, False, True]))
    y = ft.einop("y[i] = not x[i]", x=x)
    np.testing.assert_array_equal(_dense(y), [False, True, False])


def test_reduction_without_repeat_operator():
    # fastsv, gmres: choose (reshape) and add_scaled_square (vector_norm) have
    # no repeat operator, and looking one up raised instead of returning None.
    assert repeat_operator(ffuncs.add_scaled_square.ftype) is None
    assert repeat_operator(ffuncs.choose(1.0).ftype) is None
    assert not cansplitpush(ffuncs.add_scaled_square.ftype)
    x = ft.asarray(np.arange(6.0)) + 1.0
    y = ft.reshape(ft.reshape(x, (2, 3)), (3, 2))
    np.testing.assert_array_equal(_dense(y), np.arange(1.0, 7.0).reshape(3, 2))


@pytest.mark.parametrize("ord", [2, 3, 0.5, -1, -2])
def test_vector_norm(ord):
    # fastsv, gmres: vector_norm sums tuple-valued scaled powers, which the
    # Julia backend had no operators for.
    v = np.array([3.0, 4.0])
    assert float(ft.linalg.vector_norm(ft.asarray(v), ord=ord)[()]) == pytest.approx(
        np.linalg.vector_norm(v, ord=ord)
    )
    x = np.array([[0.0, 3.0, -4.0], [1.0, 0.5, 2.0]])
    np.testing.assert_allclose(
        _dense(ft.linalg.vector_norm(ft.asarray(x), ord=ord, axis=1)),
        np.linalg.vector_norm(x, ord=ord, axis=1),
    )
    # The sums are scaled, so powers of large values don't overflow as NumPy's do.
    big = np.array([3e200, 4e200])
    assert float(ft.linalg.vector_norm(ft.asarray(big), ord=ord)[()]) == pytest.approx(
        np.linalg.vector_norm(big / 1e200, ord=ord) * 1e200
    )


def test_logical_operators_on_integers():
    # NumPy's logical operators read their operands' truth; Julia's !, &&
    # and || take only Bools, and & and | act bitwise on integers.
    x = np.array([0, 1, 2, 3])
    y = np.array([2, 0, 1, 4])
    np.testing.assert_array_equal(
        _dense(ft.logical_not(ft.asarray(x))), np.logical_not(x)
    )
    np.testing.assert_array_equal(
        _dense(ft.logical_and(ft.asarray(x), ft.asarray(y))), np.logical_and(x, y)
    )
    np.testing.assert_array_equal(
        _dense(ft.logical_or(ft.asarray(x), ft.asarray(y))), np.logical_or(x, y)
    )
    z = np.array([[0, 2, 0], [3, 1, 0]])
    np.testing.assert_array_equal(
        _dense(ft.reduce(ffuncs.logical_or, ft.asarray(z), axis=1, init=False)),
        np.any(z, axis=1),
    )
    np.testing.assert_array_equal(
        _dense(ft.reduce(ffuncs.logical_and, ft.asarray(z), axis=1, init=True)),
        np.all(z, axis=1),
    )


def test_logical_operators_take_many_operands():
    # The Galley statistics apply reductions' operators to several fills.
    assert ffuncs.logical_and(True, True, False) == np.False_
    assert ffuncs.logical_or(False, False, True) == np.True_


@pytest.mark.xfail(
    strict=True,
    reason=(
        "handle_fills pins Dynamic fills to zero, but an output's declared init "
        "is then op(0, 0), which Finch.jl requires to equal the output storage's "
        "pinned fill, zero; it does not for >=, ==, exp, cos, ... The pinned "
        "storage should take the init as its fill (pin_fill as a value)."
    ),
)
def test_compare_results_with_dynamic_fills():
    # cg, jacobi, pcg: `if rr >= tol_sq:` on two 0-d results.
    b = ft.asarray(np.array([1.0, 2.0, 3.0]))
    rr = ft.vecdot(b, b)
    tol = ft.asarray(np.float64(1e-3)) * ft.vecdot(b, b)
    assert bool(ft.greater_equal(rr, tol))


def test_complex_fill_literal():
    # rqc_statevector: Python writes complex fills as `0j`, which Julia read as
    # the name `j` (UndefVarError).
    x = ft.asarray(np.array([1 + 1j, 2 - 1j, 0j]))
    np.testing.assert_array_equal(_dense(x * (1 + 0j)), [1 + 1j, 2 - 1j, 0j])


def test_signed_zero_fills_are_different_ftypes():
    # ccsd: Element{0.0} and Element{-0.0} are different Julia types, but
    # `same` equated 0.0 and -0.0, so a kernel cached for one rejected the other.
    from finch.algebra.fill import StaticFillFType

    assert StaticFillFType(np.float64(0.0)) != StaticFillFType(np.float64(-0.0))
    assert StaticFillFType(np.float64(np.nan)) == StaticFillFType(np.float64(np.nan))
    x = ft.asarray(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(_dense(-(x * 0.0) + x), [1.0, 2.0])


def test_eye():
    # lp_simplex: computing eye is a query that returns its pattern tensor,
    # which a Julia kernel cannot return.
    a = np.arange(6.0).reshape(3, 2)
    out = ft.concat([ft.asarray(a), ft.eye(3)], axis=1)
    np.testing.assert_array_equal(_dense(out), np.concatenate([a, np.eye(3)], axis=1))
