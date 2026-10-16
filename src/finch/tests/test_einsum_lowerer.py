import pytest

import numpy as np

import finch
from finch.algebra import ffuncs
from finch.autoschedule.einsum import LogicEinsumLoader
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.factorizer.optimize import DefaultLogicFactorizer
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer.loop_ordering import DefaultLoopOrderer
from finch.finch_einsum import MockEinsumLoader
from finch.interface.fuse import compute
from finch.interface.lazy import defer, reduce

from .conftest import finch_assert_allclose


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def ctx():
    return LogicExecutor(
        DefaultLogicFactorizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicEinsumLoader(ctx_load=MockEinsumLoader()))
            )
        )
    )


@pytest.mark.parametrize(
    "op, init, np_op",
    [(ffuncs.add, 5.0, np.add), (ffuncs.max, 0.5, np.maximum)],
    ids=["add", "max"],
)
def test_reduction_starts_from_its_init(rng, ctx, op, init, np_op):
    # An init which isn't the identity of the reduction is an initializing
    # einsum, which the reduction then folds into.
    A = rng.random((4, 3))
    result = compute(reduce(op, defer(finch.asarray(A)), axis=1, init=init), ctx=ctx)
    finch_assert_allclose(result, np_op(init, np_op.reduce(A, axis=1)))


def test_simple_addition(rng, ctx):
    """Test lowering of simple addition A + B"""
    A = defer(rng.random((3, 3)))
    B = defer(rng.random((3, 3)))

    C = finch.add(A, B)

    # Execute the plan
    result = compute(C, ctx=ctx)

    # Compare with expected
    expected = compute(A + B)
    finch_assert_allclose(result, expected)


def test_scalar_multiplication(rng, ctx):
    """Test lowering of scalar multiplication 2 * A"""
    A = defer(rng.random((4, 4)))

    B = finch.multiply(2, A)

    result = compute(B, ctx=ctx)

    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_element_wise_operations(rng, ctx):
    """Test lowering of element-wise operations"""
    A = defer(rng.random((3, 3)))
    B = defer(rng.random((3, 3)))
    C = defer(rng.random((3, 3)))

    D = finch.add(finch.multiply(A, B), C)

    result = compute(D, ctx=ctx)

    expected = compute(D)
    finch_assert_allclose(result, expected)


def test_sum_reduction(rng, ctx):
    """Test sum reduction using +="""
    A = defer(rng.random((3, 4)))

    B = finch.sum(A, axis=1)

    result = compute(B, ctx=ctx)

    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_maximum_reduction(rng, ctx):
    """Test maximum reduction using max="""
    A = defer(rng.random((3, 4)))

    B = finch.max(A, axis=1)

    result = compute(B, ctx=ctx)
    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_batch_matrix_multiplication(rng, ctx):
    """Test batch matrix multiplication using +="""
    A = defer(rng.random((2, 3, 4)))
    B = defer(rng.random((2, 4, 5)))

    C = finch.matmul(A, B)

    result = compute(C, ctx=ctx)
    expected = compute(C)
    finch_assert_allclose(result, expected)


def test_minimum_reduction(rng, ctx):
    """Test minimum reduction using min="""
    A = defer(rng.random((3, 4)))

    B = finch.min(A, axis=1)

    result = compute(B, ctx=ctx)
    expected = compute(B)
    finch_assert_allclose(result, expected)
