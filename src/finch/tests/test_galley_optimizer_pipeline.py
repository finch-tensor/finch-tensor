"""
Galley optimizer pipeline tests.
"""

import pytest

import numpy as np

import finch.interface as fl_interface
from finch.autoschedule import INTERPRET_NOTATION_GALLEY


# --- TEST 1: out = a * b via frontend ---
def test_elementwise_mul():
    """Running out = a * b with Finch/Galley pipeline using the frontend."""
    a = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.lazy(a) * fl_interface.lazy(b),
        ctx=INTERPRET_NOTATION_GALLEY,
    )
    expected = np.array([[1.0, 2.0], [3.0, 4.0]]) * np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 2: out = a * b + c * d via frontend ---
def test_add_of_elementwise():
    """Running out = a * b + c * d with Finch/Galley frontend."""
    a = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    c = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    d = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.lazy(a) * fl_interface.lazy(b)
        + fl_interface.lazy(c) * fl_interface.lazy(d),
        ctx=INTERPRET_NOTATION_GALLEY,
    )
    expected = np.array(a) * np.array(b) + np.array(c) * np.array(d)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 3: sum(A @ B, axis=0) via frontend ---
def test_matmul_sum_axis0():
    """
    Running out = sum_i sum_j (A[i,j] * B[j,k]) with Finch/Galley frontend.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0),
        ctx=INTERPRET_NOTATION_GALLEY,
    )
    expected = np.sum(np.array(A) @ np.array(B), axis=0)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 4: sum(A@B, axis=0) + sum(C@D, axis=1) ---
def test_sum_axis0_plus_sum_axis1():
    """
    Running out = sum(A @ B, axis=0) + sum(C @ D, axis=1). Correctness vs NumPy.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    left = np.sum(np.array(A) @ np.array(B), axis=0)
    right = np.sum(np.array(C) @ np.array(D), axis=1)
    expected = left + right
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 5: Nested aggregates sum(A @ B) ---
def test_nested_aggregates_full_sum():
    """
    Nested aggregates: out = sum_i sum_j (A[i,j] * B[j,k]).
    Verified against NumPy.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B)),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.array(A) @ np.array(B))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 6: sum((A @ B) @ C) ---
def test_deeper_nesting():
    """Deeper nesting: out = sum((A @ B) @ C).
    Verified against NumPy."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(
            (fl_interface.lazy(A) @ fl_interface.lazy(B)) @ fl_interface.lazy(C)
        ),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum((np.array(A) @ np.array(B)) @ np.array(C))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 7: expand_dims + sum over singleton (extra axis padding) ---
def test_expand_dims_sum_singleton():
    """
    Exercises extra axis padding in logic_to_stats: sum(expand_dims(A, 2), axis=2).
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

    expanded = fl_interface.expand_dims(fl_interface.lazy(A), axis=2)
    out = fl_interface.compute(
        fl_interface.sum(expanded, axis=2),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.expand_dims(np.array(A), axis=2), axis=2)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 8: Alias Matmul ---
def test_alias_matmul():
    """
    Exercises alias-based query merging and reordering.
    """
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = fl_interface.lazy(fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]])))
    B = A @ A
    C = B @ B @ B
    out = fl_interface.compute(
        C,
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = A_np @ A_np @ A_np @ A_np @ A_np @ A_np
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 9: Performance optimization (cost-based reduction order) ---
def test_galley_performance_optimization_chain_matmul():
    """
    Test Galley performing a real performance optimization based on
    input data. For a chain matmul A @ B @ C, the greedy optimizer chooses
    reduction order by cost. With asymmetric shapes, one order is
    cheaper than the other; Galley should pick the cheaper one.

    A(2,10) @ B(10,3) @ C(3,4) -> (2,4)
    - Reducing middle (j) first: 2*10*3 + 2*3*4 = 60 + 24 = 84 computations
    - Reducing k first: 10*3*4 + 2*10*4 = 120 + 80 = 200 computations
    """
    A = fl_interface.asarray(np.arange(2 * 10, dtype=float).reshape(2, 10))
    B = fl_interface.asarray(np.arange(10 * 3, dtype=float).reshape(10, 3))
    C = fl_interface.asarray(np.arange(3 * 4, dtype=float).reshape(3, 4))

    out = fl_interface.compute(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 10: Chain matmul A(10,2) @ B(2,10) @ C(10,2) -> (10,2) ---
def test_galley_chain_matmul_10_2_2_10_10_2():
    """
    Chain matmul A(10,2) @ B(2,10) @ C(10,2) -> (10,2)
    - (A @ B) @ C: (10,2)@(2,10)@(10,2) -> contracts 2, then 10
    - A @ (B @ C): (10,2)@(2,10)@(10,2) -> contracts 10, then 2
    """
    A = fl_interface.asarray(np.arange(1000 * 2, dtype=float).reshape(1000, 2))
    B = fl_interface.asarray(np.arange(2 * 1000, dtype=float).reshape(2, 1000))
    C = fl_interface.asarray(np.arange(1000 * 2, dtype=float).reshape(1000, 2))

    out = fl_interface.compute(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 11: Alias matmul with different base tensors ---
def test_alias_matmul_two_bases():
    """
    Exercises alias-based merging when operands come from different base tensors.
    B = A1 @ A2, C = B @ B.
    """
    A1_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    A2_np = np.array([[1.0, 0.0], [0.0, 1.0]])
    A1 = fl_interface.lazy(fl_interface.asarray(A1_np))
    A2 = fl_interface.lazy(fl_interface.asarray(A2_np))
    B = A1 @ A2
    C = B @ B
    out = fl_interface.compute(C, ctx=INTERPRET_NOTATION_GALLEY)

    expected = (A1_np @ A2_np) @ (A1_np @ A2_np)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 12: Chain matmul A(5,4) @ B(4,6) @ C(6,3) -> (5,3) ---
def test_galley_chain_matmul_5_4_4_6_6_3():
    """
    Chain matmul A(5,4) @ B(4,6) @ C(6,3) -> (5,3)
    - (A @ B) @ C: 5*4*6 + 5*6*3 = 120 + 90 = 210
    - A @ (B @ C): 4*6*3 + 5*4*3 = 72 + 60 = 132
    """
    A = fl_interface.asarray(np.arange(5 * 4, dtype=float).reshape(5, 4))
    B = fl_interface.asarray(np.arange(4 * 6, dtype=float).reshape(4, 6))
    C = fl_interface.asarray(np.arange(6 * 3, dtype=float).reshape(6, 3))

    out = fl_interface.compute(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 13: Chain matmul A(3,5) @ B(5,2) @ C(2,2) -> (3,2) ---
def test_galley_chain_matmul_3_5_5_2_2_2():
    """
    Chain matmul A(3,5) @ B(5,2) @ C(2,2) -> (3,2)
    - (A @ B) @ C: 3*5*2 + 3*2*2 = 30 + 12 = 42
    - A @ (B @ C): 5*2*2 + 3*5*2 = 20 + 30 = 50
    """
    A = fl_interface.asarray(np.arange(3 * 5, dtype=float).reshape(3, 5))
    B = fl_interface.asarray(np.arange(5 * 2, dtype=float).reshape(5, 2))
    C = fl_interface.asarray(np.arange(2 * 2, dtype=float).reshape(2, 2))

    out = fl_interface.compute(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 14: Longer chain matmul A @ B @ C @ D (4 matrices) ---
def test_galley_chain_matmul_four_matrices():
    """
    Longer chain matmul A(6,5) @ B(5,4) @ C(4,3) @ D(3,2) -> (6,2).
    Exercises Galley on a 4-way chain with more reduction choices.
    """
    A = fl_interface.asarray(np.arange(6 * 5, dtype=float).reshape(6, 5))
    B = fl_interface.asarray(np.arange(5 * 4, dtype=float).reshape(5, 4))
    C = fl_interface.asarray(np.arange(4 * 3, dtype=float).reshape(4, 3))
    D = fl_interface.asarray(np.arange(3 * 2, dtype=float).reshape(3, 2))

    out = fl_interface.compute(
        fl_interface.lazy(A)
        @ fl_interface.lazy(B)
        @ fl_interface.lazy(C)
        @ fl_interface.lazy(D),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C) @ np.array(D)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 15: Longer alias chain B @ B @ B @ B (4 matmuls) ---
def test_alias_matmul_longer_chain():
    """
    Longer alias chain: B = A @ A, C = B @ B @ B @ B.
    Exercises alias merging with more inlined copies (8 uses of A).
    """
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = fl_interface.lazy(fl_interface.asarray(A_np))
    B = A @ A
    C = B @ B @ B @ B
    out = fl_interface.compute(C, ctx=INTERPRET_NOTATION_GALLEY)

    expected = A_np @ A_np @ A_np @ A_np @ A_np @ A_np @ A_np @ A_np
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 16: sum(A * B) - elementwise multiply then full sum ---
def test_sum_elementwise_mul():
    """
    out = sum(A * B) - Frobenius inner product.
    Combines elementwise multiplication and sum.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) * fl_interface.lazy(B)),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.array(A) * np.array(B))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 17: sum(A * B, axis=1) - elementwise mul then sum over axis ---
def test_sum_elementwise_mul_axis1():
    """
    out = sum(A * B, axis=1) - elementwise multiply then sum over columns.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 2.0], [1.0, 2.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) * fl_interface.lazy(B), axis=1),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.array(A) * np.array(B), axis=1)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 18: (A @ B) + (C @ D) - sum of two matmuls ---
def test_matmul_plus_matmul():
    """
    out = (A @ B) + (C @ D) - sum of two matrix products.
    Combines matmul and elementwise addition.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))

    out = fl_interface.compute(
        (fl_interface.lazy(A) @ fl_interface.lazy(B))
        + (fl_interface.lazy(C) @ fl_interface.lazy(D)),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = (np.array(A) @ np.array(B)) + (np.array(C) @ np.array(D))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 18: (A @ B) + (C @ D) - sum of two matmuls ---
def test_multiple_compute():
    """
    out = (A @ B) + (C @ D) - sum of two matrix products.
    Combines matmul and elementwise addition.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))

    out = fl_interface.compute(
        (
            fl_interface.lazy(A) @ fl_interface.lazy(B),
            fl_interface.lazy(C) @ fl_interface.lazy(D),
        ),
        ctx=INTERPRET_NOTATION_GALLEY,
    )

    expected = ((np.array(A) @ np.array(B)), (np.array(C) @ np.array(D)))
    assert np.allclose(np.array(out[0]), np.array(expected[0]))
    assert np.allclose(np.array(out[1]), np.array(expected[1]))


# --- Scalar constants reduced via a repeat operator ---
@pytest.mark.parametrize("scalar", [2, 2.0, 3, 3.0, 6.0])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_repeat_operator_scalar_literal(scalar, axis):
    """
    Reducing an expression whose scalar operand is a bare `Literal` sends galley
    down the repeat-operator path (sum_i c = c*|Dom(i)|, prod_i c = c**|Dom(i)|).

    The domain size is itself a Literal, and literals compare by value, so the
    rewrite has to be applied in one pass with one combined factor. Rewriting
    one reduction index at a time let the next index match inside the subtree
    just inserted and square the result: prod(x*2) over a 2x3 tensor gave 2**24
    instead of 2**6. `scalar=6.0` is the case where the constant equals the
    combined factor exactly.
    """
    arr = np.arange(6.0).reshape(2, 3) + 1
    x = fl_interface.asarray(arr)

    for reduce_fn, np_fn in (
        (fl_interface.sum, np.sum),
        (fl_interface.prod, np.prod),
    ):
        for expr, np_expr in (
            (fl_interface.lazy(x) * scalar, arr * scalar),
            (fl_interface.lazy(x) + scalar, arr + scalar),
        ):
            out = fl_interface.compute(
                reduce_fn(expr, axis=axis), ctx=INTERPRET_NOTATION_GALLEY
            )
            assert np.allclose(np.array(out), np_fn(np_expr, axis=axis))


def test_repeat_operator_repeated_scalar_literal():
    """The same constant twice under one reduction: each occurrence is reduced
    over, so each takes the combined factor."""
    arr = np.arange(6.0).reshape(2, 3) + 1
    x = fl_interface.lazy(fl_interface.asarray(arr))

    out = fl_interface.compute(
        fl_interface.prod(x * 2.0 * 2.0), ctx=INTERPRET_NOTATION_GALLEY
    )
    assert np.allclose(np.array(out), (arr * 2.0 * 2.0).prod())

    out = fl_interface.compute(
        fl_interface.sum(x + 2.0 + 2.0), ctx=INTERPRET_NOTATION_GALLEY
    )
    assert np.allclose(np.array(out), (arr + 2.0 + 2.0).sum())


def test_repeat_operator_scalar_outside_reduction():
    """
    A constant appearing both inside and outside the reduction: the two
    occurrences are structurally equal, but only the inner one is reduced
    over. Galley's rewrites address occurrences by path, so the repeat factor
    applies to the inner occurrence only.
    """
    arr = np.arange(6.0).reshape(2, 3) + 1
    x = fl_interface.lazy(fl_interface.asarray(arr))

    out = fl_interface.compute(
        fl_interface.prod(x * 2.0) * 2.0, ctx=INTERPRET_NOTATION_GALLEY
    )
    assert np.allclose(np.array(out), (arr * 2.0).prod() * 2.0)

    out = fl_interface.compute(
        fl_interface.sum(x + 2.0) + 2.0, ctx=INTERPRET_NOTATION_GALLEY
    )
    assert np.allclose(np.array(out), (arr + 2.0).sum() + 2.0)

    out = fl_interface.compute(
        fl_interface.prod(x * 6.0) * 6.0, ctx=INTERPRET_NOTATION_GALLEY
    )
    assert np.allclose(np.array(out), (arr * 6.0).prod() * 6.0)


def test_mapjoin_strict_binary_partitioned_fills():
    """
    Regression: a strict-binary assoc/comm op (logical_and) whose argument
    fills split into annihilator (False) and non-annihilator (True) partitions
    must not union the single non-annihilator side alone — op(single_fill) is
    an arity error for binary operators.
    """
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask_arr = np.array([[True, False], [False, True]])
    x = fl_interface.lazy(fl_interface.asarray(arr))
    mask = fl_interface.lazy(fl_interface.asarray(mask_arr))

    out = fl_interface.compute(
        fl_interface.logical_and(x <= 2, mask),
        ctx=INTERPRET_NOTATION_GALLEY,
    )
    expected = np.logical_and(arr <= 2, mask_arr)
    assert np.array_equal(np.array(out), expected)
