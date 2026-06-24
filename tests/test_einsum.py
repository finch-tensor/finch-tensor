import numpy as np

import finchlite

import finch
from finch import COMPILE_JULIA


def test_pass_through(rng):
    """Test pass through of a tensor"""
    A = rng.random((5, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B = finchlite.einop("B[i,j] = A[i,j]", A=A_finch)

    np.allclose(B.todense(), A)


def test_transpose(rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B = finchlite.einop("B[i,j] = A[j, i]", A=A_finch)

    np.allclose(B.todense(), A.T)


def test_basic_addition_with_transpose(rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))
    B = rng.random((5, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B_finch = finch.asarray(B)
    C = finchlite.einop("C[i,j] = A[i,j] + B[j,i]", A=A_finch, B=B_finch)
    C_ref = A + B.T

    np.allclose(C.todense(), C_ref)


def test_matrix_multiplication(rng):
    """Test matrix multiplication using += (increment/accumulation)"""
    A = rng.random((3, 4))
    B = rng.random((4, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B_finch = finch.asarray(B)
    C = finchlite.einop("C[i,j] += A[i,k] * B[k,j]", A=A_finch, B=B_finch)
    C_ref = A @ B

    np.allclose(C.todense(), C_ref)


def test_element_wise_multiplication(rng):
    """Test element-wise multiplication"""
    A = rng.random((4, 4))
    B = rng.random((4, 4))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B_finch = finch.asarray(B)
    C = finchlite.einop("C[i,j] = A[i,j] * B[i,j]", A=A_finch, B=B_finch)
    C_ref = A * B

    np.allclose(C.todense(), C_ref)


def test_sum_reduction(rng):
    """Test sum reduction using +="""
    A = rng.random((3, 4))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    C = finchlite.einop("C[i] += A[i,j]", A=A_finch)
    C_ref = np.sum(A, axis=1)

    np.allclose(C.todense(), C_ref)


def test_maximum_reduction(rng):
    """Test maximum reduction using max="""
    A = rng.random((3, 4))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    C = finchlite.einop("C[i] max= A[i,j]", A=A_finch)
    C_ref = np.max(A, axis=1)

    np.allclose(C.todense(), C_ref)


def test_outer_product(rng):
    """Test outer product"""
    A = rng.random(3)
    B = rng.random(4)

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B_finch = finch.asarray(B)
    C = finchlite.einop("C[i,j] = A[i] * B[j]", A=A_finch, B=B_finch)
    C_ref = np.outer(A, B)

    np.allclose(C.todense(), C_ref)


def test_batch_matrix_multiplication(rng):
    """Test batch matrix multiplication using +="""
    A = rng.random((2, 3, 4))
    B = rng.random((2, 4, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    B_finch = finch.asarray(B)
    C = finchlite.einop("C[b,i,j] += A[b,i,k] * B[b,k,j]", A=A_finch, B=B_finch)
    C_ref = np.matmul(A, B)

    np.allclose(C.todense(), C_ref)


def test_minimum_reduction(rng):
    """Test minimum reduction using min="""
    A = rng.random((3, 4))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = finch.asarray(A)
    C = finchlite.einop("C[i] min= A[i,j]", A=A_finch)
    C_ref = np.min(A, axis=1)

    np.allclose(C.todense(), C_ref)
