import numpy as np

import finchlite
from juliacall import Main as jl

from finch import COMPILE_JULIA, FinchJLTensor


def test_pass_through(rng):
    """Test pass through of a tensor"""
    A = rng.random((5, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(0.0))), A))
    B = finchlite.einop("B[i,j] = A[i,j]", A=A_finch)

    np.allclose(B.todense(), A)


def test_transpose(rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))

    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A_finch = FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(0.0))), A))
    B = finchlite.einop("B[i,j] = A[j, i]", A=A_finch)

    np.allclose(B.todense(), A.T)
