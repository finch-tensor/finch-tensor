import numpy as np

import finchlite
from juliacall import Main as jl

from finch import COMPILE_JULIA, FinchJLTensor


def test_pass_through(rng):
    """Test pass through of a tensor"""
    finchlite.interface.set_default_scheduler(ctx=COMPILE_JULIA)
    A = FinchJLTensor(
        jl.Finch.Tensor(
            jl.Dense(jl.Dense(jl.Element(0.0))), np.array(rng.random((5, 5)))
        )
    )
    B = finchlite.einop("B[i,j] = A[i,j]", A=A)
    np.allclose(B.todense(), A.todense())
