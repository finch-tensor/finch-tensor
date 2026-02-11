import pytest
import numpy as np
from finchlite.finch_notation.nodes import (
    Module,
    Function,
    Variable,
    Block,
    Assign,
    Call,
    Literal,
    Update,
    Unpack,
    Slot,
    Loop,
    Increment,
    Access,
    Unwrap,
    Freeze,
    Read,
    Repack,
    Return,
    Declare,
)

import operator
from finchlite import ftype
from finchlite.algebra import overwrite, promote_min
from finchlite.compile import ExtentFType, dimension, BufferizedNDArray
from finchlite.codegen import NumpyBuffer

from finch.compiler import FinchJLCompiler

# Dummy data to obtain the bufferized ND array type
a = np.zeros(dtype=np.float64, shape=(3, 3))
a_format = ftype(BufferizedNDArray.from_numpy(a))

@pytest.mark.parametrize(
    "finch_ntn, julia_code",
    [
        (
            Module(
                (
                    Function(
                        Variable("matmul", a_format),
                        (
                            Variable("C", a_format),
                            Variable("A", a_format),
                            Variable("B", a_format),
                        ),
                        Block(
                            (
                                Assign(
                                    Variable("m", ExtentFType(np.int64, np.int64)),
                                    Call(
                                        Literal(dimension),
                                        (Variable("A", a_format), Literal(0)),
                                    ),
                                ),
                                Assign(
                                    Variable("n", ExtentFType(np.int64, np.int64)),
                                    Call(
                                        Literal(dimension),
                                        (Variable("B", a_format), Literal(1)),
                                    ),
                                ),
                                Assign(
                                    Variable("p", ExtentFType(np.int64, np.int64)),
                                    Call(
                                        Literal(dimension),
                                        (Variable("A", a_format), Literal(1)),
                                    ),
                                ),
                                Unpack(Slot("A_", a_format), Variable("A", a_format)),
                                Unpack(Slot("B_", a_format), Variable("B", a_format)),
                                Unpack(Slot("C_", a_format), Variable("C", a_format)),
                                Declare(
                                    Slot("C_", a_format),
                                    Literal(0.0),
                                    Literal(operator.add),
                                    (
                                        Variable("m", ExtentFType(np.int64, np.int64)),
                                        Variable("n", ExtentFType(np.int64, np.int64)),
                                    ),
                                ),
                                Loop(
                                    Variable("i", np.int64),
                                    Variable("m", ExtentFType(np.int64, np.int64)),
                                    Loop(
                                        Variable("k", np.int64),
                                        Variable("p", ExtentFType(np.int64, np.int64)),
                                        Loop(
                                            Variable("j", np.int64),
                                            Variable(
                                                "n", ExtentFType(np.int64, np.int64)
                                            ),
                                            Block(
                                                (
                                                    Assign(
                                                        Variable("a_ik", np.float64),
                                                        Unwrap(
                                                            Access(
                                                                Slot("A_", a_format),
                                                                Read(),
                                                                (
                                                                    Variable(
                                                                        "i", np.int64
                                                                    ),
                                                                    Variable(
                                                                        "k", np.int64
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                    ),
                                                    Assign(
                                                        Variable("b_kj", np.float64),
                                                        Unwrap(
                                                            Access(
                                                                Slot("B_", a_format),
                                                                Read(),
                                                                (
                                                                    Variable(
                                                                        "k", np.int64
                                                                    ),
                                                                    Variable(
                                                                        "j", np.int64
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                    ),
                                                    Assign(
                                                        Variable("c_ij", np.float64),
                                                        Call(
                                                            Literal(operator.mul),
                                                            (
                                                                Variable(
                                                                    "a_ik", np.float64
                                                                ),
                                                                Variable(
                                                                    "b_kj", np.float64
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                    Increment(
                                                        Access(
                                                            Slot("C_", a_format),
                                                            Update(
                                                                Literal(operator.add)
                                                            ),
                                                            (
                                                                Variable("i", np.int64),
                                                                Variable("j", np.int64),
                                                            ),
                                                        ),
                                                        Variable("c_ij", np.float64),
                                                    ),
                                                )
                                            ),
                                        ),
                                    ),
                                ),
                                Freeze(Slot("C_", a_format), Literal(operator.add)),
                                Repack(Slot("C_", a_format), Variable("C", a_format)),
                                Return(Variable("C", a_format)),
                            )
                        ),
                    ),
                )
            ),
            """function matmul(C,A,B)
    C .= 0
    for i = _
        for k = _
            for j = _
                a_ik = A[i,k]
                b_kj = B[k,j]
                c_ij = a_ik * b_kj
                C[i,j] = c_ij
            end
        end
    end
    return C
end""",
        )
    ],
)
def test_finchjl_compiler(finch_ntn: Module, julia_code):
    compiler = FinchJLCompiler()
    library = compiler(finch_ntn)
    assert getattr(library, finch_ntn.children[0].name.name).jl_code == julia_code
