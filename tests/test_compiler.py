import operator

import pytest

import numpy as np

from finchlite.compile import ExtentFType, dimension
from finchlite.finch_notation.nodes import (
    Access,
    Assign,
    Block,
    Call,
    Declare,
    Freeze,
    Function,
    Increment,
    Literal,
    Loop,
    Module,
    Read,
    Repack,
    Return,
    Slot,
    Unpack,
    Unwrap,
    Update,
    Variable,
)

from finch.compiler import FinchJLCompiler, FinchJLKernel
from finch.julia import jl
from finch.tensor import Dense, Element, FinchJLTensor

a_format = Dense(Dense(Element(0)))


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
                                                        Call(
                                                            Literal(operator.mul),
                                                            (
                                                                Unwrap(
                                                                    Access(
                                                                        Slot(
                                                                            "A_",
                                                                            a_format,
                                                                        ),
                                                                        Read(),
                                                                        (
                                                                            Variable(
                                                                                "i",
                                                                                np.int64,
                                                                            ),
                                                                            Variable(
                                                                                "k",
                                                                                np.int64,
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                Unwrap(
                                                                    Access(
                                                                        Slot(
                                                                            "B_",
                                                                            a_format,
                                                                        ),
                                                                        Read(),
                                                                        (
                                                                            Variable(
                                                                                "k",
                                                                                np.int64,
                                                                            ),
                                                                            Variable(
                                                                                "j",
                                                                                np.int64,
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                Freeze(Slot("C_", a_format), Literal(operator.add)),
                                Repack(Slot("C_", a_format), Variable("C", a_format)),
                                Return(Variable("C", a_format)),
                            ),
                        ),
                    ),
                )
            ),
            """function matmul(C,A,B)
    @finch C .= 0.0
    @finch begin
        for i = _
            for k = _
                for j = _
                    C[i,j] += *(A[i,k],B[k,j])
                end
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


@pytest.mark.parametrize(
    "func_name, julia_prgm, args, expected_result",
    [
        (
            "matmul",
            """function matmul(C,A,B)
    @finch C .= 0
    @finch begin
        for i = _
            for k = _
                for j = _
                    C[i,j] += *(A[i,k],B[k,j])
                end
            end
        end
    end
    return C
end""",
            (
                FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(0))), 3, 3)),
                FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(1))), 3, 3)),
                FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(2))), 3, 3)),
            ),
            (FinchJLTensor(jl.Finch.Tensor(jl.Dense(jl.Dense(jl.Element(6))), 3, 3)),),
        )
    ],
)
def test_finchjl_kernel(
    func_name: str,
    julia_prgm: str,
    args: tuple[FinchJLTensor, ...],
    expected_result: tuple[FinchJLTensor, ...],
):
    kernel = FinchJLKernel(func_name, julia_prgm)
    result = kernel(*args)
    assert result == expected_result
