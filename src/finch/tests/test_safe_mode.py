import subprocess
import sys

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype
from finch.codegen import NumpyBuffer
from finch.codegen.c_codegen import CCompiler
from finch.codegen.mlir_codegen import MLIRGenerator
from finch.codegen.numba_codegen import NumbaCompiler
from finch.compile import AssemblyContext, CompilerMode, NotationCompiler, make_extent
from finch.compile.lower import AssemblyGenerator
from finch.symbolic import PostOrderDFS
from finch.tensor import (
    BufferizedNDArray,
    ElementLevel,
    FiberTensor,
    SparseListLevel,
    dense,
    element,
    fiber_tensor,
)


def sum_program(tensor_type):
    tensor = ntn.Variable("tensor", tensor_type)
    slot = ntn.Slot("tensor_slot", tensor_type)
    start = ntn.Variable("start", ftype(np.intp))
    end = ntn.Variable("end", ftype(np.intp))
    idx = ntn.Variable("i", ftype(np.intp))
    result = ntn.Variable("result", tensor_type.element_type)
    args = (tensor, start, end)
    function = ntn.Function(
        ntn.Variable(
            "sum_range",
            asm.AssemblyKernelFType(
                "sum_range", tuple(arg.result_type for arg in args), result.result_type
            ),
        ),
        args,
        ntn.Block(
            (
                ntn.Unpack(slot, tensor),
                ntn.Assign(result, ntn.Literal(np.int64(0))),
                ntn.Loop(
                    idx,
                    ntn.Call(ntn.Literal(make_extent), (start, end)),
                    ntn.Assign(
                        result,
                        ntn.Call(
                            ntn.Literal(ffuncs.add),
                            (result, ntn.Unwrap(ntn.Access(slot, ntn.Read(), (idx,)))),
                        ),
                    ),
                ),
                ntn.Return(result),
            )
        ),
    )
    return ntn.Module((function,))


@pytest.fixture(params=["ndarray", "dense"])
def tensor(request):
    data = np.array([0, 2, 0, 4, 0, 6], dtype=np.int64)
    match request.param:
        case "ndarray":
            return BufferizedNDArray.from_numpy(data)
        case "dense":
            return fiber_tensor(dense(element(np.int64(0)))).from_numpy(data)


@pytest.fixture
def sparse_tensor():
    return FiberTensor(
        SparseListLevel(
            ElementLevel(
                element(np.int64(0)), NumpyBuffer(np.array([2, 4, 6], dtype=np.int64))
            ),
            np.intp(6),
            NumpyBuffer(np.array([0, 3], dtype=np.intp)),
            NumpyBuffer(np.array([1, 3, 5], dtype=np.intp)),
        )
    )


@pytest.mark.parametrize("loader", [asm.AssemblyInterpreter, NumbaCompiler])
def test_safe_unfurl_bounds(tensor, loader):
    program = sum_program(tensor.ftype)
    kernel = NotationCompiler(loader(), mode=CompilerMode(safe=True))(program).sum_range
    for start, end in [(0, 6), (1, 5), (2, 3)]:
        assert kernel(tensor, np.intp(start), np.intp(end)) == sum(
            tensor.to_numpy()[start:end]
        )
    for start, end in [(-1, 2), (1, 7)]:
        with pytest.raises(AssertionError, match="Finch assertion failed"):
            kernel(tensor, np.intp(start), np.intp(end))


@pytest.mark.parametrize("loader", [asm.AssemblyInterpreter, NumbaCompiler])
def test_safe_sparse_bounds_before_traversal(sparse_tensor, loader):
    kernel = NotationCompiler(loader(), mode=CompilerMode(safe=True))(
        sum_program(sparse_tensor.ftype)
    ).sum_range
    for start, end in [(-1, 2), (1, 7)]:
        with pytest.raises(AssertionError, match="Finch assertion failed"):
            kernel(sparse_tensor, np.intp(start), np.intp(end))


def test_safe_sparse_subrange(sparse_tensor):
    kernel = NotationCompiler(NumbaCompiler(), mode=CompilerMode(safe=True))(
        sum_program(sparse_tensor.ftype)
    ).sum_range
    assert kernel(sparse_tensor, np.intp(1), np.intp(5)) == 6


def test_compiler_mode_scopes():
    mode = CompilerMode(safe=True)
    ctx = AssemblyContext(mode=mode)
    assert ctx.block().mode is mode
    assert ctx.scope().mode is mode


def test_fast_unfurl_omits_checks(tensor):
    program = sum_program(tensor.ftype)
    fast = AssemblyGenerator()(program)
    safe = AssemblyGenerator(mode=CompilerMode(safe=True))(program)
    assert not any(isinstance(node, asm.Assert) for node in PostOrderDFS(fast))
    assert any(isinstance(node, asm.Assert) for node in PostOrderDFS(safe))
    asm.AssemblyTypeChecker()(safe)


@pytest.mark.c_backend
def test_safe_unfurl_c():
    tensor = fiber_tensor(dense(element(np.int64(0)))).from_numpy(
        np.array([0, 2, 0, 4, 0, 6], dtype=np.int64)
    )
    kernel = NotationCompiler(CCompiler(), mode=CompilerMode(safe=True))(
        sum_program(tensor.ftype)
    ).sum_range
    assert kernel(tensor, np.intp(1), np.intp(5)) == 6


@pytest.mark.c_backend
def test_safe_unfurl_c_failure():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import numpy as np
from finch.codegen.c_codegen import CCompiler
from finch.compile import CompilerMode, NotationCompiler
from finch.tensor import dense, element, fiber_tensor
from finch.tests.test_safe_mode import sum_program

tensor = fiber_tensor(dense(element(np.int64(0)))).from_numpy(
    np.arange(3, dtype=np.int64)
)
kernel = NotationCompiler(CCompiler(), mode=CompilerMode(safe=True))(
    sum_program(tensor.ftype)
).sum_range
kernel(tensor, np.intp(0), np.intp(4))
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Finch assertion failed" in result.stderr


def test_assert_mlir_generation():
    size = asm.Variable("size", ftype(np.intp))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "checked_size",
                    asm.AssemblyKernelFType(
                        "checked_size", (size.result_type,), size.result_type
                    ),
                ),
                (size,),
                asm.Block(
                    (
                        asm.Assert(
                            asm.Call(
                                asm.Literal(ffuncs.ge),
                                (size, asm.Literal(np.intp(0))),
                            )
                        ),
                        asm.Return(size),
                    )
                ),
            ),
        )
    )
    code = MLIRGenerator()(program).code
    assert "cf.assert" in code
