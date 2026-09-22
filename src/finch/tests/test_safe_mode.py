import subprocess
import sys

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype
from finch.codegen import NumpyBuffer
from finch.codegen.c_codegen import CCompiler, CContext, CGenerator
from finch.codegen.mlir_codegen import MLIRContext, MLIRGenerator
from finch.codegen.numba_codegen import NumbaCompiler, NumbaContext, NumbaGenerator
from finch.compile import AssemblyContext, CompilerMode, NotationCompiler, make_extent
from finch.compile.lower import AssemblyGenerator
from finch.symbolic import PostOrderDFS
from finch.tensor import (
    BufferizedNDArray,
    DenseLevel,
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


@pytest.mark.parametrize("mode", [CompilerMode(safe=True), CompilerMode(debug=True)])
@pytest.mark.parametrize("loader", [asm.AssemblyInterpreter, NumbaCompiler])
def test_safe_unfurl_bounds(tensor, loader, mode):
    program = sum_program(tensor.ftype)
    kernel = NotationCompiler(loader(), mode=mode)(program).sum_range
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


@pytest.mark.parametrize(
    "context", [AssemblyContext, CContext, NumbaContext, MLIRContext]
)
def test_compiler_mode_scopes(context):
    mode = CompilerMode(debug=True)
    ctx = context(mode=mode)
    assert ctx.block().mode is mode
    if context is AssemblyContext:
        assert ctx.scope().mode is mode
    else:
        assert ctx.subblock().mode is mode


def buffer_program(buffer_type, *, write=False, resize=False, scan=False):
    buffer = asm.Variable("buffer", buffer_type)
    slot = asm.Slot("buffer_slot", buffer_type)
    index = asm.Variable("index", buffer_type.length_type)
    result = asm.Variable("result", buffer_type.element_type)
    offset = asm.Call(asm.Literal(ffuncs.add), (index, asm.Literal(np.intp(0))))
    body = [asm.Unpack(slot, buffer)]
    if resize:
        body.append(asm.Resize(slot, asm.Literal(np.intp(5))))
    if scan:
        body.append(
            asm.WhileLoop(
                asm.Call(
                    asm.Literal(ffuncs.ne),
                    (asm.Load(slot, offset), asm.Literal(np.int64(0))),
                ),
                asm.Block(
                    (
                        asm.Assign(
                            index,
                            asm.Call(
                                asm.Literal(ffuncs.add),
                                (index, asm.Literal(np.intp(1))),
                            ),
                        ),
                        asm.If(
                            asm.Call(
                                asm.Literal(ffuncs.ge),
                                (index, asm.Literal(np.intp(4))),
                            ),
                            asm.Break(),
                        ),
                    )
                ),
            )
        )
    if write:
        body.append(asm.Store(slot, offset, asm.Literal(np.int64(42))))
    body.extend(
        (
            asm.Assign(result, index if scan else asm.Load(slot, offset)),
            asm.Repack(slot),
            asm.Return(result),
        )
    )
    return asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "access",
                    asm.AssemblyKernelFType(
                        "access", (buffer_type, index.result_type), result.result_type
                    ),
                ),
                (buffer, index),
                asm.Block(tuple(body)),
            ),
        )
    )


@pytest.mark.parametrize(
    "loader",
    [
        asm.AssemblyInterpreter,
        NumbaCompiler,
        pytest.param(CCompiler, marks=pytest.mark.c_backend),
    ],
)
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("size", [0, 1, 3])
def test_debug_buffer_bounds(loader, write, size):
    buffer = NumpyBuffer(np.arange(size, dtype=np.int64))
    program = buffer_program(buffer.ftype, write=write)
    kernel = loader()(program, mode=CompilerMode(debug=True)).access
    assert kernel.ftype == program.funcs[0].name.result_type
    for index in range(size):
        assert kernel(buffer, np.intp(index)) == (42 if write else index)
        assert buffer.load(index) == (42 if write else index)
    if loader is not CCompiler:
        for index in (-1, size, size + 1):
            with pytest.raises(AssertionError):
                kernel(buffer, np.intp(index))


@pytest.mark.parametrize(
    "loader",
    [
        asm.AssemblyInterpreter,
        NumbaCompiler,
        pytest.param(CCompiler, marks=pytest.mark.c_backend),
    ],
)
def test_debug_buffer_resize_and_while(loader):
    mode = CompilerMode(debug=True)
    buffer = NumpyBuffer(np.array([1, 1, 0], dtype=np.int64))
    scan = loader()(buffer_program(buffer.ftype, scan=True), mode=mode).access
    assert scan(buffer, np.intp(0)) == 2
    if loader is not CCompiler:
        with pytest.raises(AssertionError):
            scan(NumpyBuffer(np.ones(3, dtype=np.int64)), np.intp(0))
    resize = loader()(
        buffer_program(buffer.ftype, write=True, resize=True), mode=mode
    ).access
    assert resize(buffer, np.intp(4)) == 42
    assert buffer.length() == 5
    assert buffer.load(4) == 42


@pytest.mark.parametrize("loader", [asm.AssemblyInterpreter, NumbaCompiler])
def test_debug_mode_checks_storage_bounds(loader):
    tensor = fiber_tensor(dense(element(np.int64(0)))).from_numpy(
        np.arange(3, dtype=np.int64)
    )
    assert isinstance(tensor.lvl, DenseLevel)
    tensor.lvl.dimension = np.intp(4)
    kernel = NotationCompiler(loader(), mode=CompilerMode(debug=True))(
        sum_program(tensor.ftype)
    ).sum_range
    with pytest.raises(AssertionError):
        kernel(tensor, np.intp(0), np.intp(4))


@pytest.mark.parametrize("generator", [CGenerator, NumbaGenerator, MLIRGenerator])
def test_buffer_checks_only_in_debug_mode(generator):
    buffer = NumpyBuffer(np.arange(3, dtype=np.int64))
    program = buffer_program(buffer.ftype, write=True)
    for mode in (CompilerMode(), CompilerMode(safe=True)):
        assert "assert" not in generator()(program, mode=mode).code.lower()
    checked = generator()(program, mode=CompilerMode(debug=True)).code
    assertion = "cf.assert" if generator is MLIRGenerator else "Finch assertion failed"
    assert checked.count(assertion) == 2


@pytest.mark.c_backend
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("size", [0, 1, 3])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_debug_buffer_c_failure(write, size, side):
    index = -1 if side == "lower" else size
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import numpy as np
from finch.codegen import NumpyBuffer
from finch.codegen.c_codegen import CCompiler
from finch.compile import CompilerMode
from finch.tests.test_safe_mode import buffer_program

buffer = NumpyBuffer(np.arange({size}, dtype=np.int64))
kernel = CCompiler()(
    buffer_program(buffer.ftype, write={write}), mode=CompilerMode(debug=True)
).access
kernel(buffer, np.intp({index}))
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Finch assertion failed" in result.stderr


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
