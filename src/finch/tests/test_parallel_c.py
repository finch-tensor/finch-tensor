import os
import shlex

import numpy as np
import pytest

from finch import finch_assembly as asm
from finch.algebra import ffuncs, ftypes
from finch.codegen import CCompiler, NumpyBuffer


pytestmark = [
    pytest.mark.c_backend,
    pytest.mark.parallel_c,
]


def test_parallel_gemv():
    m, n = 4, 3

    # Input matrices and vectors
    A = np.arange(1, 13, dtype=np.float64).reshape(m, n)
    x = np.array([1, 2, 3], dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)

    # Finch buffers
    a_buf = NumpyBuffer(A.ravel())
    x_buf = NumpyBuffer(x)
    y_buf = NumpyBuffer(y)

    # Assembly variables and slots
    a_v = asm.Variable("a", a_buf.ftype)
    a_slt = asm.Slot("a_", a_buf.ftype)

    x_v = asm.Variable("x", x_buf.ftype)
    x_slt = asm.Slot("x_", x_buf.ftype)

    y_v = asm.Variable("y", y_buf.ftype)
    y_slt = asm.Slot("y_", y_buf.ftype)

    # Loop variables
    i = asm.Variable("i", ftypes.intp)
    j = asm.Variable("j", ftypes.intp)

    # Flattened matrix index: i * n + j
    a_idx = asm.Call(
        asm.Literal(ffuncs.add),
        (
            asm.Call(
                asm.Literal(ffuncs.mul),
                (i, asm.Literal(np.intp(n))),
            ),
            j,
        ),
    )

    # GEMV computation: y[i] += A[i*n+j] * x[j]
    body = asm.Store(
        y_slt,
        i,
        asm.Call(
            asm.Literal(ffuncs.add),
            (
                asm.Load(y_slt, i),
                asm.Call(
                    asm.Literal(ffuncs.mul),
                    (
                        asm.Load(a_slt, a_idx),
                        asm.Load(x_slt, j),
                    ),
                ),
            ),
        ),
    )

    # Sequential inner loop over columns
    inner_loop = asm.ForLoop(
        j,
        asm.Literal(np.intp(0)),
        asm.Literal(np.intp(n)),
        asm.Block((body,)),
    )

    # Parallel outer loop over rows
    outer_loop = asm.ParallelForLoop(
        i,
        asm.Literal(np.intp(0)),
        asm.Literal(np.intp(m)),
        asm.Block((inner_loop,)),
    )

    # Construct complete Finch Assembly program
    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "gemv",
                    asm.AssemblyKernelFType(
                        "gemv",
                        (
                            a_v.result_type,
                            x_v.result_type,
                            y_v.result_type,
                        ),
                        ftypes.none_,
                    ),
                ),
                (a_v, x_v, y_v),
                asm.Block(
                    (
                        asm.Unpack(a_slt, a_v),
                        asm.Unpack(x_slt, x_v),
                        asm.Unpack(y_slt, y_v),
                        outer_loop,
                        asm.Repack(a_slt),
                        asm.Repack(x_slt),
                        asm.Repack(y_slt),
                        asm.Return(asm.Literal(None)),
                    )
                ),
            ),
        )
    )

    # Get OpenMP flags from the dedicated Pixi task
    flags = shlex.split(os.environ.get("FINCH_TEST_CFLAGS", ""))

    if "-fopenmp" not in flags:
        pytest.skip("OpenMP compiler flag not enabled")

    # Compile the Assembly program using the C backend
    compiler = CCompiler()
    compiler.cflags = [*compiler.cflags, *flags]

    library = compiler(prgm)
    kernel = library.gemv

    # Execute the compiled GEMV kernel
    kernel(a_buf, x_buf, y_buf)

    # Compare against NumPy
    expected = A @ x

    np.testing.assert_allclose(
        y_buf.arr,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )