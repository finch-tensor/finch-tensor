import numpy as np

import finch as ft
from finch.codegen import NumpyBuffer
from finch.compile_jl.buffer import MinusOneBuffer
from finch.compile_jl.interop import JuliaBufferContext, _jl_index_buffer_to_python
from finch.compile_jl.julia import jl, julia_available


def _requires_julia_backend():
    if not julia_available():
        import pytest

        pytest.skip("the julia extra (juliapkg, juliacall) is not installed")


def test_minus_one_buffer_loads_and_stores_with_offset():
    arr = np.array([1, 3, 5], dtype=np.intp)
    buffer = MinusOneBuffer(NumpyBuffer(arr))

    assert buffer.load(0) == 0
    assert buffer.load(1) == 2

    buffer.store(2, 7)

    assert arr[2] == 8


def test_minus_one_buffer_does_not_copy_backing_data():
    _requires_julia_backend()

    jl_vec = jl.Vector[jl.Int]([1, 3, 4])
    buffer = _jl_index_buffer_to_python(jl_vec)

    assert isinstance(buffer, MinusOneBuffer)
    raw = jl_vec.to_numpy(copy=False)
    assert buffer.data.arr.ctypes.data == raw.ctypes.data

    buffer.store(1, 7)

    assert int(jl_vec[1]) == 8


def test_julia_buffer_context_reuses_buffers_after_kernel_invocation():
    _requires_julia_backend()

    data = np.arange(4, dtype=np.float64)
    context = JuliaBufferContext()
    first_arg = ft.asarray(data)
    first_jl = context.tensor_to_jl(first_arg)

    returned_jl = jl.first_arg(first_jl)
    context.tensor_to_python(returned_jl)

    second_arg = ft.asarray(data)
    second_jl = context.tensor_to_jl(second_arg)

    assert second_jl is first_jl
