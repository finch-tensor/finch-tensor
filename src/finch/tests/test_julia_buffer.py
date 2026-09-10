import numpy as np

import finch as ft
from finch.codegen import NumpyBuffer
from finch.compile_jl.buffer import MinusOneBuffer
from finch.compile_jl.compiler import FinchJLKernel
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


def test_julia_buffer_context_reuses_julia_backed_result_wrapper():
    _requires_julia_backend()

    context = JuliaBufferContext()
    python_input = ft.asarray(np.arange(4, dtype=np.float64))
    julia_tensor = context.tensor_to_jl(python_input)

    # JuliaCall returns a fresh Python proxy here, even though Julia returns
    # the exact same tensor object.  The context should identify it by Julia
    # object identity and return the previously recovered Python view.
    first_result = context.tensor_to_python(jl.first_arg(julia_tensor))
    second_result = context.tensor_to_python(jl.first_arg(julia_tensor))

    assert first_result is not python_input
    assert second_result is first_result
def test_julia_kernel_output_pool_never_reuses_a_current_input():
    """Ping-pong selection leaves the active state buffer read-only this call."""

    kernel = object.__new__(FinchJLKernel)
    active_state = object()
    spare_state = object()
    fresh_output = object()
    kernel._result_arg_positions = (2,)
    kernel._output_pools = {2: [active_state, spare_state]}

    call_args = kernel._recycled_output_args((active_state, object(), fresh_output))

    assert call_args[0] is active_state
    assert call_args[2] is spare_state


def test_julia_kernel_recycles_only_arguments_reset_before_first_read():
    code = """
    Finch.@finch_kernel function kernel_example(v0,v1,v2)
        v0 .= 0
        v1[] = v0[]
        v2 .= false
        return v1
    end
    """

    assert FinchJLKernel._find_reset_arg_positions(code) == frozenset({0, 2})
