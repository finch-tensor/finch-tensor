import numpy as np
import scipy.sparse as sps

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


def test_asarray_csr_honors_dense_sparse_list_format():
    matrix = sps.csr_matrix(np.array([[0.0, 2.0], [3.0, 0.0]]))
    tensor_format = ft.fiber_tensor(
        ft.dense(ft.sparse_list(ft.element(np.inf)))
    )

    tensor = ft.asarray(matrix, format=tensor_format)

    assert tensor.fill_value == np.inf
    assert isinstance(tensor.lvl, ft.DenseLevel)
    assert isinstance(tensor.lvl.lvl, ft.SparseListLevel)
