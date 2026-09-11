import numpy as np

import finch as ft
import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import overwrite
from finch.codegen import NumpyBuffer
from finch.compile_jl.analyze import find_reset_arg_positions
from finch.compile_jl.buffer import MinusOneBuffer
from finch.compile_jl.interop import (
    JuliaBufferContext,
    JuliaKernelArgs,
    _jl_index_buffer_to_python,
)
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


def _reset(var):
    return ntn.Declare(var, ntn.Literal(0), ntn.Literal(overwrite), ())


def test_julia_kernel_finds_arguments_reset_before_read():
    v0, v1, v2 = (ntn.Variable(f"v{i}") for i in range(3))
    func = ntn.Function(
        ntn.Variable("kernel_example"),
        (v0, v1, v2),
        ntn.Block((_reset(v0), ntn.Assign(ntn.Variable("tmp"), v1), _reset(v2))),
    )

    assert find_reset_arg_positions(func) == frozenset({0, 2})


def test_julia_kernel_requires_reset_on_every_path():
    v0 = ntn.Variable("v0")
    func = ntn.Function(
        ntn.Variable("kernel_example"),
        (v0,),
        ntn.IfElse(ntn.Literal(True), _reset(v0), ntn.Block(())),
    )

    assert find_reset_arg_positions(func) == frozenset()


def test_julia_kernel_accepts_reset_on_every_branch():
    v0 = ntn.Variable("v0")
    func = ntn.Function(
        ntn.Variable("kernel_example"),
        (v0,),
        ntn.IfElse(ntn.Literal(True), _reset(v0), _reset(v0)),
    )

    assert find_reset_arg_positions(func) == frozenset({0})


def test_julia_kernel_rejects_loop_only_reset():
    v0 = ntn.Variable("v0")
    func = ntn.Function(
        ntn.Variable("kernel_example"),
        (v0,),
        ntn.Loop(ntn.Variable("i"), ntn.Literal(1), _reset(v0)),
    )

    assert find_reset_arg_positions(func) == frozenset()


def test_julia_kernel_finds_reset_through_unpack_slot():
    v0 = ntn.Variable("v0")
    v0_slot = ntn.Slot("v0_slot", None)
    size = ntn.Variable("size")
    func = ntn.Function(
        ntn.Variable("kernel_example"),
        (v0,),
        ntn.Block(
            (
                ntn.Unpack(v0_slot, v0),
                ntn.Assign(size, ntn.Dimension(v0_slot, ntn.Literal(0))),
                _reset(v0_slot),
            )
        ),
    )

    assert find_reset_arg_positions(func) == frozenset({0})


def test_julia_buffer_context_reuses_free_compatible_tensor():
    _requires_julia_backend()

    context = JuliaBufferContext()
    first = ft.asarray(np.arange(4, dtype=np.float64))
    first_jl = context.tensor_to_jl(first)
    first_key = context._cache_key(first)
    type_name = str(jl.string(jl.typeof(first_jl)))
    context.release_reset_arguments((first_key,), frozenset({0}))

    second = ft.asarray(np.arange(4, dtype=np.float64) + 1)
    raw_args, _, _ = context.resolve_arguments(
        (second,),
        kernel_args=JuliaKernelArgs(
            type_names=(type_name,),
            dynamic_positions=(),
            reset_positions=frozenset({0}),
            return_positions=(),
        ),
    )

    assert raw_args[0] is first_jl
