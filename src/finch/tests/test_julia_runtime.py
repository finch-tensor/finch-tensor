import gc

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.codegen import NumpyBuffer
from finch.compile_jl.buffer import MinusOneBuffer
from finch.compile_jl.interop import _jl_index_buffer_to_python, tensor_to_jl
from finch.compile_jl.julia import jl, julia_available
from finch.compile_jl.runtime import (
    DefaultFinchJLRuntime,
    JuliaOwnedTensor,
    _KernalMetadata,
)


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


def _runtime(metadata):
    _requires_julia_backend()
    for name in metadata:
        jl.seval(f"{name}(args...) = nothing")

    runtime = DefaultFinchJLRuntime()
    runtime._kernels = {name: object() for name in metadata}
    runtime._kernel_metadata = {
        id(runtime._kernels[name]): value for name, value in metadata.items()
    }
    return runtime


def test_julia_owned_tensor_addition():
    _requires_julia_backend()
    source = ft.asarray(np.arange(4))
    tensor = JuliaOwnedTensor(
        source.ftype,
        source.shape,
        lambda _: None,
        tensor_to_jl(source),
        False,
    )

    assert (ft.add(tensor, 1).to_numpy() == np.arange(1, 5)).all()


def test_default_runtime_uses_a_free_buffer():
    runtime = _runtime(
        {"test_runtime_free_buffer": _KernalMetadata(frozenset({0}), (0,))}
    )
    tensor = ft.asarray(np.arange(4, dtype=np.float64))
    lease = runtime.free_pool.acquire(tensor.ftype, tensor.shape, False)
    runtime.free_pool.release_lease(lease)

    result = runtime.kernel_call(
        "test_runtime_free_buffer",
        runtime._kernels["test_runtime_free_buffer"],
        (tensor,),
    )[0]

    assert result.raw_julia_obj is lease.raw


def test_default_runtime_reuses_a_sparse_free_buffer():
    _requires_julia_backend()
    source = ft.asarray(sps.csr_array([[1, 0], [0, 2]], dtype=np.float64))
    target = ft.FiberTensor(
        ft.dense(ft.sparse_hash(ft.element(0.0), ft.intp), ft.intp).construct(
            (2, 2), pos=1
        )
    )

    def copy_sparse(source, target):
        return ft.add(source, target)

    copy_sparse = ft.jit(copy_sparse)
    with with_default_scheduler(COMPILE_JULIA):
        first = copy_sparse(source, target)
    assert isinstance(first, JuliaOwnedTensor)
    assert first._lease is not None
    lease = first._lease
    del first
    gc.collect()

    with with_default_scheduler(COMPILE_JULIA):
        result = copy_sparse(source, target)

    assert result.raw_julia_obj is lease.raw


def test_default_runtime_creates_a_buffer_when_the_pool_is_empty():
    runtime = _runtime(
        {"test_runtime_new_buffer": _KernalMetadata(frozenset({0}), (0,))}
    )
    tensor = ft.asarray(np.arange(4, dtype=np.float64))

    result = runtime.kernel_call(
        "test_runtime_new_buffer",
        runtime._kernels["test_runtime_new_buffer"],
        (tensor,),
    )[0]

    assert result._lease is not None
    assert result.shape == tensor.shape
    assert jl.isa(result.raw_julia_obj, jl.Finch.Tensor)


def test_default_runtime_reuses_translated_buffer_across_kernels():
    runtime = _runtime(
        {
            "test_runtime_first_kernel": _KernalMetadata(frozenset(), (0,)),
            "test_runtime_second_kernel": _KernalMetadata(frozenset(), (0,)),
        },
    )
    tensor = ft.asarray(np.arange(4, dtype=np.float64))

    first = runtime.kernel_call(
        "test_runtime_first_kernel",
        runtime._kernels["test_runtime_first_kernel"],
        (tensor,),
    )[0]
    second = runtime.kernel_call(
        "test_runtime_second_kernel",
        runtime._kernels["test_runtime_second_kernel"],
        (tensor,),
    )[0]

    assert first is second
