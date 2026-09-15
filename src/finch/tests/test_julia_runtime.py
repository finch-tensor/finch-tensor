from types import SimpleNamespace

import numpy as np

import finch as ft
import finch.compile_jl.runtime as runtime_module
from finch.codegen import NumpyBuffer
from finch.compile_jl.buffer import MinusOneBuffer
from finch.compile_jl.interop import _jl_index_buffer_to_python
from finch.compile_jl.julia import jl, julia_available
from finch.compile_jl.runtime import DefaultFinchJLRuntime, _KernalMetadata


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


class RecordingJulia:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call


def _runtime(monkeypatch, metadata):
    raw_tensors = []

    def tensor_to_julia(tensor, pin_fill=False):
        raw = object()
        raw_tensors.append(raw)
        return raw

    julia = RecordingJulia()
    monkeypatch.setattr(runtime_module, "python_tensor_to_jl", tensor_to_julia)
    monkeypatch.setattr(runtime_module, "jl", julia)
    runtime = DefaultFinchJLRuntime()
    runtime._kernels_by_name = {
        name: SimpleNamespace(func_name=name, dynamic_args=()) for name in metadata
    }
    runtime._kernel_metadata = metadata
    return runtime, julia, raw_tensors


def test_default_runtime_uses_a_free_buffer(monkeypatch):
    runtime, _, _ = _runtime(
        monkeypatch,
        {"kernel": _KernalMetadata(frozenset({0}), (0,))},
    )
    tensor = ft.asarray(np.arange(4, dtype=np.float64))
    lease = runtime.free_pool.acquire(tensor.ftype, tensor.shape, False)
    runtime.free_pool.release_lease(lease)

    result = runtime.kernel_call("kernel", (tensor,))[0]

    assert result.raw_julia_obj is lease.raw


def test_default_runtime_creates_a_buffer_when_the_pool_is_empty(monkeypatch):
    runtime, _, raw_tensors = _runtime(
        monkeypatch,
        {"kernel": _KernalMetadata(frozenset({0}), (0,))},
    )
    tensor = ft.asarray(np.arange(4, dtype=np.float64))

    result = runtime.kernel_call("kernel", (tensor,))[0]

    assert result.raw_julia_obj is raw_tensors[0]
    assert len(raw_tensors) == 1


def test_default_runtime_reuses_an_input_across_kernels(monkeypatch):
    runtime, julia, raw_tensors = _runtime(
        monkeypatch,
        {
            "first_kernel": _KernalMetadata(frozenset(), (0,)),
            "second_kernel": _KernalMetadata(frozenset(), (0,)),
        },
    )
    tensor = ft.asarray(np.arange(4, dtype=np.float64))

    runtime.kernel_call("first_kernel", (tensor,))
    runtime.kernel_call("second_kernel", (tensor,))

    assert len(raw_tensors) == 1
    assert julia.calls == [
        ("first_kernel", (raw_tensors[0],)),
        ("second_kernel", (raw_tensors[0],)),
    ]
