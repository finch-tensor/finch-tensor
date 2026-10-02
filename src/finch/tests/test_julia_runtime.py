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
            (2, 2), pos=0
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


def _addr(jl_vec) -> int:
    return int(jl.UInt(jl.pointer(jl_vec)))


def _np_addr(arr) -> int:
    return arr.__array_interface__["data"][0]


def test_tensor_to_jl_aliases_every_buffer():
    _requires_julia_backend()
    csr = ft.asarray(sps.random(40, 30, density=0.2, format="csr", random_state=0))
    raw = tensor_to_jl(csr)
    lvl = csr.lvl.lvl
    jl_lvl = raw.lvl.lvl
    assert _addr(jl_lvl.ptr.data) == _np_addr(lvl.ptr.arr)
    assert _addr(jl_lvl.idx.data) == _np_addr(lvl.idx.arr)
    assert _addr(jl_lvl.lvl.val) == _np_addr(lvl.lvl.val.arr)
    vals = lvl.lvl.val.arr
    vals[0] += 1.0
    assert float(jl_lvl.lvl.val[0]) == float(vals[0])

    coo = ft.asarray(sps.random(20, 10, density=0.3, format="coo", random_state=1))
    jl_coo = tensor_to_jl(coo).lvl
    for col, jl_col in zip(coo.lvl.tbl, jl_coo.tbl, strict=True):
        assert _addr(jl_col.data) == _np_addr(col.arr)

    dense = np.arange(12.0).reshape(3, 4)
    raw_dense = tensor_to_jl(dense)
    assert _addr(raw_dense.lvl.lvl.lvl.val) == _np_addr(dense)


def test_tensor_to_jl_copies_what_it_creates():
    _requires_julia_backend()
    strided = np.arange(20.0)[::2]
    raw = tensor_to_jl(strided)
    assert _addr(raw.lvl.lvl.val) != _np_addr(strided)
    np.testing.assert_array_equal(np.asarray(raw.lvl.lvl.val), strided)
    scalar = tensor_to_jl(3.5)
    assert float(scalar.lvl.val[0]) == 3.5
    empty = tensor_to_jl(ft.asarray(sps.csr_array((4, 4), dtype=np.float64)))
    assert int(jl.length(empty.lvl.lvl.idx)) == 0


def test_tuple_buffers_alias_with_julia_layout():
    _requires_julia_backend()
    from finch.algebra import TupleFType
    from finch.compile_jl.types import to_jl_type, to_jl_vector

    tup = TupleFType((ft.int64, ft.int32, ft.int64))
    jl_type = to_jl_type(tup)
    assert tup.dtype.itemsize == int(jl.sizeof(jl_type))
    arr = np.zeros(5, dtype=tup.dtype)
    for name in tup.dtype.names:
        arr[name] = np.arange(5)
    assert _addr(to_jl_vector(tup, arr)) == _np_addr(arr)
    shifted = to_jl_vector(tup, arr, offset=1)
    assert _addr(shifted) != _np_addr(arr)
    assert tuple(int(x) for x in shifted[2]) == (3, 3, 3)


def test_pooled_output_grows_out_of_wrapped_buffers():
    _requires_julia_backend()
    a = sps.random(60, 50, density=0.05, format="csr", random_state=3)
    b = sps.random(50, 40, density=0.05, format="csr", random_state=4)
    with with_default_scheduler(COMPILE_JULIA):
        for _ in range(2):
            got = ft.compute(ft.defer(ft.asarray(a)) @ ft.defer(ft.asarray(b)))
            gc.collect()
            jl.GC.gc()
            # Julia's dims are reversed relative to Python's
            dense = np.asarray(jl.Array(got.raw_julia_obj)).T
            np.testing.assert_allclose(dense, (a @ b).toarray())


def test_runtime_returned_translation_keeps_its_source():
    _requires_julia_backend()
    runtime = DefaultFinchJLRuntime()
    source = ft.asarray(np.arange(6.0))
    buf = runtime._to_julia_owned_tensor(source)
    handle = runtime._returned(buf, source)
    assert handle._source is source and handle.raw_julia_obj is buf.raw_julia_obj
    assert runtime._returned(buf, source) is handle
    lease = runtime.free_pool.acquire(source.ftype, source.shape, False)
    assert lease.source is not None
