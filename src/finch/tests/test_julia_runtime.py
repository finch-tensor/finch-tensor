import gc
import weakref

import pytest

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.algebra import TupleFType, np_dtype
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.codegen import NumpyBuffer
from finch.compile_jl.buffer import MinusOneBuffer
from finch.compile_jl.interop import (
    _buffer_to_jl,
    _jl_array_to_python_no_copy,
    _jl_index_buffer_to_python,
    jl_tensor_to_python,
    level_to_jl,
    tensor_to_jl,
)
from finch.compile_jl.julia import jl, julia_available
from finch.compile_jl.runtime import (
    DefaultFinchJLRuntime,
    JuliaOwnedTensor,
    _KernalMetadata,
)
from finch.compile_jl.types import _level_constructor_str, to_jl_vector


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


def test_minus_one_buffer_preserves_hash_key_state():
    key_type = TupleFType.from_tuple((ft.intp, ft.int32, ft.uint8))
    raw = np.array([(1, 3, 128), (2, 1, 0)], dtype=np_dtype(key_type))
    buffer = MinusOneBuffer(NumpyBuffer(raw), index_fields=(0, 1))

    assert buffer.load(0) == (0, 2, 128)
    assert buffer.load(1) == (1, 0, 0)
    assert buffer.arr.tolist() == [(0, 2, 128), (1, 0, 0)]
    buffer.store(1, (3, 4, 127))
    assert raw.tolist() == [(1, 3, 128), (4, 5, 127)]
    assert buffer.ftype != MinusOneBuffer(NumpyBuffer(raw)).ftype
    assert buffer.ftype(1).index_fields == (0, 1)


@pytest.mark.parametrize("new_length", [0, 2, 8])
def test_julia_buffer_takes_ownership_on_resize(new_length):
    _requires_julia_backend()
    source = np.arange(4, dtype=np.int64)
    buffer = NumpyBuffer(source)
    vector = _buffer_to_jl(buffer)
    assert buffer.owner is source
    assert not jl.FinchInterop.owns_storage(vector)
    assert _jl_array_to_python_no_copy(vector).ctypes.data == source.ctypes.data

    vector[0] = 10
    assert source[0] == 10
    getattr(jl, "resize!")(vector, new_length)
    assert jl.FinchInterop.owns_storage(vector)
    assert list(vector)[: min(4, new_length)] == source[:new_length].tolist()
    if new_length:
        vector[0] = 20
        assert source[0] == 10


def test_julia_buffer_preserves_borrowed_and_exported_storage():
    _requires_julia_backend()
    source = np.arange(4, dtype=np.int64)
    reference = weakref.ref(source)
    vector = _buffer_to_jl(NumpyBuffer(source))
    del source
    gc.collect()
    jl.GC.gc()
    assert reference() is not None
    assert list(vector) == [0, 1, 2, 3]

    old_view = _jl_array_to_python_no_copy(vector)
    getattr(jl, "resize!")(vector, 6)
    vector[0] = 10
    owned_view = _jl_array_to_python_no_copy(vector)
    assert not jl.FinchInterop.owns_storage(vector)
    getattr(jl, "resize!")(vector, 2)
    vector[0] = 20
    del vector
    gc.collect()
    jl.GC.gc()
    assert reference() is not None
    np.testing.assert_array_equal(old_view, [0, 1, 2, 3])
    np.testing.assert_array_equal(owned_view[:4], [10, 1, 2, 3])


def test_numpy_buffer_resize_preserves_julia_borrow():
    _requires_julia_backend()
    source = NumpyBuffer(np.arange(4, dtype=np.int64))
    owner = source.owner
    vector = _buffer_to_jl(source)
    source.resize(6)
    assert source.owner is not owner
    source.store(0, 10)
    assert list(vector) == [0, 1, 2, 3]
    vector[1] = 20
    assert owner[1] == 20
    assert source.load(1) == 1


def test_julia_buffer_overlapping_copy_preserves_values():
    _requires_julia_backend()
    source = np.arange(4, dtype=np.int64)
    vector = _buffer_to_jl(NumpyBuffer(source), offset=1)
    getattr(jl, "copyto!")(vector, 2, vector, 1, 3)
    assert list(vector) == [1, 1, 2, 3]
    np.testing.assert_array_equal(source, [0, 0, 1, 2])


@pytest.mark.parametrize("aligned", [False, True])
def test_julia_buffer_borrows_hash_keys_without_shifting_state(aligned):
    _requires_julia_backend()
    key_type = TupleFType.from_tuple((ft.intp, ft.int32, ft.uint8))
    dtype = np.dtype(np_dtype(key_type).descr, align=aligned)
    source = np.array([(0, 2, 128), (1, 0, 127)], dtype=dtype)
    vector = _buffer_to_jl(NumpyBuffer(source), offset=1, index_fields=(0, 1))
    assert _jl_array_to_python_no_copy(vector).ctypes.data == source.ctypes.data
    assert [tuple(entry) for entry in vector] == [(1, 3, 128), (2, 1, 127)]
    vector[1] = (4, 5, 0)
    assert source.tolist() == [(0, 2, 128), (3, 4, 0)]
    getattr(jl, "push!")(vector, (5, 6, 128))
    assert jl.FinchInterop.owns_storage(vector)
    assert [tuple(entry) for entry in vector] == [(1, 3, 128), (4, 5, 0), (5, 6, 128)]
    assert source.tolist() == [(0, 2, 128), (3, 4, 0)]
    getattr(jl, "empty!")(vector)
    assert len(vector) == 0


def test_julia_buffer_readonly_storage_detaches_on_resize():
    _requires_julia_backend()
    source = np.arange(4, dtype=np.float64)
    source.flags.writeable = False
    vector = _buffer_to_jl(NumpyBuffer(source))
    assert _jl_array_to_python_no_copy(vector).ctypes.data == source.ctypes.data
    assert not _jl_array_to_python_no_copy(vector).flags.writeable
    with pytest.raises(Exception, match="ReadOnlyMemoryError"):
        vector[0] = 10
    getattr(jl, "resize!")(vector, 5)
    vector[0] = 10
    assert source[0] == 0


def test_julia_buffer_converts_incompatible_dtype():
    _requires_julia_backend()
    source = np.arange(4, dtype=np.int32)
    vector = to_jl_vector(ft.float64, source)
    assert jl.eltype(vector) == jl.Float64
    assert list(vector) == [0.0, 1.0, 2.0, 3.0]


def test_julia_buffer_borrows_nested_tuple_layout():
    _requires_julia_backend()
    type_ = TupleFType.from_tuple(
        (TupleFType.from_tuple((ft.int32, ft.int64)), ft.uint8)
    )
    source = np.array([((0, 2), 128), ((3, 4), 127)], dtype=np_dtype(type_))
    vector = _buffer_to_jl(NumpyBuffer(source), offset=1, index_fields=(0,))
    view = _jl_array_to_python_no_copy(vector)
    assert view.ctypes.data == source.ctypes.data
    assert view.tolist() == source.tolist()
    assert tuple(vector[0][0]) == (1, 3)
    vector[1] = ((5, 6), 0)
    assert source.tolist() == [((0, 2), 128), ((4, 5), 0)]


@pytest.mark.parametrize("level_type", [ft.sparse_hash, ft.sparse_bytemap])
@pytest.mark.parametrize("dimension_type", [ft.intp, ft.int32])
def test_julia_sparse_level_constructor_matches_argument(level_type, dimension_type):
    _requires_julia_backend()
    level_ftype = level_type(ft.element(0), dimension_type)
    level = level_ftype.construct((4,), pos=1)
    prototype = jl.seval(_level_constructor_str(level_ftype))
    actual = level_to_jl(level)

    assert jl.typeof(actual) == jl.typeof(prototype)
    recovered = jl_tensor_to_python(jl.Tensor(actual)).lvl
    assert recovered.ftype == level_ftype


@pytest.mark.parametrize("kind", ["hash", "bytemap"])
@pytest.mark.parametrize("copy_buffers", [False, True])
def test_julia_sparse_level_round_trip(kind, copy_buffers):
    _requires_julia_backend()
    format_ = (
        "SparseHash{Int32}(Element(0), 3, 4, typemax(UInt) - UInt(2))"
        if kind == "hash"
        else "SparseByteMap{Int32}(Element(0), Int32(3))"
    )
    original = jl.seval(f"Tensor(Dense({format_}), [1 0; 0 3; 2 4])")
    tensor = jl_tensor_to_python(original)
    level = tensor.lvl.lvl
    assert level.dimension == np.int32(3)
    assert level.ftype.dimension_type == ft.int32

    if kind == "hash":
        assert level.subtables == 4
        assert level.pos_multiplier == np.iinfo(np.uintp).max - 2
        assert level.key.arr["element_2"].tolist() == [128] * 4
        assert (
            level.key.data.arr.ctypes.data
            == original.lvl.lvl.key.to_numpy(copy=False).ctypes.data
        )
        np.testing.assert_array_equal(
            level.tbl_count.arr, original.lvl.lvl.tbl_count.to_numpy(copy=False)
        )
    else:
        np.testing.assert_array_equal(level.ptr.arr, [0, 2, 4])
        np.testing.assert_array_equal(level.srt.arr, [0, 2, 4, 5])

    if copy_buffers:
        names = (
            ("ptr", "tbl_ctrl", "tbl", "key", "perm", "tbl_count")
            if kind == "hash"
            else ("ptr", "tbl", "srt")
        )
        for name in names:
            setattr(level, name, NumpyBuffer(getattr(level, name).arr.copy()))

    round_trip = tensor_to_jl(tensor)
    same_values = jl.seval("(A, B) -> Array(A) == Array(B)")
    assert same_values(original, round_trip)
    if kind == "hash":
        assert round_trip.lvl.lvl.pos_multiplier == original.lvl.lvl.pos_multiplier
        lookup = jl.seval(
            "(lvl, p, i) -> Finch.sparse_hash_lookup(lvl.tbl_ctrl, lvl.tbl, "
            "lvl.key, p, i, lvl.subtables, lvl.pos_multiplier)"
        )
        for parent in (1, 2):
            for index in (1, 2, 3):
                assert lookup(round_trip.lvl.lvl, parent, index) == lookup(
                    original.lvl.lvl, parent, index
                )


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
