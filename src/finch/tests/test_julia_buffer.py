import numpy as np

from finch.codegen import NumpyBuffer, NumpyBufferFType
from finch.compile_jl.buffer import (
    MinusOneBuffer,
    MinusOneBufferFType,
    PlusOneBuffer,
    PlusOneBufferFType,
    buffer_to_jlobj,
)
from finch.compile_jl.julia import julia_available


def _requires_julia_backend():
    if not julia_available():
        import pytest

        pytest.skip("the julia extra (juliapkg, juliacall) is not installed")


def test_minus_one_buffer_offsets_loads_and_stores():
    arr = np.array([1, 3, 5], dtype=np.intp)
    buf = MinusOneBuffer(NumpyBuffer(arr))

    assert buf.load(0) == 0
    assert buf.load(1) == 2

    buf.store(2, 7)

    assert arr[2] == 8


def test_offset_buffers_do_not_expose_materialized_arrays():
    arr = np.array([1, 3, 5], dtype=np.intp)
    data = NumpyBuffer(arr)

    assert not hasattr(PlusOneBuffer(data), "arr")
    assert not hasattr(MinusOneBuffer(data), "arr")


def test_offset_buffer_ftypes_forward_to_wrapped_buffer():
    arr = np.array([0, 1, 2], dtype=np.intp)
    numpy_buf = NumpyBuffer(arr)

    plus_ftype = PlusOneBuffer(numpy_buf).ftype
    minus_ftype = MinusOneBuffer(numpy_buf).ftype

    assert isinstance(plus_ftype, PlusOneBufferFType)
    assert isinstance(minus_ftype, MinusOneBufferFType)
    assert isinstance(plus_ftype.data_ftype, NumpyBufferFType)
    assert isinstance(minus_ftype.data_ftype, NumpyBufferFType)
    assert plus_ftype.element_type == numpy_buf.ftype.element_type
    assert minus_ftype.element_type == numpy_buf.ftype.element_type
    assert plus_ftype.length_type == numpy_buf.ftype.length_type
    assert minus_ftype.length_type == numpy_buf.ftype.length_type


def test_minus_one_buffer_to_julia_object_unwraps_data():
    arr = np.array([1, 2, 3], dtype=np.intp)
    buf = MinusOneBuffer(NumpyBuffer(arr))

    assert buffer_to_jlobj(buf) is arr


def test_plus_one_interop_unwraps_minus_one_buffer(monkeypatch):
    from finch.compile_jl import interop

    arr = np.array([1, 2, 3], dtype=np.intp)
    data = NumpyBuffer(arr)
    buf = MinusOneBuffer(data)
    converted = object()

    def fake_buffer_to_jl(buffer):
        assert buffer is data
        return converted

    monkeypatch.setattr(interop, "_buffer_to_jl", fake_buffer_to_jl)

    assert interop._plus_one_buffer_to_jl(buf) is converted


def test_julia_owned_buffer_to_python_is_zero_copy():
    _requires_julia_backend()
    from finch.compile_jl.interop import _jl_buffer_to_python, _python_buffer_owner
    from finch.compile_jl.julia import jl

    jl_vec = jl.Vector[jl.Int]([1, 2, 3])
    buf = _jl_buffer_to_python(jl_vec)

    assert isinstance(buf, NumpyBuffer)
    assert _python_buffer_owner(buf) == (jl_vec,)

    buf.store(0, 42)

    assert int(jl_vec[0]) == 42


def test_plain_julia_index_buffer_to_python_uses_minus_one_without_mutating():
    _requires_julia_backend()
    from finch.compile_jl.interop import _jl_index_buffer_to_python
    from finch.compile_jl.julia import jl

    jl_vec = jl.Vector[jl.Int]([1, 3, 4])
    buf = _jl_index_buffer_to_python(jl_vec)

    assert isinstance(buf, MinusOneBuffer)
    assert buf.load(0) == 0
    assert buf.load(1) == 2
    assert [int(v) for v in jl_vec] == [1, 3, 4]

    buf.store(2, 7)

    assert int(jl_vec[2]) == 8


def test_plus_one_julia_index_buffer_to_python_unwraps_data_zero_copy():
    _requires_julia_backend()
    from finch.compile_jl.interop import _jl_index_buffer_to_python
    from finch.compile_jl.julia import jl

    raw = jl.Vector[jl.Int]([0, 2, 3])
    plus_one = jl.Finch.PlusOneVector(raw)
    buf = _jl_index_buffer_to_python(plus_one)

    assert isinstance(buf, NumpyBuffer)
    np.testing.assert_array_equal(buf.arr, np.array([0, 2, 3], dtype=np.intp))

    buf.store(1, 5)

    assert int(raw[1]) == 5


def test_plus_one_julia_int32_index_buffer_to_python_is_zero_copy():
    _requires_julia_backend()
    from finch.compile_jl.interop import _jl_index_buffer_to_python
    from finch.compile_jl.julia import jl

    raw = jl.Vector[jl.Int32]([0, 2, 3])
    plus_one = jl.Finch.PlusOneVector(raw)
    buf = _jl_index_buffer_to_python(plus_one)

    assert isinstance(buf, NumpyBuffer)
    assert buf.arr.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(buf.arr, np.array([0, 2, 3], dtype=np.int32))

    buf.store(1, 5)

    assert int(raw[1]) == 5


def test_python_owned_julia_buffer_round_trip_retains_python_owner():
    _requires_julia_backend()
    from finch.compile_jl.interop import (
        _jl_buffer_to_python,
        _python_buffer_owner,
        tensor_to_jl,
    )

    arr = np.array([1, 2, 3], dtype=np.int64)
    jl_tensor = tensor_to_jl(arr)
    buf = _jl_buffer_to_python(jl_tensor.lvl.lvl.val)

    owner = _python_buffer_owner(buf)
    assert owner is not None
    assert any(item is arr for item in owner)
    assert buf.arr.ctypes.data == arr.ctypes.data

    buf.store(0, 9)

    assert arr[0] == 9
