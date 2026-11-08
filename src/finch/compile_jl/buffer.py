from abc import ABC
from typing import TypeVar

import numpy as np

from finch.algebra.ftypes import FType
from finch.codegen import NumpyBuffer
from finch.finch_assembly import Buffer, BufferFType

from .julia import jl

FT = TypeVar("FT", bound=FType)


def _shift_index_fields(arr, delta: int, index_fields: tuple[int, ...] | None):
    # New buffers hold zeros, which an offset buffer stores offset.
    if arr.dtype.names is None:
        arr += delta
        return
    for i, name in enumerate(arr.dtype.names):
        if index_fields is None or i in index_fields:
            arr[name] += delta


class PlusOneBufferFType(BufferFType[FT]):
    def __init__(
        self, data_ftype: BufferFType[FT], index_fields: tuple[int, ...] | None = None
    ):
        self.data_ftype = data_ftype
        self.index_fields = index_fields

    def __call__(self, *args, **kwargs):
        data = self.data_ftype(*args, **kwargs)
        _shift_index_fields(data.arr, -1, self.index_fields)
        return PlusOneBuffer(data, self.index_fields)

    @property
    def element_type(self):
        return self.data_ftype.element_type

    @property
    def length_type(self):
        return self.data_ftype.length_type

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PlusOneBufferFType)
            and self.data_ftype == other.data_ftype
            and self.index_fields == other.index_fields
        )

    def __hash__(self):
        return hash((self.data_ftype, self.index_fields))

    def __repr__(self):
        return f"PlusOneBufferFType({self.data_ftype!r}, {self.index_fields!r})"


class PlusOneBuffer(Buffer, ABC):
    """
    Buffer that adds one to each element when loaded and subtracts one from each
    element when stored.
    """

    def __init__(self, data: NumpyBuffer, index_fields: tuple[int, ...] | None = None):
        self.data = data
        self.index_fields = index_fields

    @property
    def owner(self):
        return self.data.owner

    @property
    def ftype(self):
        return PlusOneBufferFType(self.data.ftype, self.index_fields)

    def length(self):
        return self.data.length()

    def to_numpy(self, *, copy: bool | None = None):
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        arr = self.data.to_numpy()
        if arr.dtype.names is None:
            return arr + 1
        out = arr.copy()
        for i, name in enumerate(arr.dtype.names):
            if self.index_fields is None or i in self.index_fields:
                out[name] += 1
        return out

    @property
    def element_type(self):
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self.data.element_type

    @property
    def length_type(self):
        return self.data.length_type

    def load(self, idx: int):
        value = self.data.load(idx)
        if isinstance(value, tuple):
            return tuple(
                field + 1
                if self.index_fields is None or i in self.index_fields
                else field
                for i, field in enumerate(value)
            )
        return value + 1

    def store(self, idx: int, val):
        if isinstance(val, tuple):
            val = tuple(
                field - 1
                if self.index_fields is None or i in self.index_fields
                else field
                for i, field in enumerate(val)
            )
        else:
            val = val - 1

        self.data.store(idx, val)

    def resize(self, len: int):
        self.data.resize(len)


class MinusOneBufferFType(BufferFType[FT]):
    def __init__(
        self, data_ftype: BufferFType[FT], index_fields: tuple[int, ...] | None = None
    ):
        self.data_ftype = data_ftype
        self.index_fields = index_fields

    def __call__(self, *args, **kwargs):
        data = self.data_ftype(*args, **kwargs)
        _shift_index_fields(data.arr, 1, self.index_fields)
        return MinusOneBuffer(data, self.index_fields)

    @property
    def element_type(self):
        return self.data_ftype.element_type

    @property
    def length_type(self):
        return self.data_ftype.length_type

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MinusOneBufferFType)
            and self.data_ftype == other.data_ftype
            and self.index_fields == other.index_fields
        )

    def __hash__(self):
        return hash((self.data_ftype, self.index_fields))

    def __repr__(self):
        return f"MinusOneBufferFType({self.data_ftype!r}, {self.index_fields!r})"


class MinusOneBuffer(Buffer, ABC):
    """Subtract one on loads and add one on stores, only for index fields."""

    def __init__(self, data: NumpyBuffer, index_fields: tuple[int, ...] | None = None):
        self.data = data
        self.index_fields = index_fields

    @property
    def owner(self):
        return self.data.owner

    @property
    def ftype(self):
        return MinusOneBufferFType(self.data.ftype, self.index_fields)

    def length(self):
        return self.data.length()

    @property
    def element_type(self):
        return self.data.element_type

    def to_numpy(self, *, copy: bool | None = None):
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        arr = self.data.to_numpy()
        if arr.dtype.names is None:
            return arr - 1
        out = arr.copy()
        for i, name in enumerate(arr.dtype.names):
            if self.index_fields is None or i in self.index_fields:
                out[name] -= 1
        return out

    @property
    def length_type(self):
        return self.data.length_type

    def load(self, idx: int):
        value = self.data.load(idx)
        if isinstance(value, tuple):
            return tuple(
                field - 1
                if self.index_fields is None or i in self.index_fields
                else field
                for i, field in enumerate(value)
            )
        return value - 1

    def store(self, idx: int, val):
        if isinstance(val, tuple):
            val = tuple(
                field + 1
                if self.index_fields is None or i in self.index_fields
                else field
                for i, field in enumerate(val)
            )
        else:
            val = val + 1
        self.data.store(idx, val)

    def resize(self, len: int):
        self.data.resize(len)


def buffer_to_jlobj(buffer: Buffer):
    if isinstance(buffer, PlusOneBuffer):
        return jl.PlusOneVector(buffer_to_jlobj(buffer.data))
    if isinstance(buffer, MinusOneBuffer):
        return buffer_to_jlobj(buffer.data)
    if isinstance(buffer, NumpyBuffer):
        return buffer.arr
    raise ValueError(f"Unsupported buffer type: {type(buffer)}")


def jlobj_to_buffer(jlobj, *, is_index_vector: bool = False):
    if isinstance(jlobj, jl.PlusOneVector):
        return PlusOneBuffer(jlobj_to_buffer(jlobj.data))
    if isinstance(jlobj, np.ndarray):
        buffer = NumpyBuffer(jlobj)
        return MinusOneBuffer(buffer) if is_index_vector else buffer
    raise ValueError(f"Unsupported Julia object type: {type(jlobj)}")
