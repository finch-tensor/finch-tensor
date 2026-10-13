from abc import ABC
from typing import TypeVar

import numpy as np

from finch.algebra.ftypes import FType
from finch.codegen import NumpyBuffer
from finch.finch_assembly import Buffer, BufferFType

from .julia import jl

FT = TypeVar("FT", bound=FType)


class PlusOneBufferFType(BufferFType[FT]):
    def __init__(self, data_ftype: BufferFType[FT]):
        self.data_ftype = data_ftype

    def __call__(self, *args, **kwargs):
        return PlusOneBuffer(self.data_ftype(*args, **kwargs))

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
        )

    def __hash__(self):
        return hash(self.data_ftype)


class PlusOneBuffer(Buffer, ABC):
    """
    Buffer that adds one to each element when loaded and subtracts one from each
    element when stored.
    """

    def __init__(self, data: NumpyBuffer):
        self.data = data

    def ftype(self):
        return PlusOneBufferFType(self.data.ftype())

    def length(self):
        return self.data.length()

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
        return self.data.load(idx) + 1

    def store(self, idx: int, val):
        self.data.store(idx, val - 1)

    def resize(self, len: int):
        self.data.resize(len)


class MinusOneBufferFType(BufferFType[FT]):
    def __init__(self, data_ftype: BufferFType[FT]):
        self.data_ftype = data_ftype

    def __call__(self, *args, **kwargs):
        return MinusOneBuffer(self.data_ftype(*args, **kwargs))

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
        )

    def __hash__(self):
        return hash(self.data_ftype)


class MinusOneBuffer(Buffer, ABC):
    """Buffer that subtracts one on loads and adds one on stores."""

    def __init__(self, data: NumpyBuffer):
        self.data = data

    @property
    def ftype(self):
        return MinusOneBufferFType(self.data.ftype)

    def length(self):
        return self.data.length()

    @property
    def element_type(self):
        return self.data.element_type

    @property
    def arr(self):
        return self.data.arr - 1

    @property
    def length_type(self):
        return self.data.length_type

    def load(self, idx: int):
        return self.data.load(idx) - 1

    def store(self, idx: int, val):
        self.data.store(idx, val + 1)

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
