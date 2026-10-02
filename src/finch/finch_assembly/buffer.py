from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from finch.algebra import FType, FTyped, intp
from finch.algebra.ftypes import FDTypeInteger

FT = TypeVar("FT", bound=FType)


class BufferFType(FType, Generic[FT]):
    """
    Abstract base class for the ftype of arguments. The ftype defines how the
    data structures store data, and can construct a data structure with the call method.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Create an instance of an object in this ftype with the given arguments.
        """
        ...

    @property
    @abstractmethod
    def element_type(self) -> FT:
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        ...

    @property
    def length_type(self) -> FDTypeInteger:
        """
        Returns the type used for the length of the buffer.
        """
        return intp


class Buffer(FTyped[BufferFType[FT]], ABC):
    """
    Abstract base class for buffer-like data structures. Buffers support random access,
    and can be resized. They are used to store data in a way that allows for efficient
    reading and writing of elements.
    """

    @abstractmethod
    def __init__(self, length: int, dtype: type): ...

    @abstractmethod
    def length(self):
        """
        Return the length of the buffer.
        """
        ...

    @property
    def element_type(self) -> FT:
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self.ftype.element_type

    @property
    def length_type(self) -> FDTypeInteger:
        """
        Return the type of indices used to access elements in the buffer.
        This is typically an integer type.
        """
        return self.ftype.length_type

    @abstractmethod
    def load(self, idx: int): ...

    @abstractmethod
    def store(self, idx: int, val): ...

    @abstractmethod
    def resize(self, len: int):
        """
        Resize the buffer to the new length.
        """
        ...


def length_type(arg: FTyped | FType) -> FDTypeInteger:
    """The length type of the given argument. The length type is the type of
    the value returned by len(arg).

    Args:
        arg: The object to determine the length type for.

    Returns:
        The length type of the given object.

    Raises:
        AttributeError: If the length type is not implemented for the given type.
    """
    length_type: FDTypeInteger | None = getattr(arg, "length_type", None)
    if length_type is None:
        raise AttributeError(f"{type(arg).__name__} has no length_type")
    return length_type


def element_type(arg: Buffer) -> FType:
    return arg.element_type
