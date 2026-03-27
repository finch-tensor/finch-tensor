from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .julia import jl
from .typing import JuliaObj, number


# Abstract Formats
class LevelFormat:
    @property
    @abstractmethod
    def shape_type(self) -> tuple: ...


class NestedLevelFormat(LevelFormat):
    @property
    def ndim(self) -> np.intp:
        return self.lvl.ndim + np.intp(1)

    @property
    def fill_value(self) -> Any:
        return self.lvl.fill_value

    @property
    def element_type(self) -> Any:
        return self.lvl.element_type

    def __eq__(self, other):
        return type(other) is type(self) and self.lvl == other.lvl

    def __hash__(self):
        return hash((self.__class__.__name__, self.lvl.__hash__))

    @abstractmethod
    def create_jl_obj(self) -> JuliaObj: ...


class ElementFormat(LevelFormat):
    """Element level storage format for scalar tensor leaves.

    A subfiber of an element level is a scalar, initialized to a fill value.
    The element level is a leaf level used at the end of the tensor tree structure.
    """

    def __init__(self, fill_value: number):
        self._fill_value = fill_value

    @property
    def ndim(self) -> np.intp:
        return np.intp(0)

    @property
    def fill_value(self) -> Any:
        return self._fill_value

    @property
    def element_type(self) -> Any:
        return type(self._fill_value)

    def __eq__(self, other):
        return isinstance(other, ElementFormat) and self._fill_value == other.fill_value

    def __hash__(self):
        return hash((self.__class__.__name__, self._fill_value))

    def create_jl_obj(self) -> JuliaObj:
        return jl.Element(self._fill_value)

    def shape_type(self) -> tuple:
        return ()


@dataclass(frozen=True)
class DenseFormat(NestedLevelFormat):
    """Dense format wrapper type for Finch tensors.

    A dense format stores every slice of a tensor. A subfiber of a dense level
    is an array which stores every slice A[:, ..., :, i] as a distinct subfiber.
    Dense levels support both row-major and column-major access.
    """

    lvl: NestedLevelFormat
    dim_type: type = np.intp

    def create_jl_obj(self) -> JuliaObj:
        return jl.Dense(self.lvl.create_jl_obj())

    def shape_type(self) -> tuple:
        return self.lvl.shape_type + (self.dim_type,)


@dataclass(frozen=True)
class SparseListFormat(NestedLevelFormat):
    """Sparse list format wrapper type for Finch tensors.

    A sparse list format stores only potentially non-fill slices using a sorted list.
    Slices that are entirely fill_value are omitted. This format is efficient for
    tensors with sparse patterns and supports column-major reads and bulk updates.
    """

    lvl: NestedLevelFormat
    dim_type: type = np.intp

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseList(self.lvl.create_jl_obj())

    def shape_type(self) -> tuple:
        return self.lvl.shape_type + (self.dim_type,)


@dataclass(frozen=True)
class SparseCOOFormat(NestedLevelFormat):
    """Coordinate (COO) format wrapper type for Finch tensors.

    A coordinate format stores sparse tensors as lists of coordinates.
    It uses N separate arrays to record which coordinates are stored,
    with coordinates sorted in column-major order. This is a legacy format
    maintained for backward compatibility.
    """

    lvl: NestedLevelFormat
    N: int = 2
    dim_type: tuple | None = np.intp

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseCOO(self.lvl.create_jl_obj())

    def shape_type(self) -> tuple:
        if self.dim_type is None:
            return self.lvl.shape_type + (self.N * self.lvl.ndim,)
        return self.lvl.shape_type + self.dim_type


@dataclass(frozen=True)
class SparseByteMapFormat(NestedLevelFormat):
    """Sparse byte map format wrapper type for Finch tensors.

    A sparse byte map format uses a dense bitmap to encode which slices
    are stored, similar to SparseList but supporting random access.
    Only potentially non-fill slices are stored as subfibers.
    """

    lvl: NestedLevelFormat
    dim_type: type = np.intp

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseByteMap(self.lvl.create_jl_obj())

    def shape_type(self) -> tuple:
        return self.lvl.shape_type + (self.dim_type,)


# Helper Methods
def jlobj_to_format(obj: JuliaObj) -> LevelFormat:
    """Construct a level hierarchy from a Julia Finch tensor object.

    Recursively constructs a Python representation of the tensor's level structure
    by inspecting the Julia object's levels.

    Parameters
    ----------
    obj : JuliaObj
        A Julia Finch tensor object whose levels will be inspected.
    fill_value : float or int
        The fill value used for the tensor's sparse representation.

    Returns
    -------
    LevelFormat
        A Python representation of the level hierarchy.

    Raises
    ------
    Exception
        If an unsupported level type is encountered.
    """
    if jl.isa(obj, jl.Finch.Element):
        return ElementFormat(jl.fill_value(obj))
    if jl.isa(obj, jl.Finch.Dense):
        return DenseFormat(type(obj.shape), jlobj_to_format(obj.lvl))
    if jl.isa(obj, jl.Finch.SparseList):
        return SparseListFormat(type(obj.shape), jlobj_to_format(obj.lvl))
    if jl.isa(obj, jl.Finch.SparseByteMap):
        return SparseByteMapFormat(jlobj_to_format(obj.lvl))
    raise Exception("Unhandled exception!")
