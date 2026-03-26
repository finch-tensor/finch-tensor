from abc import abstractmethod
from typing import Any

import numpy as np

from finchlite import Tensor, TensorFType, Buffer
from .buffer import PlusOneBuffer, NumpyBuffer, buffer_to_jlobj, jlobj_to_buffer

from .julia import jl
from .typing import JuliaObj, number
from dataclasses import dataclass


# Abstract FTypes
class LevelFType(TensorFType):
    def from_numpy(self, _) -> Tensor:
        raise NotImplementedError

    @property
    def shape_type(self) -> tuple[type, ...]:
        return tuple(self.element_type for _ in range(self.ndim))

    def __call__(self, _) -> Tensor:
        raise Exception("Cannot create an object of this type!")



class NestedLevelFType(LevelFType):
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


class ElementFType(LevelFType):
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
        return isinstance(other, ElementFType) and self._fill_value == other.fill_value

    def __hash__(self):
        return hash((self.__class__.__name__, self._fill_value))

    def create_jl_obj(self) -> JuliaObj:
        return jl.Element(self._fill_value)



@dataclass(frozen=True)
class DenseFType(NestedLevelFType):
    """Dense format wrapper type for Finch tensors.
    
    A dense format stores every slice of a tensor. A subfiber of a dense level
    is an array which stores every slice A[:, ..., :, i] as a distinct subfiber.
    Dense levels support both row-major and column-major access.
    """
    lvl: NestedLevelFType

    def create_jl_obj(self) -> JuliaObj:
        return jl.Dense(self.lvl.create_jl_obj())
    
@dataclass(frozen=True)
class SparseListFType(NestedLevelFType):
    """Sparse list format wrapper type for Finch tensors.
    
    A sparse list format stores only potentially non-fill slices using a sorted list.
    Slices that are entirely fill_value are omitted. This format is efficient for
    tensors with sparse patterns and supports column-major reads and bulk updates.
    """
    lvl: NestedLevelFType

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseList(self.lvl.create_jl_obj())

@dataclass(frozen=True)
class SparseCOOFType(NestedLevelFType):
    """Coordinate (COO) format wrapper type for Finch tensors.
    
    A coordinate format stores sparse tensors as lists of coordinates.
    It uses N separate arrays to record which coordinates are stored,
    with coordinates sorted in column-major order. This is a legacy format
    maintained for backward compatibility.
    """
    lvl: NestedLevelFType
    N: int = 2
    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseCOO(self.lvl.create_jl_obj())

@dataclass(frozen=True)
class SparseByteMapFType(NestedLevelFType):
    """Sparse byte map format wrapper type for Finch tensors.
    
    A sparse byte map format uses a dense bitmap to encode which slices
    are stored, similar to SparseList but supporting random access.
    Only potentially non-fill slices are stored as subfibers.
    """
    lvl: NestedLevelFType

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseByteMap(self.lvl.create_jl_obj())


class _Display:
    _obj: JuliaObj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)


# LEVEL


class AbstractLevel(_Display):
    pass


# core levels


class Dense(AbstractLevel):
    """Dense level storage format.
    
    A subfiber of a dense level is an array which stores every slice as a distinct
    subfiber in the child level. Dense levels support efficient random access and
    both column-major and out-of-order updates.
    
    Parameters
    ----------
    lvl : AbstractLevel
        The child level that will store each slice.
    shape : int, optional
        The size of the dimension at this level. If not provided, the shape
        is inferred from the child level.
    
    Examples
    --------
    Create a 2D dense tensor:
    
    >>> dense_2d = Dense(Dense(Element(0.0)))
    """
    def __init__(self, lvl, shape=None):
        args = [lvl._obj]
        if shape is not None:
            args.append(shape)
        self._obj = jl.Dense(*args)


class Element(AbstractLevel):
    """Element level storage format (leaf level).
    
    A subfiber of an element level is a scalar of a specified type, initialized
    to a fill value. Element levels form the leaf nodes of the tensor tree and
    store the actual data values.
    
    Parameters
    ----------
    fill_value : float or int
        The default fill value for elements that are not explicitly set.
    data : array-like, optional
        Optional vector to store the element data. If not provided, a new
        vector is created.
    
    Examples
    --------
    Create an element level with zero fill value:
    
    >>> elem = Element(0.0)
    """
    def __init__(self, fill_value, data: Buffer | None=None):
        args = [fill_value]
        if data is not None:
            args.append(buffer_to_jlobj(data))
        self._obj = jl.Element(*args)

    @property
    def data(self) -> Buffer:
        """Return the data buffer for this element level."""
        return jlobj_to_buffer(self._obj.data)


class Pattern(AbstractLevel):
    """Pattern level storage format (leaf level).
    
    A subfiber of a pattern level is the boolean value true, but with a fill
    value of false. Pattern levels are used to create tensors representing
    which values are stored by other fibers, allowing for tracking sparsity
    structure without storing values.
    
    Examples
    --------
    Create a pattern level to track which elements are nonzero:
    
    >>> pattern = Pattern()
    """
    def __init__(self):
        self._obj = jl.Pattern()


# advanced levels


class SparseList(AbstractLevel):
    """Sparse list level storage format.
    
    A subfiber of a sparse list level stores only potentially non-fill slices
    using a sorted list to record which slices are stored. Slices that are
    entirely fill_value are omitted. This format is efficient for sparse tensors
    and supports column-major reads and bulk updates, but does not support
    random access or random updates.
    
    Parameters
    ----------
    lvl : AbstractLevel
        The child level that will store each non-fill slice.
    dim : int, optional
        The size of the dimension at this level. If not provided, the dimension
        is inferred from the child level.
    ptr : array-like, optional
        Array of positions/pointers for the sparse list. If not provided,
        will be created internally. Converted to ndarray if provided.
    idx : array-like, optional
        Array of indices for the sparse list. If not provided,
        will be created internally. Converted to ndarray if provided.
    
    Examples
    --------
    Create a sparse matrix in CSC format:
    
    >>> sparse_matrix = SparseList(SparseList(Element(0.0)))
    """
    def __init__(self, lvl, dim=None, ptr=None, idx=None):
        args = [lvl._obj]
        if dim is not None:
            args.append(dim)
        if ptr is not None and idx is not None:
            args.append(buffer_to_jlobj(ptr))
            args.append(buffer_to_jlobj(idx))
        self._obj = jl.SparseList(*args)

    @property
    def ptr(self) -> Buffer:
        """Return the pointer array for this sparse list level."""
        return jlobj_to_buffer(self._obj.ptr)

    @property
    def idx(self) -> Buffer:
        """Return the index array for this sparse list level."""
        return jlobj_to_buffer(self._obj.idx)

class SparseByteMap(AbstractLevel):
    """Sparse byte map level storage format.
    
    Similar to SparseList, but uses a dense bitmap to encode which slices are
    stored instead of a sparse index list. This allows for efficient random
    access while still omitting fill_value slices. The byte map approach trades
    memory for faster lookups.
    
    Parameters
    ----------
    lvl : AbstractLevel
        The child level that will store each non-fill slice.
    dim : int, optional
        The size of the dimension at this level. If not provided, the dimension
        is inferred from the child level.
    
    Examples
    --------
    Create a sparse matrix with byte map storage:
    
    >>> sparse_matrix = SparseByteMap(SparseByteMap(Element(0.0)))
    """
    def __init__(self, lvl, dim=None):
        args = [lvl._obj]
        if dim is not None:
            args.append(dim)
        self._obj = jl.SparseByteMap(*args)


class RepeatRLE(AbstractLevel):
    """Run-length encoding level for repeated values.
    
    This level stores runs of repeated values, useful for data with
    long sequences of identical values.
    
    Parameters
    ----------
    lvl : AbstractLevel
        The child level to store run information.
    dim : int, optional
        The size of the dimension at this level. If not provided, the dimension
        is inferred from the child level.
    """
    def __init__(self, lvl, dim=None):
        args = [lvl._obj]
        if dim is not None:
            args.append(dim)
        self._obj = jl.RepeatRLE(*args)


class SparseVBL(AbstractLevel):
    """Sparse variable block list level storage format.
    
    Like SparseList, but stores contiguous slices together in blocks.
    This can improve cache locality and performance for certain access patterns.
    
    Parameters
    ----------
    lvl : AbstractLevel
        The child level that will store each block of slices.
    dim : int, optional
        The size of the dimension at this level. If not provided, the dimension
        is inferred from the child level.
    """
    def __init__(self, lvl, dim=None):
        args = [lvl._obj]
        if dim is not None:
            args.append(dim)
        self._obj = jl.SparseVBL(*args)


class SparseCOO(AbstractLevel):
    """Coordinate (COO) sparse format level storage.
    
    This level stores sparse data using N coordinate lists (one per dimension).
    Coordinates are stored in column-major order. This is a legacy format
    maintained for backward compatibility; consider using other sparse formats
    for new code.
    
    Parameters
    ----------
    ndim : int
        Number of dimensions for the coordinate format.
    lvl : AbstractLevel
        The child level to store coordinate data.
    dims : tuple of int, optional
        Sizes of the last N dimensions. If not provided, dimensions are
        inferred from the child level.
    tbl : tuple of arrays, optional
        A tuple of coordinate arrays (one per dimension). Each array stores
        the coordinates for that dimension. Arrays are converted to ndarray
        if provided. If not provided, will be created internally.
    
    Examples
    --------
    Create a 2D sparse tensor in COO format:
    
    >>> coo_2d = SparseCOO(2, Element(0.0))
    """
    def __init__(self, ndim, lvl, dims=None, tbl : tuple[Buffer] | None =None):
        args = [lvl._obj]
        if dims is not None:
            if isinstance(dims, (list, tuple)):
                args.extend(dims)
            else:
                args.append(dims)
        if tbl is not None:
            args.extend([buffer_to_jlobj(t) for t in tbl])
        self._obj = jl.SparseCOO[ndim](*args)
    
    @property
    def tbl(self) -> tuple[Buffer, ...]:
        """Return the coordinate array tuple for this COO level."""
        return tuple(jlobj_to_buffer(coord) for coord in self._obj.tbl)


class SparseHash(AbstractLevel):
    """Hash table based sparse format level storage.
    
    Uses a hash table to store sparse data, supporting efficient random access
    and random updates. This format is useful when you need flexible out-of-order
    insertion of elements.
    
    Parameters
    ----------
    ndim : int
        Number of dimensions for the hash format.
    lvl : AbstractLevel
        The child level to store hash table data.
    dims : tuple of int, optional
        Sizes of the last N dimensions. If not provided, dimensions are
        inferred from the child level.
    
    Examples
    --------
    Create a 2D sparse tensor using hash storage:
    
    >>> hash_2d = SparseHash(2, Element(0.0))
    """
    def __init__(self, ndim, lvl, dims=None):
        args = [lvl._obj]
        if dims is not None:
            if isinstance(dims, (list, tuple)):
                args.extend(dims)
            else:
                args.append(dims)
        self._obj = jl.SparseHash[ndim](*args)


# Helper Methods
def construct_levels(obj: JuliaObj, fill_value: number) -> LevelFType:
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
    LevelFType
        A Python representation of the level hierarchy.
    
    Raises
    ------
    Exception
        If an unsupported level type is encountered.
    """
    if jl.isa(obj.lvl, jl.Finch.Element):
        return Element(fill_value)
    if jl.isa(obj.lvl, jl.Finch.Dense):
        return Dense(construct_levels(obj.lvl, fill_value))
    if jl.isa(obj.lvl, jl.Finch.SparseList):
        return SparseList(construct_levels(obj.lvl, fill_value))
    if jl.isa(obj.lvl, jl.Finch.SparseByteMap):
        return SparseByteMap(construct_levels(obj.lvl, fill_value))
    raise Exception("Unhandled exception!")
