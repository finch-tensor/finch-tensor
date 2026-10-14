from __future__ import annotations

from typing import Any

import numpy as np

import binsparse as bsp

from finch.algebra import StaticFill, ftype
from finch.codegen import NumpyBuffer
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    SparseCOOLevel,
    SparseListLevel,
    element,
)


def from_binsparse(array: bsp.BinsparseTensor) -> FiberTensor:
    """
    Convert a Binsparse tensor from the reference `binsparse` package into a
    Finch tensor which shares its buffers wherever possible.

    Dense levels become one Finch `DenseLevel` per dimension, rank-1 sparse
    levels become `SparseListLevel`, and higher-rank sparse levels become
    `SparseCOOLevel`.
    """
    if not isinstance(array, bsp.CustomTensor):
        container = bsp.InMemoryBinsparseContainer()
        array.serialize(container, alias=False)
        array = bsp.CustomTensor.parse(container)
    match array:
        case bsp.CustomTensor(
            shape=shape, transpose=None, level=bsp.BinsparseLevel() as level
        ):
            values = _element_values(level)
            fill_value = array.fill_value if array.fill is True else 0
            element_format = element(
                fill_type=StaticFillFType(values.dtype.type(fill_value)),
                element_type=ftype(values.dtype),
                position_type=ftype(_position_dtype(level)),
            )
            return FiberTensor(_from_level(level, shape, element_format, 0))
        case _:
            raise NotImplementedError(
                "Finch does not support transposed Binsparse tensors."
            )


def to_binsparse(tns: FiberTensor) -> bsp.CustomTensor:
    """Convert a Finch tensor to a Binsparse tensor which shares its buffers."""
    level, shape = _to_level(tns.lvl, 0)
    return bsp.CustomTensor(
        shape=shape,
        number_of_stored_values=_element_values(level).size,
        fill=True,
        fill_value=tns.fill_value,
        level=level,
    )


def _element_values(level: bsp.BinsparseLevel) -> np.ndarray:
    match level:
        case bsp.ElementLevel(values):
            return np.asarray(values)
        case bsp.DenseLevel(level=child) | bsp.SparseLevel(level=child):
            return _element_values(child)
        case _:
            raise TypeError(f"unsupported level type {type(level).__name__}")


def _position_dtype(level: bsp.BinsparseLevel) -> np.dtype:
    match level:
        case bsp.SparseLevel(pointers_to_next=pointers) if pointers is not None:
            return np.asarray(pointers).dtype
        case bsp.DenseLevel(level=child) | bsp.SparseLevel(level=child):
            return _position_dtype(child)
        case _:
            return np.dtype(np.intp)


def _from_level(
    level: bsp.BinsparseLevel,
    shape: tuple[int, ...],
    element_format: Any,
    depth: int,
) -> Any:
    match level:
        case bsp.ElementLevel(values):
            return ElementLevel(
                element_format, NumpyBuffer(np.ascontiguousarray(values))
            )
        case bsp.DenseLevel(rank=rank, level=child):
            lvl = _from_level(child, shape, element_format, depth + rank)
            for size in reversed(shape[depth : depth + rank]):
                lvl = DenseLevel(lvl, np.intp(size))
            return lvl
        case bsp.SparseLevel(
            rank=rank, level=child, indices=indices, pointers_to_next=pointers
        ):
            lvl = _from_level(child, shape, element_format, depth + rank)
            indices = tuple(np.ascontiguousarray(index) for index in indices)
            index_type = indices[0].dtype.type
            if pointers is None:
                pointers = np.array(
                    [0, indices[0].size], dtype=element_format.position_type
                )
            ptr = NumpyBuffer(np.ascontiguousarray(pointers))
            dimensions = tuple(index_type(size) for size in shape[depth : depth + rank])
            if rank == 1:
                return SparseListLevel(lvl, dimensions[0], ptr, NumpyBuffer(indices[0]))
            return SparseCOOLevel(
                lvl, dimensions, ptr, tuple(NumpyBuffer(index) for index in indices)
            )
        case _:
            raise TypeError(f"unsupported level type {type(level).__name__}")


def _to_level(lvl: Any, depth: int) -> tuple[bsp.BinsparseLevel, tuple[int, ...]]:
    match lvl:
        case ElementLevel():
            return bsp.ElementLevel(lvl.val.arr), ()
        case DenseLevel(child, dimension):
            level, shape = _to_level(child, depth + 1)
            return bsp.DenseLevel(1, level), (int(dimension), *shape)
        case SparseListLevel(child, dimension, ptr, idx):
            level, shape = _to_level(child, depth + 1)
            pointers = None if depth == 0 else ptr.arr
            return (
                bsp.SparseLevel(1, level, (idx.arr,), pointers),
                (int(dimension), *shape),
            )
        case SparseCOOLevel(child, coo_shape, ptr, tbl):
            rank = len(coo_shape)
            level, shape = _to_level(child, depth + rank)
            pointers = None if depth == 0 else ptr.arr
            return (
                bsp.SparseLevel(rank, level, tuple(idx.arr for idx in tbl), pointers),
                (*(int(size) for size in coo_shape), *shape),
            )
        case _:
            raise NotImplementedError(
                f"Finch level {type(lvl).__name__} has no Binsparse equivalent."
            )
