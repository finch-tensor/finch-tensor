from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps

from finch.algebra import (
    ftype,
)
from finch.codegen.buffers.numpy_buffer import NumpyBuffer
from finch.tensor import (
    BufferizedNDArray,
    FiberTensor,
)
from finch.tensor.level import (
    DenseLevel,
    ElementLevel,
    SparseCOOLevel,
    SparseListLevel,
    dense,
    element,
    sparse_list,
)


def _from_numpy(arr: np.ndarray, *, device=None, copy=None) -> BufferizedNDArray:
    """Zero-copy constructor from a NumPy array."""
    if copy is True:
        arr = arr.copy()
    return BufferizedNDArray.from_numpy(arr, device=device)


def _from_scipy(obj: sps.spmatrix | sps.sparray, *, device=None, dtype=None, copy=None):
    """Zero-copy constructor from a SciPy sparse matrix or array."""
    obj: Any = obj
    if obj.format == "csc":  # CSC has no native Finch layout yet
        if dtype is not None or copy is True:
            raise NotImplementedError(
                "asarray cannot honor dtype or copy for CSC input; "
                "convert to CSR or COO first."
            )
        return obj
    if copy is False and not (
        obj.format in ("coo", "csr") and obj.has_canonical_format
    ):
        raise ValueError("Unable to avoid copy while creating an array as requested.")
    if obj.format not in ("coo", "csr"):
        obj = obj.asformat("coo")

    if dtype is not None:
        if copy is False:
            obj = obj.astype(dtype, copy=False)
        else:
            obj = obj.astype(dtype, copy=True)
    elif copy is True:
        obj = obj.copy()

    if not obj.has_canonical_format:
        obj = obj.copy()
        obj.sum_duplicates()

    match obj.format:
        case "csr":
            return FiberTensor.from_scipy_csr(obj, device=device)
        case "coo":
            return FiberTensor.from_scipy_coo(obj, device=device)
    raise ValueError(f"Unexpected SciPy format after normalization: {obj.format}")


def _is_pydata_sparse(obj: Any) -> bool:
    return (
        type(obj).__module__.split(".")[0] == "sparse"
        and hasattr(obj, "coords")
        and hasattr(obj, "data")
        and hasattr(obj, "shape")
    )


def _from_pydata(
    obj: Any,
    *,
    device=None,
    dtype=None,
    copy=None,
) -> FiberTensor:
    """Zero-copy constructor from a pydata/sparse.COO array."""
    try:
        import sparse as pydata_sparse
    except ImportError as exc:
        raise ImportError(
            "pydata/sparse is not installed. Please install it to use this feature."
        ) from exc
    if not isinstance(obj, pydata_sparse.COO):
        raise TypeError(
            f"Expected pydata/sparse.COO, got {type(obj).__name__}. "
            "GCXS and other pydata formats are not supported."
        )

    if dtype is not None:
        if copy is False:
            obj = obj.astype(dtype, copy=False)
        else:
            obj = obj.astype(dtype, copy=True)
    elif copy is True:
        obj = obj.copy()

    return FiberTensor.from_pydata_coo(obj, device=device)


def _from_parts_csr(data, indices, indptr, shape, *, device=None) -> FiberTensor:
    """
    Constructs a FiberTensor from CSR parts.
    """
    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape for CSR parts, got {shape}.")
    if len(indptr) != shape[0] + 1:
        raise ValueError(
            f"Expected indptr length {shape[0] + 1} for CSR parts, got {len(indptr)}."
        )

    if len(indices) != len(data):
        raise ValueError(
            f"Expected indices length {len(data)} for CSR parts, got {len(indices)}."
        )
    element_lvl = element(
        fill_type=np.zeros((), dtype=data.dtype)[()],
        element_type=ftype(data.dtype),
        position_type=ftype(indices.dtype),
    )
    sparse_lvl = sparse_list(element_lvl, dimension_type=ftype(indptr.dtype))
    dense_lvl = dense(sparse_lvl, dimension_type=ftype(indices.dtype))
    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(
                    _format=element_lvl,
                    _val=NumpyBuffer(data),
                ),
                dimension=sparse_lvl.dimension_type(shape[1]),
                ptr=NumpyBuffer(indptr),
                idx=NumpyBuffer(indices),
            ),
            dimension=dense_lvl.dimension_type(shape[0]),
        ),
        _device=device,
    )


def _from_parts_coo(coords, data, shape, *, device=None) -> FiberTensor:
    """
    Constructs a FiberTensor from COO parts.
    """
    if coords.ndim != 2:
        raise ValueError(f"Expected 2D coords for COO parts, got {coords.ndim}D.")
    ndim = coords.shape[0]
    if len(shape) != ndim:
        raise ValueError(f"Expected {ndim}D shape for COO parts, got {shape}.")
    if coords.shape[1] != len(data):
        raise ValueError(
            f"Expected coords second dimension {len(data)} "
            f"for COO parts, got {coords.shape[1]}."
        )
    index_type = ftype(coords.dtype)

    element_lvl = element(
        fill_type=np.zeros((), dtype=data.dtype)[()],
        element_type=ftype(data.dtype),
        position_type=index_type,
    )
    tbl = tuple(NumpyBuffer(coords[d]) for d in range(ndim))

    return FiberTensor(
        SparseCOOLevel(
            ElementLevel(
                _format=element_lvl,
                _val=NumpyBuffer(data),
            ),
            shape=tuple(index_type(d) for d in shape),
            tbl=tbl,
        ),
        _device=device,
    )


# def _reformat(tensor: LazyTensor, format: TensorFType) -> LazyTensor:
#     """Reformat a tensor to a different format using FinchLogic query."""
#     """
#     if isinstance(tensor, FiberTensor) and tensor.ftype == format:
#         return tensor
#     if isinstance(tensor, BufferizedNDArray) and tensor.ftype == format:
#         return tensor

#     if isinstance(tensor, FiberTensor):
#         try:
#             source_data = tensor.to_scipy()
#         except NotImplementedError:
#             source_data = tensor.to_numpy()
#     elif isinstance(tensor, BufferizedNDArray):
#         source_data = tensor.to_numpy()
#     else:
#         source_data = np.asarray(tensor)

#     if isinstance(format, FiberTensorFType):
#         if sps.issparse(source_data):
#             return FiberTensor.from_scipy_csr(source_data)
#         scipy_mat = sps.csr_matrix(source_data)
#         return FiberTensor.from_scipy_csr(scipy_mat)

#     return BufferizedNDArray.from_numpy(np.asarray(source_data))
#     """
#     idxs = tuple(Field(gensym("i")) for _ in range(tensor.ndim))
#     data, ctx = tensor.ctx.eval(Table(tensor.data, idxs))
#     return LazyTensor(
#         data,
#         ctx,
#         tensor.shape,
#         tensor.fill_value,
#         tensor.element_type,
#         tensor.device,
#     )
