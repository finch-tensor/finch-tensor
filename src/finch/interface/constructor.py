from __future__ import annotations

from typing import Any, overload

import numpy as np
import scipy.sparse as sps

import sparse as pydata_sparse

from finch.algebra import (
    StaticFillFType,
    Tensor,
    TensorFType,
    as_fill,
    ftype,
    normalize_device,
    np_dtype,
    resolve_fill,
)
from finch.autoschedule import get_default_scheduler
from finch.codegen.buffers.numpy_buffer import NumpyBuffer
from finch.finch_logic import (
    Field,
    HardAlias,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.symbolic import gensym
from finch.tensor import (
    BufferizedNDArray,
    FiberTensor,
    FiberTensorFType,
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
from finch.tensor.scalar import Scalar


def to_numpy(self) -> np.ndarray:
    def is_dense_chain(lvl) -> bool:
        match lvl:
            case ElementLevel():
                return True
            case DenseLevel(lvl=inner):
                return is_dense_chain(inner)
            case _:
                return False

    if is_dense_chain(self.lvl):
        flat = self.lvl.val.to_numpy()
        return np.reshape(flat, self.shape, copy=False)
    return self.to_scipy().toarray()


def to_scipy(self):
    if self.ndim != 2:
        raise ValueError("SciPy sparse arrays must be two-dimensional.")

    if self.fill_value != 0:
        raise ValueError("SciPy CSR conversion requires a zero fill value.")

    if self.pos != 0:
        raise ValueError("SciPy conversion require position 0, got {self.pos!r}")

    match self.lvl:
        case SparseCOOLevel(
            lvl=ElementLevel() as element,
            tbl=(row, col),
        ):
            return sps.coo_array(
                (element.val.to_numpy(), (row.to_numpy(), col.to_numpy())),
                shape=self.shape,
                copy=False,
            )

        case DenseLevel(
            lvl=SparseListLevel(
                lvl=ElementLevel() as element,
                ptr=ptr,
                idx=idx,
            )
        ):
            return sps.csr_array(
                (element.val.to_numpy(), idx.to_numpy(), ptr.to_numpy()),
                shape=self.shape,
                copy=False,
            )

        case _:
            raise NotImplementedError(
                f"Finch format {self.ftype} is not supported by SciPy conversion."
            )


def from_numpy(arr: np.ndarray, *, device=None, copy=None) -> BufferizedNDArray:
    """Zero-copy constructor from a NumPy array."""
    if copy is True:
        arr = arr.copy()
    return BufferizedNDArray.from_numpy(arr, device=device)


def from_scipy(obj: sps.spmatrix | sps.sparray, *, device=None, dtype=None, copy=None):
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
            return from_parts_csr(
                data=obj.data,
                indices=obj.indices,
                indptr=obj.indptr,
                shape=obj.shape,
                device=device,
            )
        case "coo":
            return from_parts_coo(
                np.stack([obj.row, obj.col]),
                obj.data,
                obj.shape,
                device=device,
            )
    raise ValueError(f"Unexpected SciPy format after normalization: {obj.format}")


def from_pydata(
    obj: Any,
    *,
    device=None,
    dtype=None,
    copy=None,
) -> FiberTensor:
    """Zero-copy constructor from a pydata/sparse.COO array."""
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

    return from_parts_coo(obj.coords, obj.data, obj.shape, device=device)


def from_parts_csr(data, indices, indptr, shape, *, device=None) -> FiberTensor:
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


def from_parts_coo(coords, data, shape, *, device=None) -> FiberTensor:
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


@overload
def asarray(
    obj: np.ndarray,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: None = None,
) -> BufferizedNDArray: ...


@overload
def asarray(
    obj: sps.spmatrix | sps.sparray,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: None = None,
) -> FiberTensor: ...


@overload
def asarray(
    obj: Any,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: FiberTensorFType,
) -> FiberTensor: ...


@overload
def asarray(
    obj: np.generic | complex,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: None = None,
) -> Scalar: ...


@overload
def asarray(
    obj: Any,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: TensorFType | None = None,
) -> Tensor: ...


def asarray(
    obj: Any,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: TensorFType | None = None,
) -> Tensor:
    """
    Convert given argument and return wrapper type instance.
    If input argument is already array type, return unchanged.
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.asarray.html
    """
    from finch.interface.fuse import compute
    from finch.interface.lazy import defer

    explicit_device = device is not None
    device = normalize_device(device)

    if isinstance(obj, Scalar):
        if copy is True or (explicit_device and obj.device != device):
            tensor = Scalar(obj.val, fill_value=obj.fill_value, device=device)
        else:
            tensor = obj
    elif isinstance(obj, BufferizedNDArray):
        if copy is True:
            tensor = BufferizedNDArray.from_numpy(
                obj.to_numpy().copy(), fill_value=obj.fill_value, device=device
            )
        elif explicit_device and obj.device != device:
            tensor = obj.to_device(device)
        else:
            tensor = obj
    elif isinstance(obj, Tensor) and copy is not True:
        if explicit_device and obj.device != device:
            tensor = obj.to_device(device)
        else:
            tensor = obj
    elif isinstance(obj, np.ndarray):
        tensor = from_numpy(obj, device=device, copy=copy)
    elif sps.issparse(obj):
        tensor = from_scipy(obj, device=device, dtype=dtype, copy=copy)
    elif isinstance(obj, pydata_sparse.COO):
        tensor = from_pydata(obj, device=device, dtype=dtype, copy=copy)

    elif np.isscalar(obj) or obj is None:
        if dtype is not None:
            obj = ftype(dtype)(obj)
        elif isinstance(obj, bool | int | float | complex):
            obj = np.asarray(obj)[()]
        tensor = Scalar(obj, device=device)
    else:
        try:
            np_arr = np.asarray(obj)
            if np_arr.dtype != object:
                if dtype is not None:
                    np_arr = np_arr.astype(np_dtype(ftype(dtype)))
                elif copy is True:
                    np_arr = np_arr.copy()
                tensor = BufferizedNDArray.from_numpy(np_arr, device=device)
            else:
                raise TypeError(
                    f"Cannot convert {type(obj).__name__} to a Finch tensor"
                )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot convert {type(obj).__name__} to a Finch tensor"
            ) from exc

    if format is not None:
        lazy = reformat(defer(tensor), format, device=device)
        tensor = compute(lazy)
    return tensor.to_device(device) if explicit_device else tensor


def reformat(tensor, format, *, device=None):
    from finch.interface.lazy import (
        LazyTensor,
        defer,
    )

    if tensor.ftype == format:
        return tensor

    if not isinstance(tensor, LazyTensor):
        tensor = defer(tensor)

    out_alias = HardAlias(gensym("A"))
    idxs = tuple(Field(gensym("i")) for _ in range(tensor.ndim))
    query = Query(Table(out_alias, idxs), Table(tensor.data, idxs))
    plan = Plan((*tensor.ctx.trace(), query, Produces((out_alias,))))

    placeholder = format.construct(tensor.shape, fill_value=as_fill(tensor.fill_value))
    bindings = {out_alias: placeholder}

    scheduler = get_default_scheduler()
    results = scheduler(plan, bindings)

    result = results[-1]
    fill_type = getattr(format, "fill_type", None)
    if isinstance(fill_type, StaticFillFType):
        target_fill = resolve_fill(fill_type)
        if hasattr(result, "with_fill") and result.fill_value != target_fill:
            result = result.with_fill(target_fill)
    return result.to_device(device) if device is not None else result
