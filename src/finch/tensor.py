from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Literal

import numpy as np
from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple

from . import dtypes as jl_dtypes
from .errors import PerformanceWarning
from .julia import jc, jl
from .levels import (
    Dense,
    DenseStorage,
    Element,
    SparseCOO,
    SparseList,
    Storage,
    _Display,
    sparse_formats_names,
)
from .typing import Device, DType, JuliaObj, OrderType, TupleOf3Arrays, spmatrix
from finchlite import Tensor, TensorFType


class SparseArray:
    """
    PyData/Sparse marker class
    """

class FinchJLTensorFType(TensorFType):
    def __init__(self, jltype):
        self.jltype = jltype

    def ndims(self) -> int:
        return jl.ndims(self.jltype)
    
    def element_type(self):
        return jl.eltype(self.jltype)
    
    ...

class FinchJLTensor(_Display, SparseArray, Tensor, finchlite.EagerTensor):
    """
    A wrapper class for Finch.Tensor and Finch.SwizzleArray.

    Constructors
    ------------
    FinchJLTensor(scipy.sparse.spmatrix)
        Construct a Tensor out of a `scipy.sparse` object. Supported formats are: `COO`,
        `CSC`, and `CSR`.
    FinchJLTensor(numpy.ndarray)
        Construct a Tensor out of a NumPy array object. This is a no-copy operation.
    FinchJLTensor(Storage)
        Initialize a Tensor with a `storage` description. `storage` can already hold
        data.
    FinchJLTensor(julia_object)
        Tensor created from a compatible raw Julia object. Must be a `Tensor`.
        This is a no-copy operation.

    Parameters
    ----------
    obj : np.ndarray or scipy.sparse or Storage or Finch.Tensor
        Input to construct a Tensor. It's a no-copy operation of for NumPy and
        SciPy input. For Storage it's levels' description with order. The order
        numbers the dimensions from the fastest to slowest.  The leaf nodes have
        mode `0` and the root node has mode `n-1`. If the tensor was square of
        size `N`, then `N .^ order == strides`. Available options are "C"
        (row-major), "F" (column-major), or a custom order. Default: row-major.
    fill_value : np.number, optional
        Only used when `numpy.ndarray` or `scipy.sparse` is passed.
    copy : bool, optional
        If ``True``, then the object is copied. If ``None`` then the object is
        copied only if needed.  For ``False`` it raises a ``ValueError`` if a
        copy cannot be avoided. Default: ``None``.

    Returns
    -------
    FinchJLTensor
        Python wrapper for Finch.jl `Tensor`.

    Examples
    --------
    >>> import numpy as np
    >>> import finch
    >>> arr2d = np.arange(6).reshape((2, 3))
    >>> t1 = finch.FinchJLTensor(arr2d)
    >>> t1.todense()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.shares_memory(t1.todense(), arr2d)
    True
    >>> storage = finch.Storage(
    ...     finch.Dense(finch.SparseList(finch.Element(1))), order="C"
    ... )
    >>> t2 = t1.to_storage(storage)
    >>> t2.todense()
    array([[0, 1, 2],
           [3, 4, 5]])
    """

    def __init__(
        self,
        obj: np.ndarray | spmatrix | Storage | JuliaObj,
        /,
        *,
        fill_value: np.number | None = None,
        copy: bool | None = None,
    ):
        if isinstance(obj, int | float | complex | bool | list):
            if copy is False:
                raise ValueError(
                    "copy=False isn't supported for scalar inputs and Python lists"
                )
            obj = np.asarray(obj)
        if fill_value is None:
            fill_value = 0.0

        if _is_scipy_sparse_obj(obj):  # scipy constructor
            jl_data = self._from_scipy_sparse(obj, fill_value=fill_value, copy=copy)
            self._obj = jl_data
        elif isinstance(obj, np.ndarray):  # numpy constructor
            jl_data = self._from_numpy(obj, fill_value=fill_value, copy=copy)
            self._obj = jl_data
        elif isinstance(obj, Storage):  # from-storage constructor
            if copy:
                self._raise_julia_copy_not_supported()
            order = self.preprocess_order(
                obj.order, self.get_lvl_ndim(obj.levels_descr._obj)
            )
            self._obj = jl.swizzle(jl.Tensor(obj.levels_descr._obj), *order)
        elif jl.isa(obj, jl.Finch.Tensor):  # raw-Julia-object constructors
            if copy:
                self._raise_julia_copy_not_supported()
            self._obj = jl.swizzle(obj, *tuple(range(1, jl.ndims(obj) + 1)))
        elif jl.isa(obj, jl.Finch.Tensor):
            if copy:
                self._raise_julia_copy_not_supported()
            self._obj = obj
        elif isinstance(obj, FinchJLTensor):
            self._obj = obj._obj
        else:
            raise ValueError(
                "Either scalar, numpy, scipy.sparse or a raw julia object should "
                f"be provided. Found: {type(obj)}"
            )

    @property
    def element_type(self):
        return jl.eltype(self._obj.body)

    @property
    def dtype(self) -> DType:
        return jl.eltype(self._obj.body)

    @property
    def ndim(self) -> int:
        return jl.ndims(self._obj)

    @property
    def shape(self) -> tuple[int, ...]:
        return jl.size(self._obj)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def fill_value(self) -> np.number:
        return jl.fill_value(self._obj)

    @property
    def _is_dense(self) -> bool:
        lvl = self._obj.body.lvl
        for _ in self.shape:
            if not jl.isa(lvl, jl.Finch.Dense):
                return False
            lvl = lvl.lvl
        return True

    @property
    def _order(self) -> tuple[int, ...]:
        return jl.typeof(self._obj).parameters[1]

    @property
    def mT(self) -> Tensor:
        axes = list(range(self.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        axes = tuple(axes)
        return self.permute_dims(axes)

    @property
    def device(self) -> str:
        return "cpu"

    def to_device(
        self, device: Device, /, *, stream: int | Any | None = None
    ) -> Tensor:
        if device != "cpu":
            raise ValueError("Only `device='cpu'` is supported.")

        return self

    @classmethod
    def get_lvl_ndim(cls, lvl: JuliaObj) -> int:
        ndim = 0
        while True:
            ndim += 1
            lvl = lvl.lvl
            if jl.isa(lvl, jl.Finch.Element):
                break
        return ndim

    def todense(self) -> np.ndarray:
        obj = self._obj

        if self._is_dense:
            # don't materialize a dense finch tensor
            shape = jl.size(obj.body)
            dense_tensor = obj.body.lvl
        else:
            # create materialized dense array
            shape = jl.size(obj)
            dense_lvls = jl.Element(jc.convert(self.dtype, jl.fill_value(obj)))
            for _ in range(self.ndim):
                dense_lvls = jl.Dense(dense_lvls)
            dense_tensor = jl.Tensor(dense_lvls, obj).lvl  # materialize

        for _ in range(self.ndim):
            dense_tensor = dense_tensor.lvl

        result = np.asarray(jl.reshape(dense_tensor.val, shape))
        return result.transpose(self.get_order()) if self._is_dense else result

    #TODO: Do we need?
    def permute_dims(self, axes: tuple[int, ...]) -> Tensor:
        axes = tuple(i + 1 for i in axes)
        new_obj = jl.permutedims(self._obj, axes)
        return Tensor(new_obj)

    def to_storage(self, storage: Storage) -> Tensor:
        return Tensor(self._from_other_tensor(self, storage=storage))

    @classmethod
    def _from_other_tensor(cls, tensor: Tensor, storage: Storage) -> JuliaObj:
        order = cls.preprocess_order(storage.order, tensor.ndim)
        result = jl.copyto_b(
            jl.swizzle(jl.Tensor(storage.levels_descr._obj), *order), tensor._obj
        )
        return jl.dropfills(result) if tensor._is_dense else result

    @classmethod
    def _from_numpy(
        cls, arr: np.ndarray, fill_value: np.number, copy: bool | None = None
    ) -> JuliaObj:
        if copy:
            arr = arr.copy()
        order_char = "F" if np.isfortran(arr) else "C"
        order = cls.preprocess_order(order_char, arr.ndim)
        inv_order = tuple(i - 1 for i in jl.invperm(order))

        dtype = arr.dtype.type
        if (
            dtype == np.bool_
        ):  # Fails with: Finch currently only supports isbits defaults
            dtype = jl_dtypes.bool
        fill_value = dtype(fill_value)
        lvl = Element(fill_value, arr.reshape(-1, order=order_char))
        for i in inv_order:
            lvl = Dense(lvl, arr.shape[i])
        return jl.swizzle(jl.Tensor(lvl._obj), *order)

    @classmethod
    def from_scipy_sparse(
        cls,
        x,
        fill_value: np.number | None = None,
        copy: bool | None = None,
    ) -> Tensor:
        if not _is_scipy_sparse_obj(x):
            raise ValueError("{x} is not a SciPy sparse object.")
        return Tensor(x, fill_value=fill_value, copy=copy)

    @classmethod
    def _from_scipy_sparse(
        cls,
        x,
        *,
        fill_value: np.number | None = None,
        copy: bool | None = None,
    ) -> JuliaObj:
        if copy is False and not (
            x.format in ("coo", "csr", "csc") and x.has_canonical_format
        ):
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        if x.format not in ("coo", "csr", "csc"):
            x = x.asformat("coo")
        if copy:
            x = x.copy()
        if not x.has_canonical_format:
            x.sum_duplicates()
            assert x.has_canonical_format

        if x.format == "coo":
            return cls.construct_coo_jl_object(
                coords=(x.col, x.row),
                data=x.data,
                shape=x.shape[::-1],
                order=Tensor.row_major,
                fill_value=fill_value,
            )
        if x.format == "csc":
            return cls.construct_csc_jl_object(
                arg=(x.data, x.indices, x.indptr),
                shape=x.shape,
                fill_value=fill_value,
            )
        if x.format == "csr":
            return cls.construct_csr_jl_object(
                arg=(x.data, x.indices, x.indptr),
                shape=x.shape,
                fill_value=fill_value,
            )
        raise ValueError(f"Unsupported SciPy format: {type(x)}")

    @classmethod
    def construct_coo_jl_object(
        cls, coords, data, shape, order, fill_value=0.0
    ) -> JuliaObj:
        assert len(coords) == 2
        ndim = len(shape)
        order = cls.preprocess_order(order, ndim)

        lvl = jl.Element(data.dtype.type(fill_value), data)
        ptr = jl.Vector[jl.Int]([1, len(data) + 1])
        tbl = tuple(jl.PlusOneVector(arr) for arr in coords)

        return jl.swizzle(jl.Tensor(jl.SparseCOO[ndim](lvl, shape, ptr, tbl)), *order)

    @classmethod
    def construct_coo(
        cls, coords, data, shape, order=row_major, fill_value=0.0
    ) -> Tensor:
        return Tensor(
            cls.construct_coo_jl_object(coords, data, shape, order, fill_value)
        )

    @staticmethod
    def _construct_compressed2d_jl_object(
        arg: TupleOf3Arrays,
        shape: tuple[int, ...],
        order: tuple[int, ...],
        fill_value: np.number = 0.0,
    ) -> JuliaObj:
        assert isinstance(arg, tuple) and len(arg) == 3
        assert len(shape) == 2

        data, indices, indptr = arg
        dtype = data.dtype.type
        indices = jl.PlusOneVector(indices)
        indptr = jl.PlusOneVector(indptr)

        lvl = jl.Element(dtype(fill_value), data)
        return jl.swizzle(
            jl.Tensor(
                jl.Dense(jl.SparseList(lvl, shape[0], indptr, indices), shape[1])
            ),
            *order,
        )

    @classmethod
    def construct_csc_jl_object(
        cls, arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> JuliaObj:
        return cls._construct_compressed2d_jl_object(
            arg=arg, shape=shape, order=(1, 2), fill_value=fill_value
        )

    @classmethod
    def construct_csc(
        cls, arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> Tensor:
        return Tensor(cls.construct_csc_jl_object(arg, shape, fill_value))

    @classmethod
    def construct_csr_jl_object(
        cls, arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> JuliaObj:
        return cls._construct_compressed2d_jl_object(
            arg=arg, shape=shape[::-1], order=(2, 1), fill_value=fill_value
        )

    @classmethod
    def construct_csr(
        cls, arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> Tensor:
        return Tensor(cls.construct_csr_jl_object(arg, shape, fill_value))

    @staticmethod
    def construct_csf_jl_object(
        arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> JuliaObj:
        assert isinstance(arg, tuple) and len(arg) == 3

        data, indices_list, indptr_list = arg
        dtype = data.dtype.type

        assert len(indices_list) == len(shape) - 1
        assert len(indptr_list) == len(shape) - 1

        indices_list = [jl.PlusOneVector(i) for i in indices_list]
        indptr_list = [jl.PlusOneVector(i) for i in indptr_list]

        lvl = jl.Element(dtype(fill_value), data)
        for size, indices, indptr in zip(
            shape[:-1], indices_list, indptr_list, strict=False
        ):
            lvl = jl.SparseList(lvl, size, indptr, indices)

        return jl.swizzle(
            jl.Tensor(jl.Dense(lvl, shape[-1])), *range(1, len(shape) + 1)
        )

    @classmethod
    def construct_csf(
        cls, arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> Tensor:
        return Tensor(cls.construct_csf_jl_object(arg, shape, fill_value))

    def to_scipy_sparse(self, accept_fv=None):
        import scipy.sparse as sp

        if accept_fv is None:
            accept_fv = [0]
        elif not isinstance(accept_fv, Iterable):
            accept_fv = [accept_fv]

        if self.ndim != 2:
            raise ValueError(
                "Can only convert a 2-dimensional array to a Scipy sparse matrix."
            )
        if not builtins.any(_eq_scalars(self.fill_value, fv) for fv in accept_fv):
            raise ValueError(
                f"Can only convert arrays with {accept_fv} fill-values "
                "to a Scipy sparse matrix."
            )
        order = self.get_order()
        body = self._obj.body

        if str(jl.typeof(body.lvl).name.name) == "SparseCOOLevel":
            data = np.asarray(body.lvl.lvl.val)
            coords = body.lvl.tbl
            row, col = coords[::-1] if order == (1, 0) else coords
            row, col = np.asarray(row) - 1, np.asarray(col) - 1
            return sp.coo_matrix((data, (row, col)), shape=self.shape)

        if (
            str(jl.typeof(body.lvl).name.name) == "DenseLevel"
            and str(jl.typeof(body.lvl.lvl).name.name) == "SparseListLevel"
        ):
            data = np.asarray(body.lvl.lvl.lvl.val)
            indices = np.asarray(body.lvl.lvl.idx) - 1
            indptr = np.asarray(body.lvl.lvl.ptr) - 1
            sp_class = sp.csr_matrix if order == (1, 0) else sp.csc_matrix
            return sp_class((data, indices, indptr), shape=self.shape)
        if (
            jl.typeof(body.lvl).name.name in sparse_formats_names
            or jl.typeof(body.lvl.lvl).name.name in sparse_formats_names
        ):
            storage = Storage(SparseCOO(self.ndim, Element(self.fill_value)), order)
            return self.to_storage(storage).to_scipy_sparse()
        raise ValueError("Tensor can't be converted to scipy.sparse object.")

    @staticmethod
    def _raise_julia_copy_not_supported() -> None:
        raise ValueError("copy=True isn't supported for Julia object inputs")

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2021.12", "2022.12", "2023.12", "2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')
        import finch

        return finch


def random(shape, density=0.01, random_state=None):
    args = [*shape, density]
    if random_state is not None:
        if isinstance(random_state, np.random.Generator):
            seed = random_state.integers(np.iinfo(np.int32).max)
        else:
            seed = random_state
        rng = jl.Random.default_rng()
        jl.Random.seed_b(rng, seed)
        args = [rng] + args
    return Tensor(jl.fsprand(*args))


def asarray(
    obj,
    /,
    *,
    dtype: DType | None = None,
    format: str | None = None,
    fill_value: np.number | None = None,
    device: Device | None = None,
    copy: bool | None = None,
) -> Tensor:
    if format not in {"coo", "csr", "csc", "csf", "dense", None}:
        raise ValueError(f"{format} format not supported.")
    _validate_device(device)
    tensor = (
        obj
        if isinstance(obj, Tensor)
        else Tensor(obj, fill_value=fill_value, copy=copy)
    )
    if format is not None:
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        order = tensor.get_order()
        if format == "coo":
            storage = Storage(SparseCOO(tensor.ndim, Element(tensor.fill_value)), order)
        elif format == "csr":
            storage = Storage(Dense(SparseList(Element(tensor.fill_value))), (2, 1))
        elif format == "csc":
            storage = Storage(Dense(SparseList(Element(tensor.fill_value))), (1, 2))
        elif format == "csf":
            storage = Element(tensor.fill_value)
            for _ in range(tensor.ndim - 1):
                storage = SparseList(storage)
            storage = Storage(Dense(storage), order)
        elif format == "dense":
            storage = DenseStorage(tensor.ndim, tensor.dtype, order)
        tensor = tensor.to_storage(storage)

    if dtype is not None:
        return astype(tensor, dtype, copy=copy)
    return tensor


def reshape(
    x: Tensor, /, shape: tuple[int, ...], *, copy: bool | None = None
) -> Tensor:
    if copy is False:
        raise ValueError("Unable to avoid copy during reshape.")
    # TODO: https://github.com/finch-tensor/Finch.jl/issues/743
    #       Revert to `jl.reshape` implementation once aforementioned
    #       issue is solved.
    warnings.warn(
        "`reshape` densified the input tensor.", PerformanceWarning, stacklevel=2
    )
    arr = x.todense()
    arr = arr.reshape(shape)
    return Tensor(arr)


def _validate_device(device: Device) -> None:
    if device not in {"cpu", None}:
        raise ValueError(
            f'Device not understood. Only "cpu" is allowed, but received: {device}'
        )
