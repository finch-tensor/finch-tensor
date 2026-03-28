from typing import Any
import operator

import numpy as np

from finchlite import EagerTensor, Tensor, TensorFType

from . import dtypes as jl_dtypes
from .julia import jc, jl
from .levels import (
    ElementFormat,
    LevelFormat,
    SparseCOOFormat,
    jlobj_to_format,
)
from .typing import DType, JuliaObj
from .utils import add_missing_dims, add_plus_one, expand_ellipsis


# Tensor Class and associated ftype
class FinchJLTensorFType(TensorFType):
    def __init__(self, lvl):
        self._lvl: LevelFormat = lvl

    @property
    def ndim(self) -> np.intp:
        return self._lvl.ndim

    @property
    def fill_value(self) -> Any:
        return self._lvl.fill_value

    @property
    def element_type(self) -> Any:
        return self._lvl.element_type

    @property
    def dtype(self) -> Any:
        return self.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(reversed(self._lvl.shape_type))

    def __call__(self, shape: tuple) -> Tensor:
        return FinchJLTensor(
            jl.Finch.Tensor(self._lvl.create_jl_obj(), tuple(reversed(shape)))
        )

    def from_numpy(self, _) -> Tensor:
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, FinchJLTensorFType):
            return False
        return self._lvl == other._lvl

    def __hash__(self):
        return hash(("FinchJLTensorFType", self._lvl))


class FinchJLTensor(EagerTensor):
    def __init__(self, obj: JuliaObj):
        if isinstance(obj, JuliaObj):
            assert jl.isa(obj, jl.Finch.Tensor)
            self._obj = obj
        else:
            raise ValueError(f"Raw julia object expected. Found: {type(obj)}")

    @property
    def ftype(self) -> TensorFType:
        """Returns the ftype of the buffer"""
        return FinchJLTensorFType(jlobj_to_format(self._obj.lvl))

    @property
    def dtype(self) -> Any:
        return self.element_type

    @property
    def shape(self) -> tuple:
        """Shape of the tensor."""
        return jl.size(self._obj)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Array API behavior: indexing a 0-D array with () is a no-op.
        if self.shape == () and key == ():
            return self

        # standard indexing mode
        key = expand_ellipsis(key, self.shape)
        key = add_missing_dims(key, self.shape)
        key = add_plus_one(key, self.shape)

        result = jl.getindex(self._obj, *reversed(key))
        if all(isinstance(k, int) for k in key):
            return result
        assert jl.isa(result, jl.Finch.Tensor)
        return FinchJLTensor(result)

    def _is_dense(self) -> bool:
        lvl = self._obj.lvl
        for _ in self.shape:
            if not jl.isa(lvl, jl.Finch.Dense):
                return False
            lvl = lvl.lvl
        return True

    def todense(self) -> np.ndarray:
        obj = self._obj

        if self._is_dense:
            # don't materialize a dense finch tensor
            shape = jl.size(obj)
            dense_tensor = obj.lvl
        else:
            # create materialized dense array
            shape = jl.size(obj)
            dense_lvls = jl.Element(jc.convert(self.dtype, jl.fill_value(obj)))
            for _ in range(self.ndim):
                dense_lvls = jl.Dense(dense_lvls)
            dense_tensor = jl.Tensor(dense_lvls, obj).lvl  # materialize

        for _ in range(self.ndim):
            dense_tensor = dense_tensor.lvl

        arr = jl.reshape(dense_tensor.val, tuple(shape))
        return np.asarray(arr)

    def __eq__(self, other):
        return isinstance(other, FinchJLTensor) and self._obj == other._obj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        swiz = jl.swizzle(self._obj, tuple(reversed(range(self.ndim, 1, -1))))
        return jl.sprint(jl.show, jl.MIME("text/plain"), swiz)

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2021.12", "2022.12", "2023.12", "2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')
        import finch

        return finch

    def _scalar_value(self):
        if self.shape != ():
            raise TypeError("only 0-dimensional arrays can be converted to Python scalars")
        return jl.getindex(self._obj)

    def __bool__(self) -> bool:
        return bool(self._scalar_value())

    def __int__(self) -> int:
        return int(self._scalar_value())

    def __float__(self) -> float:
        return float(self._scalar_value())

    def __index__(self) -> int:
        try:
            return operator.index(self._scalar_value())
        except TypeError as exc:
            raise TypeError(
                "only integer scalar arrays can be converted to an index"
            ) from exc


def asarray(
    obj,
    /,
    *,
    dtype: DType | None = None,
    fill_value: np.number | None = None,
    copy: bool | None = None,
) -> FinchJLTensor:
    if fill_value is None:
        fill_value = 0.0
    if copy is None:
        copy = True
    if isinstance(obj, FinchJLTensor):
        if copy:
            return obj.copy()
        return obj
    if isinstance(obj, int | float | complex | bool | list):
        if copy is False:
            raise ValueError(
                "copy=False isn't supported for scalar inputs and Python lists"
            )
        obj = np.asarray(obj)
    if isinstance(obj, np.ndarray):
        if dtype is not None:
            obj = np.asarray(obj, dtype=jl_dtypes.jl_to_np_dtype[dtype])

        # np.asfortranarray converts 0-D arrays into shape-(1,) arrays,
        # so keep scalar inputs on the dedicated rank-0 construction path.
        if obj.ndim == 0:
            return full((), obj.item(), dtype=dtype)

        if copy:
            obj = obj.copy() if np.isfortran(obj) else np.asfortranarray(obj)
        else:
            if not np.isfortran(obj):
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
        buf = np.reshape(np.permute_dims(obj, tuple(reversed(range(obj.ndim)))), -1)

        lvl = jl.ElementLevel(np.asarray(fill_value, dtype=obj.dtype).item(), buf)
        for i in obj.shape:
            lvl = jl.DenseLevel(lvl, i)
        return FinchJLTensor(jl.Tensor(lvl))
    if hasattr(obj, "__module__") and obj.__module__.startswith("scipy.sparse"):
        if copy:
            if obj.format in ("coo", "csr"):
                obj = obj.copy() if obj.has_sorted_indices else obj.sorted_indices()
                if not obj.has_canonical_format:
                    obj.sum_duplicates()
            else:
                obj = obj.asformat("csr")
        if (
            copy is False
            and obj.format not in ("coo", "csr")
            and not obj.has_canonical_format
        ):
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        m, n = obj.shape
        if dtype is not None:
            fill_value = np.asarray(fill_value, dtype=jl_dtypes.jl_to_np_dtype[dtype]).item()
        if obj.format == "coo":
            return FinchJLTensor(
                jl.Tensor(
                    jl.SparseCOOLevel(
                        (n, m),
                        jl.ElementLevel(dtype, fill_value, obj.data),
                        2,
                        idxs=(
                            jl.Finch.PlusOneVector(obj.cols),
                            jl.Finch.PlusOneVector(obj.rows),
                        ),
                    )
                )
            )
        if obj.format == "csr":
            return FinchJLTensor(
                jl.Tensor(
                    jl.DenseLevel(
                        jl.SparseListLevel(
                            jl.ElementLevel(dtype, fill_value, obj.data),
                            m,
                            jl.Finch.PlusOneVector(obj.indptr),
                            jl.Finch.PlusOneVector(obj.indices),
                        ),
                        n,
                    )
                )
            )
        raise ValueError(f"Unsupported SciPy format: {type(obj)}")
    raise ValueError(
        f"Either numpy array or a Finch tensor should be provided. Found: {type(obj)}"
    )


def reshape(
    x: FinchJLTensor, /, shape: tuple[int, ...], *, copy: bool | None = None
) -> FinchJLTensor:
    if copy is False:
        raise ValueError("Unable to avoid copy during reshape.")
    if all(i == 1 for i in x.shape):
        return full(shape, x[tuple(i - 1 for i in x.shape)], dtype=x.dtype)
    return FinchJLTensor(jl.reshape(x._obj, tuple(reversed(shape))))


def full(
    shape: int | tuple[int, ...],
    val: jl_dtypes.number,
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    if not np.isscalar(val):
        raise ValueError("`fill_value` must be a scalar")
    if isinstance(shape, int):
        shape = (shape,)
    dtype = (
        np.asarray(val).dtype.type if dtype is None else jl_dtypes.jl_to_np_dtype[dtype]
    )
    if dtype == np.bool_:  # Fails with: Finch currently only supports isbits defaults
        dtype = bool

    # Rank-0 tensors should be represented as a leaf element level.
    # Building them through SparseCOO requires an explicit rank parameter.
    if len(shape) == 0 and format is None:
        return FinchJLTensor(jl.Tensor(ElementFormat(val, dtype).create_jl_obj()))

    if format is None:
        format = SparseCOOFormat(ElementFormat(val, dtype), len(shape))

    if format.fill_value != val:
        return FinchJLTensor(
            jl.Tensor(format.create_jl_obj(), np.full(val, reversed(shape)))
        )
    return FinchJLTensor(jl.Tensor(format.create_jl_obj(), *reversed(shape)))


def full_like(
    x: FinchJLTensor,
    /,
    fill_value: jl_dtypes.number,
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    return full(x.shape, fill_value, dtype=dtype, format=format)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    return full(shape, np.float64(1), dtype=dtype, format=format)


def ones_like(
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    dtype = x.dtype if dtype is None else dtype
    return ones(x.shape, dtype=dtype, format=format)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    return full(shape, np.float64(0), dtype=dtype, format=format)


def zeros_like(
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    dtype = x.dtype if dtype is None else dtype
    return zeros(x.shape, dtype=dtype, format=format)


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    return full(shape, np.float64(0), dtype=dtype, format=format)


def empty_like(
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
    format: str = "coo",
) -> FinchJLTensor:
    dtype = x.dtype if dtype is None else dtype
    return empty(x.shape, dtype=dtype, format=format)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
) -> FinchJLTensor:
    return asarray(np.arange(start, stop, step, jl_dtypes.jl_to_np_dtype[dtype]))


def linspace(
    start: complex,
    stop: complex,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    endpoint: bool = True,
) -> FinchJLTensor:
    return asarray(
        np.linspace(
            start,
            stop,
            num=num,
            dtype=jl_dtypes.jl_to_np_dtype[dtype],
            endpoint=endpoint,
        )
    )
