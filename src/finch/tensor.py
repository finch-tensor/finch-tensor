from typing import Any

import numpy as np

from finchlite import EagerTensor, Tensor, TensorFType

from .julia import jc, jl
from .levels import (
    LevelFormat,
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
    def shape_type(self) -> tuple[type, ...]:
        return self._lvl.shape_type

    def __call__(self, shape: tuple | None = None) -> Tensor:
        if shape is None:
            raise ValueError("shape argument cannot be None for non scalar tensors.")
        return FinchJLTensor(jl.Finch.Tensor(self._lvl.create_jl_obj(), shape))

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
            self._obj = obj
        else:
            raise ValueError(f"Raw julia object expected. Found: {type(obj)}")

    @property
    def ftype(self) -> TensorFType:
        """Returns the ftype of the buffer"""
        return FinchJLTensorFType(jlobj_to_format(self._obj, jl.fill_value(self._obj)))

    @property
    def shape(self) -> tuple:
        """Shape of the tensor."""
        return jl.size(self._obj)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # standard indexing mode
        key = expand_ellipsis(key, self.shape)
        key = add_missing_dims(key, self.shape)
        key = add_plus_one(key, self.shape)

        result = self._obj[key]
        if jl.isa(result, jl.Finch.Tensor):
            return FinchJLTensor(result)
        return np.array(result)

    def _is_dense(self) -> bool:
        if self._is_scalar():
            return False

        lvl = self._obj.lvl
        for _ in self.shape:
            if not jl.isa(lvl, jl.Finch.Dense):
                return False
            lvl = lvl.lvl
        return True

    def todense(self) -> np.ndarray:
        if self._is_scalar():
            return np.asarray(self._obj.val)

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

        return np.asarray(jl.reshape(dense_tensor.val, shape))

    def __eq__(self, other):
        return isinstance(other, FinchJLTensor) and self._obj == other._obj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2021.12", "2022.12", "2023.12", "2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')
        import finch

        return finch


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
    if isinstance(obj, FinchJLTensor):
        if copy:
            return obj.copy()
        return obj
    if isinstance(obj, np.ndarray):
        if copy:
            obj = obj.copy() if np.isfortran(obj) else np.asfortranarray(obj)
        else:
            if not np.isfortran(obj):
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )

        lvl = jl.ElementLevel(fill_value, obj.reshape(-1))
        for i in obj.shape:
            lvl = jl.DenseLevel(lvl, i)
        return FinchJLTensor(lvl)
    if hasattr(obj, "__module__") and obj.__module__.startswith("scipy.sparse"):
        if obj.format == "coo":
            obj = obj.T
        if copy:
            if obj.format in ("coo", "csc"):
                obj = obj.copy() if obj.has_sorted_indices else obj.sorted_indices()
                if not obj.has_canonical_format:
                    obj.sum_duplicates()
            else:
                obj = obj.asformat("csc")
        if (
            copy is False
            and obj.format not in ("coo", "csc")
            and not obj.has_canonical_format
        ):
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        m, n = obj.shape
        if obj.format == "coo":
            return Tensor(
                jl.SparseCOOLevel(
                    (m, n),
                    jl.ElementLevel(dtype, fill_value, obj.data),
                    2,
                    idxs=(
                        jl.Finch.PlusOneVector(obj.cols),
                        jl.Finch.PlusOneVector(obj.rows),
                    ),
                )
            )
        if obj.format == "csc":
            return Tensor(
                jl.DenseLevel(
                    jl.SparseListLevel(
                        jl.ElementLevel(dtype, fill_value, obj.data),
                        n,
                        jl.Finch.PlusOneVector(obj.indptr),
                        jl.Finch.PlusOneVector(obj.indices),
                    ),
                    m,
                )
            )
        raise ValueError(f"Unsupported SciPy format: {type(obj)}")
    raise ValueError(
        f"Either numpy array or a Finch tensor should be provided. Found: {type(obj)}"
    )
