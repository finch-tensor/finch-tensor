from typing import Any

import numpy as np

from finchlite import EagerTensor, Tensor, TensorFType

from .julia import jc, jl
from .levels import LevelFType, Scalar, construct_levels
from .typing import JuliaObj
from .utils import add_missing_dims, add_plus_one, expand_ellipsis


# Tensor Class and associated ftype
class FinchJLTensorFType(TensorFType):
    def __init__(self, lvl):
        self._lvl: LevelFType = lvl

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
        if isinstance(self._lvl, Scalar):
            return FinchJLTensor(self._lvl.create_jl_obj())

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
        if self._is_scalar():
            return FinchJLTensorFType(Scalar(self._obj.val))
        return FinchJLTensorFType(construct_levels(self._obj, jl.fill_value(self._obj)))

    @property
    def shape(self) -> tuple:
        """Shape of the tensor."""
        return jl.size(self._obj)

    def __getitem__(self, key):
        if self._is_scalar():
            raise ValueError("Scalars are not subscriptable!")

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

    def _is_scalar(self) -> bool:
        return jl.isa(self._obj, jl.Finch.Scalar)

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
