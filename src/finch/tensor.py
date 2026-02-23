from typing import Any

import numpy as np

from finchlite import EagerTensor, Tensor, TensorFType

from .julia import jl
from .levels import Dense, Element, NestedLevelFType
from .typing import JuliaObj
from .utils import add_missing_dims, add_plus_one, expand_ellipsis


# Tensor Class and associated ftype
class FinchJLTensorFType(TensorFType):
    def __init__(self, lvl):
        self._lvl: NestedLevelFType = lvl

    def ndim(self) -> np.intp:
        return self._lvl.ndim

    def fill_value(self) -> Any:
        return self._lvl.fill_value

    def element_type(self) -> Any:
        return self._lvl.element_type

    def shape_type(self) -> tuple[type, ...]:
        return self._lvl.shape_type

    def __call__(self, shape: tuple) -> Tensor:
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

    def ftype(self):
        """Returns the ftype of the buffer"""
        # TODO: figure out a way to walk through the levels and construct the ftype
        return FinchJLTensorFType(Dense(Dense(Element(0))))

    def shape(self) -> tuple:
        """Shape of the tensor."""
        return self.obj.shape

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
        return result

    def __eq__(self, other):
        return isinstance(other, FinchJLTensor) and self._obj == other._obj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)
