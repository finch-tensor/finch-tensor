from abc import abstractmethod
from typing import Any

import numpy as np

from finchlite import EagerTensor, Tensor, TensorFType

from .julia import jl
from .typing import JuliaObj, number


# Abstract FTypes
class LevelFType(TensorFType):
    def from_numpy(self, _) -> Tensor:
        raise NotImplementedError

    def shape_type(self) -> tuple[type, ...]:
        return tuple(self.element_type for _ in range(self.ndim))


class NestedLevelFType(LevelFType):
    def __init__(self, lvl: LevelFType):
        self.lvl = lvl

    def ndim(self) -> np.intp:
        return self.lvl.ndim + np.intp(1)

    def fill_value(self) -> Any:
        return self.lvl.fill_value

    def element_type(self) -> Any:
        return self.lvl.element_type

    def __call__(self, shape: tuple) -> Tensor:
        return FinchJLTensor(jl.Finch.Tensor(self.create_jl_obj(), shape))

    def __eq__(self, other):
        return type(other) is type(self) and self.lvl == other.lvl

    def __hash__(self):
        return hash((self.__class__.__name__, self.lvl.__hash__))

    @abstractmethod
    def create_jl_obj(self) -> JuliaObj: ...


# Concrete FTypes
class Element(LevelFType):
    def __init__(self, fill_value: number):
        self._fill_value = fill_value

    def ndim(self) -> np.intp:
        return np.intp(0)

    def fill_value(self) -> Any:
        return self._fill_value

    def element_type(self) -> Any:
        return type(self._fill_value)

    def __call__(self, _) -> Tensor:
        raise Exception("Cannot create an object of element type!")

    def __eq__(self, other):
        return isinstance(other, Element) and self._fill_value == other.fill_value

    def __hash__(self):
        return hash((self.__class__.__name__, self._fill_value))

    def create_jl_obj(self) -> JuliaObj:
        return jl.Element(self._fill_value)


class Dense(NestedLevelFType):
    def create_jl_obj(self) -> JuliaObj:
        return jl.Dense(self.lvl.create_jl_obj())


class SparseList(NestedLevelFType):
    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseList(self.lvl.create_jl_obj())


class SparseByteMap(NestedLevelFType):
    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseByteMap(self.lvl.create_jl_obj())


# Tensor Class
class FinchJLTensor(EagerTensor):
    def __init__(self, obj: JuliaObj):
        if isinstance(obj, JuliaObj):
            self._obj = obj
        else:
            raise ValueError(f"Raw julia object expected. Found: {type(obj)}")

        # TODO: figure out a way to walk through the levels and construct the ftype
        self._ftype = Dense(Dense(Element(0)))

    def ftype(self):
        """Returns the ftype of the buffer"""
        return self._ftype

    def shape(self) -> tuple:
        """Shape of the tensor."""
        return self.obj.shape

    def ndim(self) -> np.intp:
        return self._ftype.ndim

    def fill_value(self) -> Any:
        return self._ftype.fill_value

    def element_type(self) -> Any:
        return self._ftype.element_type

    def __eq__(self, other):
        return isinstance(other, FinchJLTensor) and self._obj == other._obj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)
