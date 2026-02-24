from abc import abstractmethod
from typing import Any

import numpy as np

from finchlite import Tensor, TensorFType

from .julia import jl
from .typing import JuliaObj, number


# Abstract FTypes
class LevelFType(TensorFType):
    def from_numpy(self, _) -> Tensor:
        raise NotImplementedError

    def shape_type(self) -> tuple[type, ...]:
        return tuple(self.element_type for _ in range(self.ndim))

    def __call__(self, _) -> Tensor:
        raise Exception("Cannot create an object of this type!")


class NestedLevelFType(LevelFType):
    def __init__(self, lvl: LevelFType):
        self.lvl = lvl

    @property
    def ndim(self) -> np.intp:
        return self.lvl.ndim + np.intp(1)

    def fill_value(self) -> Any:
        return self.lvl.fill_value

    def element_type(self) -> Any:
        return self.lvl.element_type

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

    @property
    def ndim(self) -> np.intp:
        return np.intp(0)

    def fill_value(self) -> Any:
        return self._fill_value

    def element_type(self) -> Any:
        return type(self._fill_value)

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


# Helper Methods
def construct_levels(obj: JuliaObj, fill_value: number) -> LevelFType:
    if jl.isa(obj.lvl, jl.ElementLevel):
        return Element(fill_value)
    if jl.isa(obj.lvl, jl.DenseLevel):
        return Dense(construct_levels(obj.lvl, fill_value))
    raise Exception("Unhandled exception!")
