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

    @property
    def shape_type(self) -> tuple[type, ...]:
        return tuple(self.element_type for _ in range(self.ndim))

    def __call__(self, _) -> Tensor:
        raise Exception("Cannot create an object of this type!")


class Scalar(LevelFType):
    def __init__(self, val: number):
        self._val = val

    @property
    def ndim(self) -> np.intp:
        return np.intp(0)

    @property
    def fill_value(self) -> Any:
        return self._val

    @property
    def element_type(self) -> Any:
        return type(self._val)

    def __eq__(self, other):
        return isinstance(other, Scalar) and self._val == other._val

    def __hash__(self):
        return hash((self.__class__.__name__, self._val))

    def create_jl_obj(self) -> JuliaObj:
        return jl.Scalar(self._val)


class Element(LevelFType):
    def __init__(self, fill_value: number):
        self._fill_value = fill_value

    @property
    def ndim(self) -> np.intp:
        return np.intp(0)

    @property
    def fill_value(self) -> Any:
        return self._fill_value

    @property
    def element_type(self) -> Any:
        return type(self._fill_value)

    def __eq__(self, other):
        return isinstance(other, Element) and self._fill_value == other.fill_value

    def __hash__(self):
        return hash((self.__class__.__name__, self._fill_value))

    def create_jl_obj(self) -> JuliaObj:
        return jl.Element(self._fill_value)


class NestedLevelFType(LevelFType):
    @property
    def ndim(self) -> np.intp:
        return self.lvl.ndim + np.intp(1)

    @property
    def fill_value(self) -> Any:
        return self.lvl.fill_value

    @property
    def element_type(self) -> Any:
        return self.lvl.element_type

    def __eq__(self, other):
        return type(other) is type(self) and self.lvl == other.lvl

    def __hash__(self):
        return hash((self.__class__.__name__, self.lvl.__hash__))

    @abstractmethod
    def create_jl_obj(self) -> JuliaObj: ...


@dataclass(frozen=True)
class Dense(NestedLevelFType):
    lvl: NestedLevelFType

    def create_jl_obj(self) -> JuliaObj:
        return jl.Dense(self.lvl.create_jl_obj())
    
@dataclass(frozen=True)
class SparseList(NestedLevelFType):
    lvl: NestedLevelFType

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseList(self.lvl.create_jl_obj())

@dataclass(frozen=True)
class SparseCOO(NestedLevelFType):
    lvl: NestedLevelFType
    N: int = 2
    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseCOO(self.lvl.create_jl_obj())

@dataclass(frozen=True)
class SparseByteMap(NestedLevelFType):
    lvl: NestedLevelFType

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseByteMap(self.lvl.create_jl_obj())


class _Display:
    _obj: JuliaObj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)


# LEVEL


class AbstractLevel(_Display):
    pass


# core levels


class Dense(AbstractLevel):
    def __init__(self, lvl, shape=None):
        args = [lvl._obj]
        if shape is not None:
            args.append(shape)
        self._obj = jl.Dense(*args)


class Element(AbstractLevel):
    def __init__(self, fill_value, data=None):
        args = [fill_value]
        if data is not None:
            args.append(data)
        self._obj = jl.Element(*args)


class Pattern(AbstractLevel):
    def __init__(self):
        self._obj = jl.Pattern()


# advanced levels


class SparseList(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.SparseList(lvl._obj)


class SparseByteMap(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.SparseByteMap(lvl._obj)

jl.PlusOneVector(arr)

class RepeatRLE(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.RepeatRLE(lvl._obj)


class SparseVBL(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.SparseVBL(lvl._obj)


class SparseCOO(AbstractLevel):
    def __init__(self, ndim, lvl):
        self._obj = jl.SparseCOO[ndim](lvl._obj)


class SparseHash(AbstractLevel):
    def __init__(self, ndim, lvl):
        self._obj = jl.SparseHash[ndim](lvl._obj)


# Helper Methods
def construct_levels(obj: JuliaObj, fill_value: number) -> LevelFType:
    if jl.isa(obj.lvl, jl.Finch.Element):
        return Element(fill_value)
    if jl.isa(obj.lvl, jl.Finch.Dense):
        return Dense(construct_levels(obj.lvl, fill_value))
    if jl.isa(obj.lvl, jl.Finch.SparseList):
        return SparseList(construct_levels(obj.lvl, fill_value))
    if jl.isa(obj.lvl, jl.Finch.SparseByteMap):
        return SparseByteMap(construct_levels(obj.lvl, fill_value))
    raise Exception("Unhandled exception!")
