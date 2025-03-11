from .julia import jl
from .typing import OrderType, DType, JuliaObj


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


sparse_formats_names = (
    "SparseList",
    "Sparse",
    "SparseHash",
    "SparseCOO",
    "SparseRLE",
    "SparseVBL",
    "SparseBand",
    "SparsePoint",
    "SparseInterval",
)


# STORAGE


class Storage:
    def __init__(self, levels_descr: AbstractLevel, order: OrderType = None):
        self.levels_descr = levels_descr
        self.order = order if order is not None else "C"

    def __str__(self) -> str:
        return f"Storage(lvl={str(self.levels_descr)}, order={self.order})"


class DenseStorage(Storage):
    def __init__(self, ndim: int, dtype: DType, order: OrderType = None):
        lvl = Element(dtype(0))
        for _ in range(ndim):
            lvl = Dense(lvl)

        super().__init__(levels_descr=lvl, order=order)
