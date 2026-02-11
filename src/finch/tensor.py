import numpy as np
from typing import Any
from finchlite import EagerTensor, TensorFType, Tensor
from . import dtypes as jl_dtypes

from .typing import OrderType, JuliaObj
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

# Singleton classes for levels types
# finch tensor lite, formatter stage
# level ftype without the need to create tthe object
# https://github.com/finch-tensor/finch-tensor-lite/blob/main/src/finchlite/autoschedule/formatter.py

class FinchJLTensorFType(TensorFType):
    def __init__(self, jltype, shape_type):
        # Julia type associated with the tensor
        self.jltype = jltype
        self._shape_type = shape_type

    def ndims(self) -> np.intp:
        return np.intp(jl.ndims(self.jltype))

    def fill_value(self) -> Any:
        return jl.fill_value(self.jltype)

    def element_type(self) -> Any:
        return jl.eltype(self.jltype)

    def shape_type(self) -> tuple[type, ...]:
        return self._shape_type

    def __call__(self, shape: tuple) -> Tensor:
        return FinchJLTensor(np.ones(shape=shape))

    def from_numpy(self, arr: np.ndarray) -> Tensor:
        return FinchJLTensor(arr)

    def __eq__(self, other):
        if not isinstance(other, FinchJLTensorFType):
            return False
        return self.jltype == other.jltype

    def __hash__(self):
        return hash(self.jltype)


# TODO: Do we need the scipy and raw julia stuff
class FinchJLTensor(_Display, EagerTensor):
    def __init__(
        self,
        obj: np.ndarray,
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

        if isinstance(obj, np.ndarray):  # numpy constructor
            jl_data = self._from_numpy(obj, fill_value=fill_value, copy=copy)
            self._shape = obj.shape
            self._obj = jl_data
        else:
            raise ValueError(
                "Either scalar, numpy, scipy.sparse or a raw julia object should "
                f"be provided. Found: {type(obj)}"
            )

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a BufferizedNDArrayFType.
        """
        shape_type = []
        for idx in self._shape:
            shape_type.append(type(idx))
        return FinchJLTensorFType(jltype=jl.typeof(self._obj), shape_type=shape_type)

    @property
    def shape(self) -> tuple:
        """Shape of the tensor."""
        return self._shape

    # TODO: do we need to have all the order stuff still?
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
    def preprocess_order(cls, order: OrderType, ndim: int) -> tuple[int, ...]:
        if order == "F":
            permutation = tuple(range(1, ndim + 1))
        elif order == "C":
            permutation = tuple(range(1, ndim + 1)[::-1])
        else:
            raise ValueError(f"order must be 'C' or 'F'.")
        return permutation
