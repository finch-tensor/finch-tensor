from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sps

from finch.algebra import (
    ImmutableStructFType,
    TupleFType,
    bool_,
    ftype,
    normalize_device,
)
from finch.codegen import NumpyBuffer
from finch.compile.lower import FinchTensorFType

from .level import (
    DenseLevel,
    ElementLevel,
    Level,
    LevelFType,
    SparseCOOLevel,
    SparseListLevel,
    dense,
    element,
    sparse_list,
)
from .override_tensor import OverrideTensor
from .traits import FormatProperty


@dataclass
class FiberTensor(OverrideTensor):
    """
    A class representing a tensor with fiber structure.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: Level
    pos: np.integer = np.intp(0)
    dirty_bit: bool = False
    _device: Any = None

    def __post_init__(self):
        self._device = normalize_device(self._device)

    def __repr__(self):
        return f"FiberTensor(lvl={self.lvl})"

    @property
    def ftype(self):
        """
        Returns the ftype of the fiber tensor, which is a FiberTensorFType.
        """
        return FiberTensorFType(self.lvl.ftype, self._device)

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def stride(self):
        return self.lvl.stride

    @property
    def val(self):
        return self.lvl.val

    @property
    def ndim(self):
        return self.lvl.ndim

    @property
    def shape_type(self):
        return self.lvl.shape_type

    @property
    def element_type(self):
        return self.lvl.element_type

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def device(self):
        return self._device

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError(f"stream argument is not supported; got {stream!r}")
        device = normalize_device(device)
        if device == self.device:
            return self
        return FiberTensor(self.lvl, self.pos, self.dirty_bit, device)

    @property
    def position_type(self):
        return self.lvl.position_type

    @property
    def buffer_factory(self):
        return self.lvl.buffer_factory

    def to_numpy(self) -> np.ndarray:
        # TODO: temporary for dense only. TBD in sparse_level PR
        return np.reshape(self.lvl.val.arr, self.shape, copy=False)

    def to_scipy(self):
        if self.ndim != 2:
            raise ValueError("SciPy sparse arrays must be two-dimensional.")

        if self.fill_value != 0:
            raise ValueError("SciPy CSR conversion requires a zero fill value.")

        match self.lvl:
            case SparseCOOLevel(
                lvl=ElementLevel() as element,
                tbl=(row, col),
            ):
                return sps.coo_array(
                    (element.val.arr, (row.arr, col.arr)),
                    shape=self.shape,
                    copy=False,
                )

            case DenseLevel(
                lvl=SparseListLevel(
                    lvl=ElementLevel() as element,
                    ptr=ptr,
                    idx=idx,
                )
            ):
                return sps.csr_array(
                    (element.val.arr, idx.arr, ptr.arr),
                    shape=self.shape,
                    copy=False,
                )

            case _:
                raise NotImplementedError(
                    f"Finch format {self.ftype} is not supported by SciPy conversion."
                )

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self.to_numpy().item()

    def transpose(self):
        return TransposedFiberTensor(self)

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError("FiberTensor transpose requires a two-dimensional tensor.")
        return self.transpose()

    @classmethod
    def from_scipy_csr(
        cls,
        obj: Any,
        /,
        *,
        dtype=None,
        device=None,
        copy=None,
    ):
        if dtype is not None:
            if copy is False:
                obj = obj.astype(dtype, copy=False)
            else:
                obj = obj.astype(dtype, copy=True)
        elif copy is True:
            obj = obj.copy()

        if not obj.has_canonical_format:
            if copy is False:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            obj = obj.copy()
            obj.sum_duplicates()

        data = obj.data
        indices = obj.indices
        indptr = obj.indptr

        element_lvl = element(
            fill_value=np.zeros((), dtype=data.dtype)[()],
            element_type=ftype(data.dtype),
            position_type=ftype(indices.dtype),
        )
        sparse_lvl = sparse_list(element_lvl, dimension_type=ftype(indices.dtype))
        dense_lvl = dense(sparse_lvl, dimension_type=ftype(indices.dtype))

        return cls(
            DenseLevel(
                SparseListLevel(
                    ElementLevel(
                        _format=element_lvl,
                        _val=NumpyBuffer(data),
                    ),
                    dimension=sparse_lvl.dimension_type(obj.shape[1]),
                    ptr=NumpyBuffer(indptr),
                    idx=NumpyBuffer(indices),
                ),
                dimension=dense_lvl.dimension_type(obj.shape[0]),
            ),
            _device=device,
        )


@dataclass(unsafe_hash=True)
class FiberTensorFType(FinchTensorFType, ImmutableStructFType):
    """
    An abstract base class representing the ftype of a fiber tensor.

    Attributes:
        lvl_t: The level ftype to be used for the tensor.
    """

    lvl_t: LevelFType
    _device: Any = None

    def __post_init__(self):
        self._device = normalize_device(self._device)

    @property
    def struct_name(self):
        # TODO: include dt = np.dtype(self.buf_t.element_type)
        return "FiberTensorFType"

    @property
    def struct_fields(self):
        return [
            ("lvl", self.lvl_t),
            ("shape", TupleFType.from_tuple(self.shape_type)),
            ("pos", self.position_type),
            ("dirty_bit", bool_),
        ]

    def construct(self, shape: tuple[int, ...]) -> FiberTensor:
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        return FiberTensor(
            self.lvl_t.construct(shape=shape, pos=1),
            self.position_type(0),
            _device=self.device,
        )

    def __call__(self, val: Any) -> FiberTensor:
        """
        Convert a tensor to this fiber tensor type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A FiberTensor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def __str__(self):
        return f"FiberTensorFType({self.lvl_t})"

    @property
    def ndim(self):
        return self.lvl_t.ndim

    @property
    def shape_type(self):
        return self.lvl_t.shape_type

    @property
    def element_type(self):
        return self.lvl_t.element_type

    @property
    def fill_value(self):
        return self.lvl_t.fill_value

    @property
    def device(self):
        return self._device

    @property
    def position_type(self):
        return self.lvl_t.position_type

    @property
    def buffer_factory(self):
        return self.lvl_t.buffer_factory

    @property
    def buffer_type(self):
        return self.lvl_t.buffer_type

    @property
    def level_format_properties(self) -> list[FormatProperty]:
        return self.lvl_t.level_format_properties(0)

    def unfurl(self, ctx, tns, ext, mode, proto):
        tns = ctx.resolve(tns)
        return self.lvl_t.level_unfurl(ctx, tns, ext, mode, proto, tns.pos)

    def lower_freeze(self, ctx, tns, op):
        return self.lvl_t.level_lower_freeze(ctx, ctx.fiber_level(tns), op, tns.pos)

    def lower_thaw(self, ctx, tns, op):
        return self.lvl_t.level_lower_thaw(ctx, ctx.fiber_level(tns), op, tns.pos)

    def lower_unwrap(self, ctx, tns):
        return self.lvl_t.level_lower_unwrap(ctx, tns, tns.pos)

    def lower_increment(self, ctx, tns, op, val):
        return self.lvl_t.level_lower_increment(ctx, tns, op, val, tns.pos)

    def lower_declare(self, ctx, tns, init, op, shape):
        return self.lvl_t.level_lower_declare(
            ctx, ctx.fiber_level(tns), init, op, shape, tns.pos
        )

    def lower_dim(self, ctx, obj, r):
        return self.lvl_t.level_lower_dim(ctx, ctx.fiber_level(obj), r)

    def from_fields(self, *args) -> FiberTensor:
        lvl, shape, pos, dirty_bit = args
        return FiberTensor(lvl, pos, dirty_bit, self.device)

    # TODO: To be removed - use BufferizedNDArray instead.
    def from_numpy(self, arr: np.ndarray) -> FiberTensor:
        return FiberTensor(
            self.lvl_t.from_numpy(arr.shape, arr),
            pos=self.position_type(0),
            dirty_bit=False,
            _device=self.device,
        )


@dataclass
class TransposedFiberTensor(OverrideTensor):
    """A zero-copy used to represent a transposed FiberTensor."""

    tensor: FiberTensor

    @property
    def ftype(self):
        return self.tensor.ftype

    @property
    def shape(self):
        return self.tensor.shape[::-1]

    @property
    def shape_type(self):
        return self.tensor.shape_type[::-1]

    @property
    def T(self):
        return self.transpose()

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self.to_numpy().item()

    def to_numpy(self):
        match self.tensor.lvl:
            case DenseLevel(lvl=DenseLevel(lvl=ElementLevel())):
                return self.tensor.to_numpy().transpose()
            case _:
                raise NotImplementedError(
                    "NumPy conversion is only supported for dense "
                    "transposed FiberTensor."
                )

    def to_scipy(self):
        """
        Return a zero-copy Scipy CSC of the transposed matrix
        """
        return self.tensor.to_scipy().transpose(copy=False)

    def transpose(self):
        return self.tensor


def fiber_tensor(lvl: LevelFType):
    """
    Creates a FiberTensorFType with the given level ftype and position type.

    Args:
        lvl: The level ftype to be used for the tensor.
    Returns:
        An instance of a fiber tensor format.
    """
    # mypy does not understand that dataclasses generate __hash__ and __eq__
    # https://github.com/python/mypy/issues/19799
    return FiberTensorFType(lvl)  # type: ignore[abstract]
