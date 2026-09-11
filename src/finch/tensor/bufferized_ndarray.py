from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    AbstractFill,
    DynamicFill,
    FType,
    ImmutableStructFType,
    StaticFill,
    Tensor,
    TupleFType,
    as_fill,
    ffuncs,
    ftype,
    is_dynamic,
    normalize_device,
)
from finch.codegen import NumpyBuffer, NumpyBufferFType
from finch.codegen.numba_codegen import to_numpy_type
from finch.compile import looplets as lplt
from finch.compile.lower import AssemblyContext, FinchTensorFType

from .fiber_tensor import FiberTensorFType
from .level import Level, LevelFType
from .override_tensor import OverrideTensor
from .scalar import Scalar
from .traits import Dense, FormatProperty


def _get_default_strides(size: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(np.cumprod((1,) + size[::-1]).astype(int))[-2::-1]


class BufferizedNDArray(OverrideTensor):
    def __init__(
        self,
        val: NumpyBuffer,
        shape: tuple[np.integer, ...],
        strides: tuple[np.integer, ...],
        fill_value: Any = 0,
        device=None,
    ):
        self.val = val
        self._shape = shape
        pos_t = val.ftype.length_type
        self.strides = tuple(pos_t(stride) for stride in strides)
        elem_t = val.ftype.element_type
        self._fill_value = (
            fill_value
            if isinstance(fill_value, AbstractFill)
            else as_fill(elem_t(fill_value))
        )

        self._device = normalize_device(device)

    def to_numpy(self):
        """
        Convert the bufferized NDArray to a NumPy array.
        This is used to get the underlying NumPy array from the bufferized NDArray.
        """
        return self.val.arr.reshape(self._shape, copy=False)

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")

    @classmethod
    def from_numpy(
        cls, arr: np.ndarray, fill_value: Any = 0, device=None
    ) -> "BufferizedNDArray":
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        itemsize = arr.dtype.itemsize
        strides = tuple(np.intp(stride // itemsize) for stride in arr.strides)
        shape = tuple(np.intp(s) for s in arr.shape)
        val = NumpyBuffer(arr.reshape(-1, copy=False))
        fill = as_fill(fill_value)
        coerced = np.asarray(fill.value, dtype=arr.dtype).flat[0]
        return BufferizedNDArray(
            val,
            shape,
            strides,
            DynamicFill(coerced) if is_dynamic(fill) else StaticFill(coerced),
            device=device,
        )

    def __array__(self, dtype=None, copy=None):
        out = self.to_numpy()
        if dtype is not None and out.dtype != dtype:
            if copy is not None and not copy:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            out = out.astype(dtype)
        return out

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a BufferizedNDArrayFType.
        """
        return BufferizedNDArrayFType(
            buffer_type=ftype(self.val),
            ndim=self.ndim,
            dimension_type=ftype(self._shape),
            fill_value=self._fill_value,
            device=self._device,
        )

    @property
    def shape(self):
        return tuple(int(s) for s in self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        return self._fill_value.value

    @property
    def fill(self) -> Any:
        """The fill value as a struct field, for marshaling to kernels
        compiled against a dynamic fill."""
        return self._fill_value.value

    @property
    def device(self):
        return self._device

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError(f"stream argument is not supported; got {stream!r}")
        device = normalize_device(device)
        if device == self.device:
            return self
        return BufferizedNDArray(
            self.val,
            self._shape,
            self.strides,
            self._fill_value,
            device=device,
        )

    @property
    def element_type(self) -> FType:
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    def declare(self, init, op, shape):
        """
        Declare a bufferized NDArray with the given initialization value,
        operation, and shape.
        """
        if is_dynamic(init):
            # The value arrives at bind time: use this instance's fill.
            init = self._fill_value.value
        for i in range(self.val.length()):
            self.val.store(i, init)
        return self

    def freeze(self, op):
        return self

    def thaw(self, op):
        return self

    @property
    def lvl(self):
        return BufferizedNDArrayLevel(self)

    def access(self, indices, op):
        return BufferizedNDArrayAccessor(self).access(indices, op)

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        val = self.val.load(0)
        return val.item() if hasattr(val, "item") else val

    def __getitem__(self, index):
        """
        Get an item from the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, slice | np.ndarray) or (
            isinstance(index, tuple) and any(isinstance(i, slice) for i in index)
        ):
            result = self.to_numpy()[index]
            if isinstance(result, np.ndarray):
                return BufferizedNDArray.from_numpy(
                    result, fill_value=self.fill_value, device=self.device
                )
            return Scalar(result, fill_value=self.fill_value, device=self.device)
        if not isinstance(index, tuple) and self.ndim != 1:
            result = self.to_numpy()[index]
            if isinstance(result, np.ndarray):
                return BufferizedNDArray.from_numpy(
                    result, fill_value=self.fill_value, device=self.device
                )
            return Scalar(result, fill_value=self.fill_value, device=self.device)
        if isinstance(index, tuple):
            index = tuple(i for i in index if i is not Ellipsis)
            if index == () and self.ndim == 0:
                return Scalar(
                    self.val.load(0), fill_value=self.fill_value, device=self.device
                )
            if len(index) < self.ndim:
                return BufferizedNDArray.from_numpy(
                    self.to_numpy()[index],
                    fill_value=self.fill_value,
                    device=self.device,
                )
            index = 0 if index == () else np.dot(index, self.strides)
        return Scalar(
            self.val.load(index), fill_value=self.fill_value, device=self.device
        )

    def __setitem__(self, index, value):
        """
        Set an item in the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = tuple(i % s for i, s in zip(index, self._shape, strict=False))
            index = np.ravel_multi_index(index, self._shape)
        self.val.store(index, value)

    def reshape(self, shape, /, *, copy=None):
        return BufferizedNDArray.from_numpy(
            self.to_numpy().reshape(shape),
            fill_value=self.fill_value,
            device=self.device,
        )

    def __str__(self):
        return f"{self.ftype}(shape={self.shape})"

    def __repr__(self):
        return f"{self.ftype}(shape={self.shape})"


class BufferizedNDArrayFType(FinchTensorFType, ImmutableStructFType):
    """
    A ftype for bufferized NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    @property
    def struct_name(self):
        return "BufferizedNDArray"

    @property
    def struct_fields(self):
        fields = [
            ("val", self.buf_t),
            ("shape", self.shape_t),
            ("strides", self.strides_t),
        ]
        if is_dynamic(self._fill_value):
            # The fill value is bound at call time through a struct field.
            fields.append(("fill", self.element_type))
        return fields

    @property
    def lvl_t(self):
        return BufferizedNDArrayLevelFType(self)

    def get_child_type(self, attr):
        if attr != "lvl":
            raise TypeError(f"{self} does not support child {attr!r}")
        return self.lvl_t

    def get_child(self, obj, attr):
        self.get_child_type(attr)
        return obj

    def from_fields(self, buf, shape, strides, fill=None):
        return BufferizedNDArray(
            buf,
            shape,
            strides,
            self.fill_value if fill is None else fill,
            device=self.device,
        )

    def from_numpy(self, arr, fill_value=None):
        val = NumpyBuffer(arr.reshape(-1, copy=False))
        strides = _get_default_strides(arr.shape)
        return BufferizedNDArray(
            val=val,
            shape=tuple(
                t(s)
                for s, t in zip(arr.shape, self.shape_t.struct_fieldtypes, strict=True)
            ),
            strides=tuple(
                t(s)
                for (s, t) in zip(
                    strides, self.strides_t.struct_fieldtypes, strict=True
                )
            ),
            fill_value=self.fill_value if fill_value is None else fill_value,
            device=self.device,
        )

    def __init__(
        self,
        *,
        buffer_type: NumpyBufferFType,
        ndim: int,
        dimension_type: TupleFType | tuple[FType, ...],
        fill_value: Any = 0,
        device=None,
    ):
        if not isinstance(dimension_type, TupleFType):
            dimension_type = TupleFType.from_tuple(dimension_type)
        assert isinstance(dimension_type, TupleFType)
        # Normalize dimension field types to Finch ftypes so generated
        # result_type values are consistent with strict asm type checks.
        dimension_type = TupleFType.from_tuple(
            tuple(ftype(t) for t in dimension_type.struct_fieldtypes)
        )
        self.buf_t = buffer_type
        self._ndim = ndim
        self.shape_t = dimension_type
        self.strides_t = TupleFType.from_tuple((self.buf_t.length_type,) * ndim)
        fill = as_fill(fill_value)
        elem_t = self.buf_t.element_type
        self._fill_value = (
            DynamicFill(elem_t(fill.value), elem_t)
            if is_dynamic(fill)
            else StaticFill(elem_t(fill.value))
        )
        self._device = normalize_device(device)

    def construct(
        self,
        shape: tuple[int, ...],
        fill_value: Any = None,
    ) -> BufferizedNDArray:
        fill = self.fill_value if fill_value is None else as_fill(fill_value)
        arr = np.empty(shape, dtype=to_numpy_type(self.element_type))
        arr[...] = fill.value
        return self.from_numpy(arr, fill_value=fill)

    def __call__(
        self,
        val: Any,
    ) -> BufferizedNDArray:
        """
        Convert a tensor to this bufferized ndarray type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A BufferizedNDArray instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def __eq__(self, other):
        if not isinstance(other, BufferizedNDArrayFType):
            return False
        return (
            self.buf_t == other.buf_t
            and self.ndim == other.ndim
            and bool(np.all(ffuncs.same(self.fill_value, other.fill_value)))
            and self.device == other.device
        )

    def fisinstance(self, other):
        other_t = ftype(other)
        if is_dynamic(self._fill_value) and isinstance(other_t, BufferizedNDArrayFType):
            # A dynamic-fill ftype accepts any fill value of matching dtype.
            other_t = BufferizedNDArrayFType(
                buffer_type=other_t.buf_t,
                ndim=other_t.ndim,
                dimension_type=other_t.shape_t,
                fill_value=self._fill_value,
                device=other_t.device,
            )
        return other_t == self

    def __hash__(self):
        return hash(
            (self.buf_t, self.ndim, ffuncs.samehash(self.fill_value), self.device)
        )

    def __str__(self):
        return str(self.struct_name)

    def __repr__(self):
        return (
            f"BufferizedNDArrayFType(buffer_type={repr(self.buf_t)},"
            f" ndim = {self.ndim}, dimension_type ={repr(self.shape_t)},"
            f" fill_value={self.fill_value!r})"
        )

    @property
    def ndim(self) -> int:
        return int(self._ndim)

    @ndim.setter
    def ndim(self, val):
        self._ndim = val

    @property
    def fill_value(self) -> AbstractFill:
        return self._fill_value

    @property
    def device(self):
        return self._device

    @property
    def element_type(self):
        return self.buf_t.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(self.shape_t.struct_fieldtypes)

    @property
    def level_format_properties(self) -> list[FormatProperty]:
        return [Dense(tuple(range(n + 1))) for n in range(self.ndim)]

    def lower_dim(self, ctx, obj, r):
        return self.lvl_t.level_lower_dim(ctx, ctx(obj.lvl), r)

    def lower_declare(self, ctx, tns: ntn.Fiber, init, op, shape):
        i_var = asm.Variable("i", self.buf_t.length_type)
        buf = asm.GetAttr(ctx(tns.lvl.root), asm.Literal("val"))
        init_e: asm.AssemblyExpression = (
            # The init value arrives at bind time through the fill field.
            asm.GetAttr(ctx(tns.lvl.root), asm.Literal("fill"))
            if is_dynamic(getattr(init, "val", None))
            else asm.Literal(init.val)
        )
        body = asm.Store(
            buf,
            i_var,
            init_e,
        )
        ctx.exec(asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(buf), body))
        if isinstance(tns.lvl.root, asm.Slot):
            ctx.slots[tns.lvl.root.name] = replace(tns, dirty=True)
        return

    def lower_freeze(self, ctx, tns, op):
        return tns

    def lower_thaw(self, ctx, tns, op):
        return tns

    def unfurl(self, ctx, tns, ext, mode, proto):
        tns = ctx.resolve(tns)
        return self.lvl_t.level_unfurl(ctx, tns, ext, mode, proto, tns.pos)

    def reshape(self, arr, new_shape: tuple):
        new_shape = tuple(np.intp(s) for s in new_shape)
        old_size = int(np.prod(arr.shape, dtype=np.intp)) if arr.shape else 1
        new_size = int(np.prod(new_shape, dtype=np.intp)) if new_shape else 1
        if old_size != new_size:
            raise ValueError(
                f"Cannot reshape array of size {old_size} into shape {new_shape}"
            )
        new_strides = tuple(np.intp(s) for s in _get_default_strides(new_shape))
        default_strides = tuple(np.intp(s) for s in _get_default_strides(arr.shape))
        if arr.strides == default_strides:
            return BufferizedNDArray(
                arr.val,
                new_shape,
                new_strides,
                fill_value=arr.fill_value,
                device=arr.device,
            )
        return BufferizedNDArray.from_numpy(
            arr.to_numpy().reshape(new_shape),
            fill_value=arr.fill_value,
            device=arr.device,
        )

    def lower_unwrap(self, ctx, obj):
        return self.lvl_t.level_lower_unwrap(ctx, obj, obj.pos)

    def lower_increment(self, ctx, obj, op, val):
        return self.lvl_t.level_lower_increment(ctx, obj, op, val, obj.pos)


class BufferizedNDArrayAccessor(Tensor):
    """
    A class representing a tensor view that is bufferized.
    This is used to create a view of a tensor with a specific extent.
    """

    def __init__(self, tns: BufferizedNDArray, nind=None, pos=None, op=None):
        self.tns = tns
        if pos is None:
            tns_ftype = cast(BufferizedNDArrayFType, ftype(self.tns))
            pos = tns_ftype.buf_t.length_type(0)
        self.pos = pos
        self.op = op
        if nind is None:
            nind = 0
        self.nind = nind

    @property
    def lvl(self):
        return BufferizedNDArrayLevel(self.tns, self.nind)

    @property
    def ftype(self):
        return FiberTensorFType(self.lvl.ftype, self.tns.device)

    @property
    def dirty_bit(self):
        return False

    @property
    def shape(self):
        return self.tns.shape[self.nind :]

    @property
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        return self.ftype.fill_value.value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    def access(self, indices, op):
        if len(indices) + self.nind > self.tns.ndim:
            raise IndexError(
                f"Too many indices for tensor access: "
                f"got {len(indices)} indices for tensor with "
                f"{self.tns.ndim - self.nind} dimensions."
            )
        for i, idx in enumerate(indices):
            if not (0 <= idx < self.tns.shape[self.nind + i]):
                raise IndexError(
                    f"Index {idx} out of bounds for axis {self.nind + i} "
                    f"with size {self.tns.shape[self.nind + i]}"
                )
        pos = self.pos
        for i, idx in enumerate(indices):
            pos += idx * self.tns.strides[self.nind + i]
        return BufferizedNDArrayAccessor(self.tns, self.nind + len(indices), pos, op)

    def unwrap(self):
        """
        Unwrap the tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        return self.tns.val.load(self.pos)

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self.unwrap()

    def to_numpy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_numpy.")

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")

    def increment(self, val):
        """
        Increment the tensor view with a value.
        This updates the tensor at the specified index with the operation and value.
        """
        if self.op is None:
            raise ValueError("No operation defined for increment.")
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        if isinstance(val, Scalar):
            val = val.item()
        self.tns.val.store(self.pos, self.op(self.tns.val.load(self.pos), val))
        return self


@dataclass(frozen=True)
class BufferizedNDArrayLevel(Level):
    """A level cursor sharing an ndarray's storage and shape metadata."""

    tns: BufferizedNDArray
    nind: int = 0

    @property
    def ftype(self):
        return BufferizedNDArrayLevelFType(self.tns.ftype, self.nind)

    @property
    def shape(self):
        return self.tns.shape[self.nind :]

    @property
    def stride(self):
        return self.tns.strides[self.nind] if self.ndim else self.position_type(1)

    @property
    def val(self):
        return self.tns.val

    @property
    def lvl(self):
        if self.ndim == 0:
            raise TypeError("Scalar ndarray levels have no child level")
        return BufferizedNDArrayLevel(self.tns, self.nind + 1)


@dataclass(frozen=True)
class BufferizedNDArrayLevelFType(LevelFType, ImmutableStructFType):
    tns: BufferizedNDArrayFType
    nind: int = 0

    @property
    def struct_name(self):
        return "BufferizedNDArrayLevel"

    @property
    def struct_fields(self):
        return [("tns", self.tns)]

    def from_fields(self, tns):
        return BufferizedNDArrayLevel(tns, self.nind)

    def with_fill(self, fill_value):
        tns = BufferizedNDArrayFType(
            buffer_type=self.tns.buf_t,
            ndim=self.tns.ndim,
            dimension_type=self.tns.shape_t,
            fill_value=fill_value,
            device=self.tns.device,
        )
        return BufferizedNDArrayLevelFType(tns, self.nind)

    def level_get_child_type(self, attr):
        if attr != "lvl" or self.ndim == 0:
            raise TypeError(f"{self} does not support child {attr!r}")
        return self.lvl_t

    def level_get_child(self, obj, attr):
        self.level_get_child_type(attr)
        return obj

    @property
    def lvl_t(self):
        if self.ndim == 0:
            raise TypeError("Scalar ndarray levels have no child level")
        return BufferizedNDArrayLevelFType(self.tns, self.nind + 1)

    @property
    def ndim(self):
        return self.tns.ndim - self.nind

    @property
    def shape_type(self):
        return self.tns.shape_type[self.nind :]

    @property
    def fill_value(self):
        return self.tns.fill_value

    @property
    def element_type(self):
        return self.tns.element_type

    @property
    def position_type(self):
        return self.tns.buf_t.length_type

    @property
    def buffer_type(self):
        return self.tns.buf_t

    @property
    def buffer_factory(self):
        return type(self.buffer_type)

    def level_format_properties(self, n):
        return [Dense(tuple(range(n + r + 1))) for r in range(self.ndim)]

    def level_cost(self, fields, stats, stats_factory, num_pos, lvl):
        for r in range(self.ndim):
            num_pos *= stats.get_dim_size(fields[lvl + r])
        return num_pos * np.dtype(to_numpy_type(self.element_type)).itemsize

    def construct(self, shape, *, pos):
        raise NotImplementedError("An ndarray level is a view of existing storage")

    def __call__(self, val):
        raise NotImplementedError("An ndarray level is a view of existing storage")

    def from_numpy(self, shape, val):
        raise NotImplementedError("An ndarray level is a view of existing storage")

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        raise NotImplementedError(
            "Declare the owning ndarray before accessing its levels"
        )

    def level_lower_freeze(self, ctx, tns, op, pos):
        return tns

    def level_lower_thaw(self, ctx, tns, op, pos):
        return tns

    def level_lower_dim(self, ctx, obj, r):
        return asm.GetAttr(
            asm.GetAttr(obj, asm.Literal("shape")),
            asm.Literal(f"element_{self.nind + r}"),
        )

    def lower_fill(self, lvl_expr):
        if is_dynamic(self.fill_value):
            return asm.GetAttr(lvl_expr, asm.Literal("fill"))
        return asm.Literal(self.fill_value.value)

    def level_lower_unwrap(self, ctx, tns, pos):
        return asm.Load(
            asm.GetAttr(ctx(tns.lvl), asm.Literal("val")),
            pos,
        )

    def level_lower_increment(
        self,
        ctx: AssemblyContext,
        tns: ntn.Fiber,
        op: ntn.Literal,
        val: ntn.NotationExpression,
        pos: asm.AssemblyExpression,
    ):
        buf = asm.GetAttr(ctx(tns.lvl), asm.Literal("val"))
        op_e, pos_e, val_e = ctx(op), pos, ctx(val)
        increment_call = asm.Call(
            op_e,
            (asm.Load(buf, pos_e), val_e),
        )
        if (
            tns.dirty
            and op.val is ffuncs.overwrite
            # init_write only helps the simplifier elide stores of a Known fill
            and not is_dynamic(self.fill_value)
        ):
            increment_call = asm.Call(
                asm.Literal(ffuncs.init_write(self.fill_value)),
                (asm.Load(buf, pos_e), increment_call),
            )

        ctx.exec(asm.Store(buf, pos_e, increment_call))

    def level_unfurl(self, ctx: AssemblyContext, tns, ext, mode, proto, pos):
        level = tns.lvl
        root = ctx(level)

        def child_accessor(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(
                asm.Assign(
                    pos_2,
                    asm.Call(
                        asm.Literal(ffuncs.add),
                        (
                            pos,
                            asm.Call(
                                asm.Literal(ffuncs.mul),
                                (
                                    asm.GetAttr(
                                        asm.GetAttr(root, asm.Literal("strides")),
                                        asm.Literal(f"element_{self.nind}"),
                                    ),
                                    asm.Variable(idx.name, idx.type_),
                                ),
                            ),
                        ),
                    ),
                )
            )
            return lplt.Run(
                ntn.Fiber(
                    ntn.Child(level),
                    pos_2,
                    (*tns.idxs, idx),
                    tns.dirty,
                )
            )

        return lplt.Lookup(child_accessor)
