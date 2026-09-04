from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    AbstractFill,
    DynamicFill,
    FType,
    ImmutableStructFType,
    StaticFill,
    as_fill,
    ftype,
    is_dynamic,
)
from finch.codegen import NumpyBufferFType
from finch.compile.lower import AssemblyContext

from .level import Level, LevelFType


@dataclass(unsafe_hash=True)
class ElementLevelFType(LevelFType, ImmutableStructFType):
    fill_value: Any = None
    element_type: FType | None = None
    position_type: FType | None = None
    buffer_factory: Any = NumpyBufferFType
    buffer_type: Any = None

    @property
    def struct_name(self):
        return "ElementLevelFType"

    @property
    def struct_fields(self):
        fields = [
            ("val", self.buffer_type),
        ]
        if is_dynamic(self.fill_value):
            # The fill value is bound at call time through a struct field.
            fields.append(("fill", self.element_type))
        return fields

    def __post_init__(self):
        # Ensure element_type is an FType
        if self.element_type is None:
            assert self.fill_value is not None, (
                "Must provide either element_type or fill_value."
            )
            self.element_type = ftype(self.fill_value)
        assert isinstance(self.element_type, FType), (
            "element_type must be an instance of FType"
        )
        if self.buffer_type is None:
            self.buffer_type = self.buffer_factory(self.element_type)
        if self.position_type is None:
            self.position_type = np.intp
        self.position_type = ftype(self.position_type)
        self.element_type = self.buffer_type.element_type
        fill = as_fill(self.fill_value)
        self.fill_value = (
            fill if is_dynamic(fill) else StaticFill(self.element_type(fill.value))
        )

    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "ElementLevel":
        """
        Creates an instance of ElementLevel with the given ftype.

        Args:
            shape: Should be always `()`, used for validation.
        Returns:
            An instance of ElementLevel.
        """
        if len(shape) != 0:
            raise ValueError("ElementLevelFType must be called with an empty shape.")
        val = self.buffer_type(len=pos)
        return ElementLevel(self, val)

    def __call__(self, val: Any) -> "ElementLevel":
        """
        Convert a level to this element level type.

        Args:
            val: A value to convert to this type.
        Returns:
            An ElementLevel instance of this type.
        """
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    def __str__(self):
        match self.fill_value:
            case DynamicFill() as fill:
                return f"ElementLevelFType(fv={fill})"
            case StaticFill() as fill:
                return f"ElementLevelFType(fv={fill.value})"
            case fill:
                return f"ElementLevelFType(fv={fill})"

    @property
    def ndim(self):
        return 0

    @property
    def shape_type(self):
        return ()

    @property
    def lvl_t(self):
        raise Exception("ElementLevelFType is the leaf level.")

    def level_format_properties(self, n):
        return []

    def with_fill(self, fill_value: Any) -> "ElementLevelFType":
        return ElementLevelFType(
            fill_value=fill_value,
            element_type=self.element_type,
            position_type=self.position_type,
            buffer_factory=self.buffer_factory,
            buffer_type=self.buffer_type,
        )

    def lower_fill(self, lvl_expr):
        return asm.GetAttr(lvl_expr, asm.Literal("fill"))

    def from_fields(self, val=None, fill=None) -> "ElementLevel":
        # Wrap numpy arrays in NumpyBuffer and flatten, similar to BufferizedNDArray
        if val is not None and isinstance(val, np.ndarray):
            from finch.codegen import NumpyBuffer

            val = NumpyBuffer(np.asarray(val).reshape(-1, copy=False))
        fmt = self if fill is None else self.with_fill(fill)
        return ElementLevel(_format=fmt, _val=val)

    def level_lower_declare(self, ctx, lvl, init, op, shape, pos):
        buf = asm.GetAttr(lvl, asm.Literal("val"))
        i_var = asm.Variable("i", self.buffer_type.length_type)
        init_e: asm.AssemblyExpression = (
            # The init value arrives at bind time through the fill field.
            asm.GetAttr(lvl, asm.Literal("fill"))
            if is_dynamic(getattr(init, "val", None))
            else asm.Literal(init.val)
        )
        body = asm.Store(buf, i_var, init_e)
        ctx.exec(asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(buf), body))

    def level_lower_unwrap(self, ctx, obj, pos):
        buf = asm.GetAttr(ctx.fiber_level(obj), asm.Literal("val"))
        return asm.Load(buf, pos)

    def level_lower_increment(
        self,
        ctx: AssemblyContext,
        obj,
        op: ntn.Literal,
        val: ntn.NotationExpression,
        pos: ntn.Variable,
    ):
        buf = asm.GetAttr(ctx.fiber_level(obj), asm.Literal("val"))
        pos_e, op_e, val_e = ctx(pos), ctx(op), ctx(val)
        ctx.exec(
            asm.Store(
                buf,
                pos_e,
                asm.Call(op_e, (asm.Load(buf, pos_e), val_e)),
            )
        )

    def level_lower_freeze(self, ctx, lvl, op, pos):
        return asm.GetAttr(lvl, asm.Literal("val"))

    def level_lower_thaw(self, ctx, lvl, op, pos):
        return asm.GetAttr(lvl, asm.Literal("val"))

    def level_lower_dim(self, ctx, obj, r):
        raise NotImplementedError("ElementLevelFType does not support level_lower_dim.")

    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        raise NotImplementedError("ElementLevelFType does not support level_unfurl.")

    def from_numpy(self, shape, val):
        if len(shape) != 0:
            raise ValueError("ElementLevelFType must be called with an empty shape.")
        return self.from_fields(val)


def element(
    fill_value: Any = None,
    element_type: FType | None = None,
    position_type: FType | None = None,
    buffer_factory: Any = NumpyBufferFType,
    buffer_type: Any = None,
) -> ElementLevelFType:
    """
    Creates an ElementLevelFType with the given parameters.

    Args:
        fill_value: The value to be used as the fill value for the level.
        element_type: The type of elements stored in the level.
        position_type: The type of positions within the level.
        buffer_factory: The factory used to create buffers for the level.
        buffer_type: Format of the value stored in the level.

    Returns:
        An instance of ElementLevelFType.
    """
    return ElementLevelFType(
        fill_value=fill_value,
        element_type=element_type,
        position_type=position_type,
        buffer_factory=buffer_factory,
        buffer_type=buffer_type,
    )


@dataclass
class ElementLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    _format: ElementLevelFType = field(repr=False)
    _val: Any | None = None

    def __post_init__(self):
        if self._val is None:
            self._val = self._format.buffer_type(len=0)

    @property
    def shape(self) -> tuple:
        return ()

    @property
    def stride(self) -> np.integer:
        return np.intp(1)  # TODO: add dimension_type to element_level.py

    @property
    def ftype(self) -> ElementLevelFType:
        return self._format

    def with_fill(self, fill_value: AbstractFill) -> "ElementLevel":
        return replace(self, _format=self._format.with_fill(fill_value))

    @property
    def val(self) -> Any:
        return self._val

    @property
    def fill(self) -> Any:
        """The fill value as a struct field, for marshaling to kernels
        compiled against a dynamic fill."""
        return self._format.fill_value.value

    def __str__(self):
        return f"ElementLevel(val={self._val})"
