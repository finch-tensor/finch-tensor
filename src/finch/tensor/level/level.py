from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from finch.algebra import (
    FType,
    FTyped,
)


class LevelFType(FType, ABC):
    """
    An abstract base class representing the ftype of levels.
    """

    @property
    @abstractmethod
    def ndim(self):
        """
        Number of dimensions of the fibers in the structure.
        """
        ...

    @property
    @abstractmethod
    def fill_value(self):
        """
        Fill value of the fibers, or `None` if dynamic.
        """
        ...

    @property
    @abstractmethod
    def element_type(self):
        """
        Type of elements stored in the fibers.
        """
        ...

    @property
    @abstractmethod
    def shape_type(self):
        """
        Tuple of types of the dimensions in the shape.
        """
        ...

    @property
    @abstractmethod
    def position_type(self):
        """
        Type of positions within the levels.
        """
        ...

    @property
    @abstractmethod
    def buffer_factory(self):
        """
        Function to create default buffers for the fibers.
        """
        ...

    @property
    @abstractmethod
    def buffer_type(self): ...

    @property
    @abstractmethod
    def lvl_t(self):
        """
        Get the nested level.
        """
        ...

    @abstractmethod
    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        """
        Emit code to unfurl the fiber at position `pos` in the level.
        """
        ...

    @abstractmethod
    def level_lower_freeze(self, ctx, tns, op, pos):
        """
        Emit code to freeze `pos` previously assembled positions in the level.
        """
        ...

    @abstractmethod
    def level_lower_thaw(self, ctx, tns, op, pos):
        """
        Emit code to thaw `pos` previously assembled positions in the level.
        """
        ...

    @abstractmethod
    def level_lower_unwrap(self, ctx, obj, pos):
        """
        Emit code to return the unwrapped scalar at position `pos` in the level.
        """
        ...

    @abstractmethod
    def level_lower_increment(self, ctx, obj, op, val, pos):
        """
        Emit code to increment position `pos` in the level.
        """
        ...

    @abstractmethod
    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        """
        Emit code to lower a declare of `pos` previously assembled positions in
        the level.
        """
        ...

    @abstractmethod
    def level_lower_dim(self, ctx, obj, r):
        """
        Emit code to return the size of dimension `r` of the subtensors in the level.
        """
        ...

    @abstractmethod
    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "Level":
        """
        Construct a level instance with the given shape.
        """
        ...

    @abstractmethod
    def from_numpy(self, shape, val):
        """
        Construct level from numpy array
        (TODO not strictly safe, only works for dense, replace later)
        """
        ...

    @abstractmethod
    def level_format_properties(self, n):
        """
        Return the format properties contributed by this level type and children.

        ``n`` is the outer dimension index represented by this level. Nested
        levels use increasing indices, so the returned properties can describe how
        this dimension constrains dimensions further inside the fiber tree.
        """
        ...


class Level(FTyped, ABC):
    """
    An abstract base class representing a fiber allocator that manages fibers in
    a tensor.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        Shape of the fibers in the structure.
        """
        ...

    @property
    @abstractmethod
    def stride(self) -> np.integer: ...

    @property
    @abstractmethod
    def val(self) -> Any: ...

    @property
    def ndim(self):
        return self.ftype.ndim

    @property
    def fill_value(self):
        return self.ftype.fill_value

    @property
    def element_type(self):
        return self.ftype.element_type

    @property
    def shape_type(self):
        return self.ftype.shape_type

    @property
    def position_type(self):
        return self.ftype.position_type

    @property
    def buffer_factory(self):
        return self.ftype.buffer_factory

    @property
    def buffer_type(self):
        return self.ftype.buffer_type
