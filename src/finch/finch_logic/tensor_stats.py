from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Generic, TypeVar

from finch.algebra import AbstractFillFType, FinchOperator, StaticFillFType

from .nodes import Field


class TensorStats(ABC):
    @property
    @abstractmethod
    def idxs(self) -> tuple[Field, ...]: ...

    @property
    def index_order(self) -> tuple[Field, ...]:
        return self.idxs

    @property
    @abstractmethod
    def dim_sizes(self) -> MutableMapping[Field, float]: ...

    @property
    @abstractmethod
    def fill_type(self) -> AbstractFillFType:
        """The fill of the tensor's format, which a kernel may depend on."""

    @fill_type.setter
    @abstractmethod
    def fill_type(self, value: AbstractFillFType) -> None: ...


TS = TypeVar("TS", bound=TensorStats)


class StatsFactory(ABC, Generic[TS]):
    @abstractmethod
    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> TS: ...

    @abstractmethod
    def copy(self, stat: TS) -> TS: ...

    def support(self, stat: TS) -> TS:
        """Statistics of the boolean indicator of non-fill entries."""
        # Support preserves the non-fill structure, even for dynamic fills.
        result = self.copy(stat)
        result.fill_type = StaticFillFType(False)
        return result

    @abstractmethod
    def mapjoin(self, op: FinchOperator, *args: TS) -> TS: ...

    @abstractmethod
    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TS,
    ) -> TS: ...

    @abstractmethod
    def relabel(self, stats: TS, relabel_indices: tuple[Field, ...]) -> TS: ...

    @abstractmethod
    def reorder(self, stats: TS, reorder_indices: tuple[Field, ...]) -> TS: ...
