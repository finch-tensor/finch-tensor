from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Generic, TypeVar

from finch.algebra import AbstractFill, FinchOperator

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
    def fill_value(self) -> AbstractFill: ...

    @fill_value.setter
    @abstractmethod
    def fill_value(self, value: AbstractFill) -> None: ...


TS = TypeVar("TS", bound=TensorStats)


class StatsFactory(ABC, Generic[TS]):
    @abstractmethod
    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> TS: ...

    @abstractmethod
    def copy(self, stat: TS) -> TS: ...

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
