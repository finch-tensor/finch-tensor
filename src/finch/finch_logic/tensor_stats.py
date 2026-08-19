from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, MutableMapping, TypeVar

from finch.algebra import FinchOperator

from .nodes import Field

if TYPE_CHECKING:
    from finch.autoschedule.tensor_stats.bound_stats import DC


class TensorStats(ABC):
    dcs: Iterable[DC]

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
    def fill_value(self) -> Any: ...

    @abstractmethod
    def get_dim_size(self, field: Field) -> float: ...


T = TypeVar("T", bound=TensorStats)


class StatsFactory(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> T: ...

    @abstractmethod
    def copy(self, stat: T) -> T: ...

    @abstractmethod
    def mapjoin(self, op: FinchOperator, *args: T) -> T: ...

    @abstractmethod
    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: T,
    ) -> T: ...

    @abstractmethod
    def relabel(self, stats: T, relabel_indices: tuple[Field, ...]) -> T: ...

    @abstractmethod
    def reorder(self, stats: T, reorder_indices: tuple[Field, ...]) -> T: ...
