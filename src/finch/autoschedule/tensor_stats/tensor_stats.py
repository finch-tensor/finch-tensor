from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping
from typing import Any, Generic, Self, TypeVar

import numpy as np

from finch.algebra import (
    AbstractFill,
    FinchOperator,
    FType,
    NAryFinchOperatorFType,
    apply_fill,
    as_fill,
    ftype,
    is_annihilator,
    is_dynamic,
    is_idempotent,
    is_identity,
    repeat_operator,
)
from finch.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    Table,
    TensorStats,
)
from finch.finch_logic.nodes import merge_dim_type

TS = TypeVar("TS", bound="BaseTensorStats")


class BaseTensorStats(TensorStats):
    """
    Base class for tensor statistics.
    """

    def __init__(
        self,
        index_order: Iterable[Field] | BaseTensorStats,
        dim_sizes: MutableMapping[Field, float] | None = None,
        fill_value: Any = None,
        dim_types: MutableMapping[Field, FType | None] | None = None,
    ):
        if isinstance(index_order, BaseTensorStats):
            if dim_sizes is not None:
                raise TypeError(
                    "dim_sizes must be omitted when constructing from BaseTensorStats"
                )
            self._init_from_fields(
                index_order.index_order,
                index_order.dim_sizes,
                index_order.fill_value,
                index_order.dim_types,
            )
        else:
            if dim_sizes is None:
                raise TypeError("dim_sizes are required when constructing from fields")
            self._init_from_fields(index_order, dim_sizes, fill_value, dim_types)

    def _init_from_fields(
        self,
        index_order: Iterable[Field],
        dim_sizes: MutableMapping[Field, float],
        fill_value: Any,
        dim_types: MutableMapping[Field, FType | None] | None = None,
    ):
        self._index_order = tuple(index_order)
        self._dim_sizes: OrderedDict[Field, float] = OrderedDict(dim_sizes)
        # A dimension type is None when it is unknown, as for a broadcast.
        self._dim_types: OrderedDict[Field, FType | None] = OrderedDict(
            dim_types if dim_types is not None else dict.fromkeys(self._dim_sizes)
        )
        # Stats always carry an AbstractFill: propagation must be able to say
        # "this fill exists but must not be specialized on".
        self._fill_value: AbstractFill = as_fill(fill_value)

    @classmethod
    def from_fields(
        cls,
        index_order: Iterable[Field],
        dim_sizes: MutableMapping[Field, float],
        fill_value: Any,
        dim_types: MutableMapping[Field, FType | None] | None = None,
    ) -> Self:
        return cls(index_order, dim_sizes, fill_value, dim_types=dim_types)

    def copy(self) -> Self:
        new = object.__new__(type(self))
        new.__dict__ = {name: copy.copy(value) for name, value in self.__dict__.items()}
        return new

    @property
    def dim_sizes(self) -> MutableMapping[Field, float]:
        return self._dim_sizes

    @dim_sizes.setter
    def dim_sizes(self, value: MutableMapping[Field, float]):
        self._dim_sizes = OrderedDict(value)

    def get_dim_size(self, idx: Field) -> float:
        return self.dim_sizes[idx]

    @property
    def dim_types(self) -> MutableMapping[Field, FType | None]:
        return self._dim_types

    @property
    def shape_type(self) -> tuple[FType | None, ...]:
        """The type of each dimension in index order, or None when unknown."""
        return tuple(self.dim_types.get(idx) for idx in self.index_order)

    @property
    def element_type(self) -> FType:
        return ftype(self.fill_value)

    @property
    def index_order(self) -> tuple[Field, ...]:
        return self._index_order

    @index_order.setter
    def index_order(self, value: Iterable[Field]):
        self._index_order = tuple(value)

    @property
    def idxs(self) -> tuple[Field, ...]:
        return self.index_order

    @property
    def fill_value(self) -> AbstractFill:
        """The fill, and whether kernels may specialize on it."""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self._fill_value = as_fill(value)

    def get_dim_space_size(self, idx: Iterable[Field]) -> float:
        prod = 1
        for i in idx:
            prod *= int(self.dim_sizes[i])
            if prod == 0 or prod > np.iinfo(np.int64).max:
                return float("inf")
        return float(prod)


class BaseTensorStatsFactory(ABC, Generic[TS]):
    def __init__(self, stats_cls: type[TS]):
        self.stats_cls = stats_cls

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> BaseTensorStats:
        shape = tensor.shape
        dim_sizes = OrderedDict(
            (axis, float(shape[i])) for i, axis in enumerate(fields)
        )
        dim_types = OrderedDict(zip(fields, tensor.ftype.shape_type, strict=True))
        return BaseTensorStats(fields, dim_sizes, tensor.fill_value, dim_types)

    def copy(self, stat: TS) -> TS:
        if not isinstance(stat, self.stats_cls):
            raise TypeError(f"copy expected a {self.stats_cls.__name__} instance")
        return stat.copy()

    def mapjoin(self, op: FinchOperator, *args: TS) -> TS:
        join_args: list[TS] = []
        union_args: list[TS] = []
        for s in args:
            if is_annihilator(op.ftype, s.fill_value):
                join_args.append(s)
            else:
                union_args.append(s)
        if (
            op.ftype.is_associative
            and op.ftype.is_commutative
            and isinstance(op.ftype, NAryFinchOperatorFType)
        ):
            if union_args:
                join_args.append(self._mapjoin_union(op, *union_args))

            return self._mapjoin_join(op, *join_args)
        if len(union_args) > 1:
            return self._mapjoin_union(op, *args)
        return self._mapjoin_join(op, *args)

    def _mapjoin_union(self, op: FinchOperator, *union_args: TS) -> TS:
        raise NotImplementedError

    def _mapjoin_join(self, op: FinchOperator, *join_args: TS) -> TS:
        raise NotImplementedError

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

    @staticmethod
    def _mapjoin_defs(op: FinchOperator, *args: BaseTensorStats) -> BaseTensorStats:
        new_fill_value = apply_fill(op, *(s.fill_value for s in args))
        new_index_order = MapJoin(
            Literal(op),
            tuple(
                Table(Alias(f"_{i}"), tuple(a.index_order)) for i, a in enumerate(args)
            ),
        ).fields()
        new_dim_sizes: dict = {}
        for index in new_index_order:
            for s in args:
                if index in s.index_order:
                    new_dim_sizes[index] = s.dim_sizes[index]
                    break
        assert set(new_dim_sizes.keys()) == set(new_index_order)
        new_dim_types: dict = dict.fromkeys(new_index_order)
        for s in args:
            for index in s.index_order:
                new_dim_types[index] = merge_dim_type(
                    new_dim_types[index], s.dim_types.get(index)
                )
        return BaseTensorStats.from_fields(
            new_index_order, new_dim_sizes, new_fill_value, new_dim_types
        )

    @staticmethod
    def aggregate_def(
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        d: BaseTensorStats,
    ) -> BaseTensorStats:
        red_set = set(reduce_indices) & set(d.index_order)
        n = math.prod(int(d.dim_sizes[x]) for x in red_set)

        old_fill = d.fill_value.value
        new_fill = init
        if init is None:
            if is_dynamic(d.fill_value):
                # A dynamic background must stay dynamic through the reduction.
                # NOTE: the computed d.fill_value.value is not necessarily correct.
                new_fill = apply_fill(op, d.fill_value, d.fill_value)
            elif is_identity(op.ftype, old_fill) or is_idempotent(op.ftype):
                new_fill = as_fill(op(old_fill, old_fill))
            else:
                try:
                    new_fill = as_fill(repeat_operator(op.ftype)(old_fill, n))
                except AttributeError:
                    # This is going to be VERY SLOW. Should raise a warning about
                    #  reductions over non-identity fill values. Depending on the
                    # semantics of reductions, we might be able to do this faster.
                    print(
                        "Warning: A reduction can take place over a tensor whose fill"
                        "value is not the reduction operator's identity. This can"
                        "result in a large slowdown as the new fill is calculated."
                    )
                    acc = old_fill
                    for _ in range(max(n - 1, 0)):
                        acc = op(acc, old_fill)
                    new_fill = as_fill(acc)

        new_dim_sizes = OrderedDict(
            (ax, d.dim_sizes[ax]) for ax in d.dim_sizes if ax not in red_set
        )
        new_dim_types = OrderedDict((ax, d.dim_types.get(ax)) for ax in new_dim_sizes)
        new_index_order = tuple(new_dim_sizes)
        return BaseTensorStats.from_fields(
            new_index_order, new_dim_sizes, new_fill, new_dim_types
        )

    @staticmethod
    def relabel_def(
        d: BaseTensorStats, relabel_indices: tuple[Field, ...]
    ) -> BaseTensorStats:
        if len(relabel_indices) != len(d.index_order):
            raise ValueError(
                f"Tensor has {len(d.index_order)} dims, "
                f"but {len(relabel_indices)} names provided."
            )

        new_dim_sizes = OrderedDict(
            zip(relabel_indices, d.dim_sizes.values(), strict=True)
        )
        new_dim_types = OrderedDict(
            zip(
                relabel_indices,
                (d.dim_types.get(ax) for ax in d.dim_sizes),
                strict=True,
            )
        )

        return BaseTensorStats.from_fields(
            relabel_indices, new_dim_sizes, d.fill_value, new_dim_types
        )

    @staticmethod
    def reorder_def(
        d: BaseTensorStats, reorder_indices: tuple[Field, ...]
    ) -> BaseTensorStats:
        for old_idx in d.index_order:
            if old_idx not in set(reorder_indices) and d.get_dim_size(old_idx) != 1:
                raise ValueError(
                    f"Trying to drop dimension '{old_idx}' of size"
                    f" {d.get_dim_size(old_idx)}."
                    " Only size 1 dimensions can be dropped."
                )

        new_dims: OrderedDict[Field, float] = OrderedDict()
        new_dim_types: OrderedDict[Field, FType | None] = OrderedDict()
        for idx in reorder_indices:
            if idx in d.index_order:
                new_dims[idx] = d.get_dim_size(idx)
                new_dim_types[idx] = d.dim_types.get(idx)
            else:
                new_dims[idx] = 1
                new_dim_types[idx] = None

        return BaseTensorStats.from_fields(
            reorder_indices, new_dims, d.fill_value, new_dim_types
        )
