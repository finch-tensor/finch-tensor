from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from finch.algebra import (
    FinchOperator,
    ffuncs,
    is_annihilator,
    is_idempotent,
    is_identity,
)
from finch.finch_logic import (
    Aggregate,
    Field,
    Literal,
    LogicExpression,
    LogicNode,
    MapJoin,
    Relabel,
    Reorder,
    StatsFactory,
    Table,
)

from .bound_stats import DCStatsFactory
from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class MemoStatsFactory(
    BaseTensorStatsFactory["MemoStats"],
    StatsFactory["MemoStats"],
):
    """
    Wraps an eager statistic `inner`, such as DC, so that the statistics of an
    expression are computed once however many times it is built. Each tensor is
    converted to its support, and the statistics of an expression are built from
    those with and, or, exists and forall.
    """

    def __init__(self, inner: StatsFactory[Any] | None = None):
        super().__init__(MemoStats)
        self.inner = DCStatsFactory() if inner is None else inner
        # Tensors are unhashable, so a leaf is cached by id until its tensor dies.
        self.leaves: dict[int, Any] = {}
        self.memo: WeakKeyDictionary[LogicNode, Any] = WeakKeyDictionary()

    def _memo(
        self, base: BaseTensorStats, expr: LogicExpression, method, *args
    ) -> MemoStats:
        # The inner statistic of expr is `method(*args)`, computed only once.
        if expr not in self.memo:
            self.memo[expr] = method(*args)
        return MemoStats(base, expr, self.memo[expr])

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> MemoStats:
        base = super().__call__(tensor, fields)
        if id(tensor) not in self.leaves:
            self.leaves[id(tensor)] = self.inner.support(self.inner(tensor, fields))
            weakref.finalize(tensor, self.leaves.pop, id(tensor), None)
        return self._memo(
            base,
            Table(Literal(tensor), fields),
            self.inner.relabel,
            self.leaves[id(tensor)],
            fields,
        )

    def _mapjoin_join(self, op: FinchOperator, *join_args: MemoStats) -> MemoStats:
        base = super()._mapjoin_defs(op, *join_args)
        expr = MapJoin(Literal(ffuncs.and_), tuple(s.expr for s in join_args))
        return self._memo(
            base, expr, self.inner.mapjoin, ffuncs.and_, *(s.inner for s in join_args)
        )

    def _mapjoin_union(self, op: FinchOperator, *union_args: MemoStats) -> MemoStats:
        base = super()._mapjoin_defs(op, *union_args)
        expr = MapJoin(Literal(ffuncs.or_), tuple(s.expr for s in union_args))
        return self._memo(
            base, expr, self.inner.mapjoin, ffuncs.or_, *(s.inner for s in union_args)
        )

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: MemoStats,
    ) -> MemoStats:
        # An output is fill when all of its inputs are, as in an or, unless the
        # fill annihilates op, as in an and, or the init joins a fill which
        # repeats once for each reduced element, as in a sum.
        f = stats.fill_type
        if is_identity(op.ftype, f):
            bool_op, bool_init = ffuncs.or_, False
        elif is_annihilator(op.ftype, f):
            bool_op, bool_init = ffuncs.and_, True
        else:
            bool_op, bool_init = (
                ffuncs.or_,
                init is not None and not is_idempotent(op.ftype),
            )

        base = self.aggregate_def(op, init, reduce_indices, stats)
        expr = Aggregate(
            Literal(bool_op), Literal(bool_init), stats.expr, reduce_indices
        )
        # TODO: an or from True is every entry, but `False` is the identity of
        # or, so the inner statistics count it as an exists.
        return self._memo(
            base,
            expr,
            self.inner.aggregate,
            bool_op,
            bool_init,
            reduce_indices,
            stats.inner,
        )

    def relabel(
        self, stats: MemoStats, relabel_indices: tuple[Field, ...]
    ) -> MemoStats:
        base = self.relabel_def(stats, relabel_indices)
        expr = Relabel(stats.expr, relabel_indices)
        return self._memo(base, expr, self.inner.relabel, stats.inner, relabel_indices)

    def reorder(
        self, stats: MemoStats, reorder_indices: tuple[Field, ...]
    ) -> MemoStats:
        base = self.reorder_def(stats, reorder_indices)
        expr = Reorder(stats.expr, reorder_indices)
        return self._memo(base, expr, self.inner.reorder, stats.inner, reorder_indices)


class MemoStats(NumericStats):
    def __init__(self, base: BaseTensorStats, expr: LogicExpression, inner: Any):
        super().__init__(base)
        self.expr = expr
        self.inner = inner

    def estimate_non_fill_values(self, over: Iterable[Field] = ()) -> float:
        return self.inner.estimate_non_fill_values(over)

    def get_embedding(self) -> np.ndarray:
        return self.inner.get_embedding()
