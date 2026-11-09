from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Self

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
    MapJoin,
    Relabel,
    Reorder,
    StatsFactory,
)

from .bound_stats import DCStatsFactory
from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class MemoNode:
    """
    The inner statistic of an expression, with its estimates, and the nodes of
    the expressions built from it with it as their first argument, under their
    operation and their other arguments. Every MemoStats of the expression
    shares its node, so a copy of one shares it too.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.estimates: dict[frozenset[Field], float] = {}
        self.derived: dict[tuple, MemoNode] = {}

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        return self


class MemoStatsFactory(
    BaseTensorStatsFactory["MemoStats"],
    StatsFactory["MemoStats"],
):
    """
    Wraps an eager statistic `inner`, such as DC, so that the statistics of an
    expression are computed once however many times it is built. Each tensor is
    converted to its support, and the statistics of an expression are built from
    those with and, or, exists and forall. The statistics of an expression are
    kept by those of its first argument, so they last as long as the stats of
    the inputs they are built from, however often the expression is dropped and
    built again.
    """

    def __init__(self, inner: StatsFactory[Any] | None = None):
        super().__init__(MemoStats)
        self.inner = DCStatsFactory() if inner is None else inner
        # Tensors are unhashable, so the inner statistic of a tensor is cached
        # by id until it dies, and the inner factory reads each tensor once.
        self.leaves: dict[int, tuple[BaseTensorStats, Any]] = {}

    def _derive(
        self,
        base: BaseTensorStats,
        args: Sequence[MemoStats],
        key: tuple,
        compute: Callable[[], Any],
    ) -> MemoStats:
        # Nodes are shared, so an expression is found by the identity of the
        # nodes of its arguments, without hashing it.
        first, *rest = (s.node for s in args)
        key = (*key, *rest)
        if key not in first.derived:
            first.derived[key] = MemoNode(compute())
        return MemoStats(base, first.derived[key])

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> MemoStats:
        # The inner statistic describes the tensor, which needn't have an ftype,
        # so its base is kept with its support.
        if id(tensor) not in self.leaves:
            stats = self.inner(tensor, fields)
            self.leaves[id(tensor)] = (
                BaseTensorStats(stats),
                self.inner.support(stats),
            )
            weakref.finalize(tensor, self.leaves.pop, id(tensor), None)
        base, support = self.leaves[id(tensor)]
        # Each call makes new stats, which keep what is built from them.
        return MemoStats(
            self.relabel_def(base, fields),
            MemoNode(self.inner.relabel(support, fields)),
        )

    def _mapjoin_join(self, op: FinchOperator, *join_args: MemoStats) -> MemoStats:
        base = super()._mapjoin_defs(op, *join_args)
        return self._derive(
            base,
            join_args,
            (MapJoin, ffuncs.and_),
            lambda: self.inner.mapjoin(ffuncs.and_, *(s.inner for s in join_args)),
        )

    def _mapjoin_union(self, op: FinchOperator, *union_args: MemoStats) -> MemoStats:
        base = super()._mapjoin_defs(op, *union_args)
        return self._derive(
            base,
            union_args,
            (MapJoin, ffuncs.or_),
            lambda: self.inner.mapjoin(ffuncs.or_, *(s.inner for s in union_args)),
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
        # TODO: an or from True is every entry, but `False` is the identity of
        # or, so the inner statistics count it as an exists.
        return self._derive(
            base,
            (stats,),
            (Aggregate, bool_op, bool_init, reduce_indices),
            lambda: self.inner.aggregate(
                bool_op, bool_init, reduce_indices, stats.inner
            ),
        )

    def relabel(
        self, stats: MemoStats, relabel_indices: tuple[Field, ...]
    ) -> MemoStats:
        base = self.relabel_def(stats, relabel_indices)
        return self._derive(
            base,
            (stats,),
            (Relabel, relabel_indices),
            lambda: self.inner.relabel(stats.inner, relabel_indices),
        )

    def reorder(
        self, stats: MemoStats, reorder_indices: tuple[Field, ...]
    ) -> MemoStats:
        base = self.reorder_def(stats, reorder_indices)
        return self._derive(
            base,
            (stats,),
            (Reorder, reorder_indices),
            lambda: self.inner.reorder(stats.inner, reorder_indices),
        )


class MemoStats(NumericStats):
    def __init__(self, base: BaseTensorStats, node: MemoNode):
        super().__init__(base)
        self.node = node

    @property
    def inner(self) -> Any:
        return self.node.inner

    def estimate_non_fill_values(self, over: Iterable[Field] = ()) -> float:
        over = frozenset(over)
        estimates = self.node.estimates
        if over not in estimates:
            estimates[over] = self.inner.estimate_non_fill_values(over)
        return estimates[over]

    def get_embedding(self) -> np.ndarray:
        return self.inner.get_embedding()
