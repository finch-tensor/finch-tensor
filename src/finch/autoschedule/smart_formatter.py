from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from finch import finch_logic as lgc
from finch.algebra import AbstractFill, FType, TensorFType, ffuncs, ftype, ftypes
from finch.finch_logic import LogicLoader, StatsFactory
from finch.finch_logic.tensor_stats import TensorStats
from finch.tensor import dense, element, fiber_tensor, sparse_hash
from finch.tensor.level import DenseLevelFType, SparseHashLevelFType
from finch.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter
from .tensor_stats import FDStats, StatsInterpreter

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


def nnz_after(fields, stats, stats_factory, level):
    reduce_fields = tuple(fields[level + 1 :])
    if reduce_fields:
        reduced = stats_factory.aggregate(ffuncs.or_, False, reduce_fields, stats)
    else:
        reduced = stats
    return reduced.estimate_non_fill_values()


def optimize_format(
    fields,
    shape_type,
    stats,
    stats_factory,
    fill_value,
    candidates,
    cost_of,
    leaf_cost_fn,
):
    n = len(fields)
    fill_ftype = ftype(fill_value)
    leaf = element(fill_value, fill_ftype)
    val_size = np.dtype(fill_ftype.dtype).itemsize
    pos_size = np.dtype(leaf.position_type.dtype).itemsize

    # memoizing (l,nnz) : (best_cost,best_fmt)
    memo = {}

    def rec(level, num_pos):
        """
        We are trying to keep this as :
        def optimise_fmt(S,n,p):
            d_cost = C(S,(D,optimise_fmt(S,n-1,p*n_l)))
            s_cost = C(S,(D,optimise_fmt(S,n-1,q)))

            if d_cost < s_cost :
                return D, optimise_fmt(S,n-1,p*n_l)
            else:
                return D, optimise_fmt(S,n-1,q)
        """
        if level == n:
            return leaf_cost_fn(num_pos, val_size, pos_size), leaf

        key = (level, num_pos)
        if key in memo:
            return memo[key]

        n_l = stats.get_dim_size(fields[level])
        nnz_l = nnz_after(fields, stats, stats_factory, level)

        best_cost = None
        best_format = None
        for option in candidates:
            local = cost_of(option)(num_pos, n_l, nnz_l, val_size, pos_size)
            child_num_pos = option.next_num_pos(num_pos, n_l, nnz_l)
            child_cost, child_fmt = rec(level + 1, child_num_pos)
            total = local + child_cost
            if best_cost is None or total < best_cost:
                best_cost, best_format = (
                    total,
                    option.build(child_fmt, shape_type[level]),
                )

        memo[key] = (best_cost, best_format)
        return best_cost, best_format

    _, fmt = rec(0, 1.0)
    return fmt


def total_tree_cost(
    lvl, fields, stats, stats_factory, num_pos, level, candidates, cost_of, leaf_cost_fn
):
    val_size = np.dtype(ftype(lvl.fill_value).dtype).itemsize
    pos_size = np.dtype(ftype(lvl.position_type).dtype).itemsize

    if level == len(fields):
        return leaf_cost_fn(num_pos, val_size, pos_size)

    option = next(o for o in candidates if o.level_type is type(lvl))
    n_l = stats.get_dim_size(fields[level])
    nnz_l = nnz_after(fields, stats, stats_factory, level)
    local_cost = cost_of(option)(num_pos, n_l, nnz_l, val_size, pos_size)
    child_num_pos = option.next_num_pos(num_pos, n_l, nnz_l)
    return local_cost + total_tree_cost(
        lvl.lvl_t,
        fields,
        stats,
        stats_factory,
        child_num_pos,
        level + 1,
        candidates,
        cost_of,
        leaf_cost_fn,
    )


class SmartFormatter(LogicFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader)

    @abstractmethod
    def get_tensor_ftype(
        self,
        fill_value: AbstractFill,
        shape_type: tuple[FType, ...],
        stats: TensorStats,
    ) -> TensorFType: ...

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        bindings = bindings.copy()
        stats_bindings: OrderedDict[lgc.Alias, TensorStats] = OrderedDict(stats)
        stats_interpreter = StatsInterpreter(stats_factory=stats_factory)
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement) -> lgc.LogicStatement:
            match node:
                case lgc.Plan(bodies):
                    return lgc.Plan(tuple(formatter(body) for body in bodies))
                case lgc.Query(lhs, rhs):
                    rhs_stats = stats_interpreter(rhs, stats_bindings)
                    if not isinstance(rhs_stats, TensorStats):
                        raise TypeError("Expected query RHS to produce TensorStats.")
                    stats_bindings[lhs] = rhs_stats

                    if lhs not in bindings:
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftypes.intp
                            for dim in shape_types[lhs]
                        )
                        bindings[lhs] = self.get_tensor_ftype(
                            fill_values[lhs],
                            shape_type,
                            rhs_stats,
                        )

                    match rhs:
                        case lgc.Reorder():
                            return node
                        case _:
                            return lgc.Query(lhs, lgc.Reorder(rhs, rhs.fields()))
                case lgc.Produces():
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        prgm = formatter(prgm)

        logger.debug(prgm)

        return self.ctx(prgm, bindings, stats_bindings, stats_factory)


class FDFormatter(SmartFormatter):
    def get_tensor_ftype(
        self,
        fill_value: AbstractFill,
        shape_type: tuple[FType, ...],
        stats: TensorStats,
    ) -> TensorFType:
        if not isinstance(stats, FDStats):
            raise TypeError("FDFormatter requires FDStats.")
        if len(shape_type) != len(stats.index_order):
            raise ValueError(
                f"Got {len(shape_type)} shape dimensions for "
                f"{len(stats.index_order)} stats dimensions."
            )

        fill_ftype = ftype(fill_value)
        lvl = element(fill_value, fill_ftype)
        for dim in reversed(range(len(stats.index_order))):
            field = stats.index_order[dim]
            outer_fields = frozenset(stats.index_order[:dim])
            required_fields = outer_fields | {field}
            is_dense = any(
                required_fields.issubset(dense_fields)
                for dense_fields in stats.dense_props
            )
            if is_dense:
                lvl = dense(lvl, shape_type[dim])
            else:
                lvl = sparse_hash(lvl, shape_type[dim], single_writer=False)

        return fiber_tensor(lvl)


@dataclass(frozen=True)
class LevelOption:
    level_type: type
    build: Callable
    next_num_pos: Callable
    storage_cost_fn: Callable
    iter_cost_fn: Callable


def dense_next_num_pos(num_pos, n_l, nnz_l):
    return num_pos * n_l


def sparse_hash_next_num_pos(num_pos, n_l, nnz_l):
    return nnz_l


def build_dense(lvl, dim_type):
    return dense(lvl, dim_type)


def build_sparse_hash(lvl, dim_type):
    return sparse_hash(lvl, dim_type, single_writer=False)


CANDIDATES = (
    LevelOption(
        level_type=DenseLevelFType,
        build=build_dense,
        next_num_pos=dense_next_num_pos,
        storage_cost_fn=lambda num_pos, n_l, nnz_l, val_size, pos_size: 0.0,
        iter_cost_fn=lambda num_pos, n_l, nnz_l, val_size, pos_size: num_pos * n_l,
    ),
    LevelOption(
        level_type=SparseHashLevelFType,
        build=build_sparse_hash,
        next_num_pos=sparse_hash_next_num_pos,
        storage_cost_fn=(
            lambda num_pos, n_l, nnz_l, val_size, pos_size: (
                (num_pos + 1) * pos_size + nnz_l * pos_size
            )
        ),
        iter_cost_fn=lambda num_pos, n_l, nnz_l, val_size, pos_size: num_pos + nnz_l,
    ),
)


class CostFormatter(SmartFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader)
        self._stats_factory = None

    @abstractmethod
    def leaf_cost_fn(self, num_pos, val_size, pos_size): ...

    def lower(self, prgm, bindings, stats, stats_factory):
        """
        get_tensor_ftype is called by lower which needs stats_factory which
        isn't passed by smart formatter lower
        """
        self._stats_factory = stats_factory
        return super().lower(prgm, bindings, stats, stats_factory)

    def get_tensor_ftype(self, fill_value, shape_type, stats):
        if self._stats_factory is None:
            raise ValueError("CostFormatter requires StatsFactory")
        lvl = optimize_format(
            stats.index_order,
            shape_type,
            stats,
            self._stats_factory,
            fill_value,
            CANDIDATES,
            self.cost_of,
            self.leaf_cost_fn,
        )
        return fiber_tensor(lvl)


class StorageCostFormatter(CostFormatter):
    def cost_of(self, option: LevelOption):
        return option.storage_cost_fn

    def leaf_cost_fn(self, num_pos, val_size, pos_size):
        return num_pos * val_size


class IterCostFormatter(CostFormatter):
    def cost_of(self, option: LevelOption):
        return option.iter_cost_fn

    def leaf_cost_fn(self, num_pos, val_size, pos_size):
        return 0.0
