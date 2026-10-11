from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from typing import TypeVar

from finch import finch_logic as lgc
from finch.algebra import (
    AbstractFill,
    FType,
    TensorFType,
    ftype,
    ftypes,
)
from finch.autoschedule.cost_models import (
    DenseLevelOption,
    FormatCostModel,
    SparseHashLevelOption,
)
from finch.autoschedule.tensor_stats import FDStats, StatsInterpreter
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import LogicLoader, MockLogicLoader, StatsFactory
from finch.finch_logic.tensor_stats import TensorStats
from finch.tensor import FiberTensorFType, dense, element, fiber_tensor, sparse_hash
from finch.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


# Builders for the level options of a FormatCostModel.
LEVEL_BUILDERS = {"dense": dense, "sparse_hash": sparse_hash}


def optimize_format(alias, shape_type, machine, fill_value, cost_model, term):
    stats = machine(alias)
    fields = stats.index_order
    n = len(fields)
    leaf = element(fill_value, ftype(fill_value))
    memo = {}

    def rec(level, num_pos):
        if level == n:
            return (
                cost_model.predict_format_cost(
                    alias, level, num_pos, None, term, machine
                ),
                leaf,
            )

        key = (level, num_pos)
        if key in memo:
            return memo[key]

        n_l = stats.get_dim_size(fields[level])
        nnz_l = cost_model.nnz_after(stats, machine.stats_factory, level)

        best_cost = None
        best_format = None
        for format_key, option in cost_model.formats.items():
            local = cost_model.predict_format_cost(
                alias, level, num_pos, format_key, term, machine
            )
            child_num_pos = option.next_num_pos(num_pos, n_l, nnz_l)
            child_cost, child_fmt = rec(level + 1, child_num_pos)
            total = local + child_cost
            if best_cost is None or total < best_cost:
                best_cost, best_format = (
                    total,
                    LEVEL_BUILDERS[format_key](child_fmt, shape_type[level]),
                )

        memo[key] = (best_cost, best_format)
        return best_cost, best_format

    _, fmt = rec(0, 1.0)
    return fmt


TS = TypeVar("TS", bound=TensorStats)


class SmartFormatter(LogicFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader

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
        stats: dict[lgc.Alias, TS],
        stats_factory: StatsFactory,
    ):
        bindings = bindings.copy()
        stats_bindings: OrderedDict[lgc.Alias, TS] = OrderedDict(stats)
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
                case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                    lgc.Table(lgc.Alias() as lhs, _), _, _
                ):
                    # Evaluating the query gives the stats of the stored result,
                    # in the order of its left-hand table.
                    rhs_stats = stats_interpreter(node, stats_bindings)
                    if not isinstance(rhs_stats, TensorStats):
                        raise TypeError("Expected query RHS to produce TensorStats.")
                    assert isinstance(rhs_stats, TensorStats)
                    stats_bindings[lhs] = rhs_stats  # ty: ignore[invalid-assignment]

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

                    return node
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
    ) -> FiberTensorFType:
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
                lvl = sparse_hash(lvl, shape_type[dim], single_writer=True)

        return fiber_tensor(lvl)


# Level features are positions, slots, entries, position bytes, index bytes,
# and value bytes.
ITER_COST_MODEL = FormatCostModel(
    {
        "dense": DenseLevelOption((0, 1, 0, 0, 0, 0)),
        "sparse_hash": SparseHashLevelOption((1, 0, 1, 0, 0, 0)),
    },
    (0, 0, 0, 0, 0, 0),
)
STORAGE_COST_MODEL = FormatCostModel(
    {
        "dense": DenseLevelOption((0, 0, 0, 0, 0, 0)),
        "sparse_hash": SparseHashLevelOption((0, 0, 0, 1, 1, 0)),
    },
    (0, 0, 0, 0, 0, 1),
)


class CostFormatter(LogicFormatter):
    """
    CostFormatter chooses the format of each alias with the lowest cost under
    cost_model, memoizing the best format for each depth and input position count.
    """

    def __init__(
        self,
        loader: LogicLoader | None = None,
        cost_model: FormatCostModel = STORAGE_COST_MODEL,
    ):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader
        self.cost_model = cost_model

    def get_tensor_ftype(
        self,
        alias: lgc.Alias,
        term: lgc.LogicStatement,
        machine: StatsMachine,
        fill_value: AbstractFill,
        shape_type: tuple[FType, ...],
    ) -> FiberTensorFType:
        lvl = optimize_format(
            alias, shape_type, machine, fill_value, self.cost_model, term
        )
        return fiber_tensor(lvl)

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TS],
        stats_factory: StatsFactory,
    ):
        bindings = bindings.copy()
        machine = StatsMachine(stats_factory, OrderedDict(stats))
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement):
            match node:
                case lgc.Plan(bodies):
                    for body in bodies:
                        formatter(body)
                case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                    lgc.Table(lgc.Alias() as lhs, _), _, _
                ):
                    machine(node)
                    if lhs not in bindings:
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftypes.intp
                            for dim in shape_types[lhs]
                        )
                        bindings[lhs] = self.get_tensor_ftype(
                            lhs, node, machine, fill_values[lhs], shape_type
                        )
                case lgc.Produces():
                    return
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)

        logger.debug(prgm)

        return self.ctx(prgm, bindings, machine.bindings, stats_factory)


class StorageCostFormatter(CostFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader, STORAGE_COST_MODEL)


class IterCostFormatter(CostFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader, ITER_COST_MODEL)
