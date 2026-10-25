from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from finch import finch_logic as lgc
from finch.algebra import (
    AbstractFillFType,
    FType,
    TensorFType,
    ftype,
    ftypes,
    np_dtype,
)
from finch.autoschedule.cost_models import (
    ByKind,
    CostModel,
    DenseLevelOption,
    ElementLevelOption,
    LevelChoice,
    LevelCostModel,
    LevelOption,
    LinearPredictor,
    SparseHashLevelOption,
)
from finch.autoschedule.cost_models.format_cost_models import alias_blocks
from finch.autoschedule.tensor_stats import FDStats, StatsInterpreter
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import LogicLoader, MockLogicLoader, StatsFactory
from finch.finch_logic.tensor_stats import TensorStats
from finch.tensor import FiberTensorFType, dense, element, fiber_tensor, sparse_hash
from finch.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


# The level options CostFormatter chooses from by default, and their builders.
LEVEL_OPTIONS = (DenseLevelOption(), SparseHashLevelOption())
LEVEL_BUILDERS = {"dense": dense, "sparse_hash": sparse_hash}


def optimize_format(
    alias, shape_type, machine, fill_type, cost_model, term, options=LEVEL_OPTIONS
):
    blocks = alias_blocks(alias, term, machine)
    n = len(shape_type)
    position_type = ftypes.intp
    leaf = element(fill_type, fill_type.element_type, position_type)
    pos_size = np_dtype(position_type).itemsize
    memo = {}

    def cost(level, num_pos, option):
        choice = LevelChoice(term, alias, level, num_pos, option, pos_size)
        return cost_model.predict_component_cost(choice, machine)

    def rec(level, num_pos):
        if level == n:
            return cost(level, num_pos, ElementLevelOption()), leaf

        key = (level, num_pos)
        if key in memo:
            return memo[key]

        best_cost = None
        best_format = None
        for option in options:
            local = cost(level, num_pos, option)
            child_num_pos = tuple(
                option.next_num_pos(
                    stats, machine.stats_factory, len(over) + level, n, over
                )
                for (stats, over), n in zip(blocks, num_pos, strict=True)
            )
            child_cost, child_fmt = rec(level + 1, child_num_pos)
            total = local + child_cost
            if best_cost is None or total < best_cost:
                best_cost, best_format = (
                    total,
                    LEVEL_BUILDERS[option.kind](child_fmt, shape_type[level]),
                )

        memo[key] = (best_cost, best_format)
        return best_cost, best_format

    _, fmt = rec(0, (1.0,) * len(blocks))
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
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
        stats: TensorStats,
        over: tuple[lgc.Field, ...] = (),
    ) -> TensorFType:
        """The format of a tensor with the given stats. If `over` lists
        fields, each tensor stores one slice which fixes those fields."""

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
        shape_types: dict[lgc.Alias, tuple[FType | None, ...]] = {
            var: val.shape_type for var, val in bindings.items()
        }
        fill_types = prgm.infer_fill_type(
            {var: val.fill_type for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement) -> lgc.LogicStatement:
            match node:
                case lgc.Plan(bodies):
                    return lgc.Plan(tuple(formatter(body) for body in bodies))
                case lgc.Fuse(idx, body):
                    return lgc.Fuse(idx, formatter(body))
                case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                    lgc.Table(lgc.Alias() as lhs, _), _, _
                ):
                    node.infer_shape_type(shape_types)
                    # Evaluating the query gives the stats of the stored result,
                    # in the order of its left-hand table.
                    rhs_stats = stats_interpreter(node, stats_bindings)
                    if not isinstance(rhs_stats, TensorStats):
                        raise TypeError("Expected query RHS to produce TensorStats.")
                    assert isinstance(rhs_stats, TensorStats)
                    stats_bindings[lhs] = rhs_stats  # ty: ignore[invalid-assignment]

                    alias, nfused = lhs.unfused, lhs.nfused
                    if alias not in bindings:
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftypes.intp
                            for dim in shape_types[alias][nfused:]
                        )
                        bindings[alias] = self.get_tensor_ftype(
                            fill_types[alias],
                            shape_type,
                            rhs_stats,
                            rhs_stats.index_order[:nfused],
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
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
        stats: TensorStats,
        over: tuple[lgc.Field, ...] = (),
    ) -> FiberTensorFType:
        if not isinstance(stats, FDStats):
            raise TypeError("FDFormatter requires FDStats.")
        fields = tuple(idx for idx in stats.index_order if idx not in over)
        if len(shape_type) != len(fields):
            raise ValueError(
                f"Got {len(shape_type)} shape dimensions for "
                f"{len(fields)} stats dimensions."
            )

        # Slices are formatted by the support of every slice, so the fields of
        # `over` are dropped from the dense dimensions.
        fill_ftype = fill_type.element_type
        lvl = element(fill_type, fill_ftype)
        for dim in reversed(range(len(fields))):
            field = fields[dim]
            outer_fields = frozenset(fields[:dim])
            required_fields = outer_fields | {field}
            is_dense = any(
                required_fields.issubset(dense_fields)
                for dense_fields in stats.dense_props
            )
            if is_dense:
                lvl = dense(lvl, shape_type[dim])
            else:
                lvl = sparse_hash(lvl, shape_type[dim])

        return fiber_tensor(lvl)


def linear_level_model(
    option_type: type[LevelOption],
    storage: Sequence[float],
    access: Sequence[float] = (0, 0, 0, 0),
) -> LevelCostModel:
    coefficients = np.array([*storage, *access], dtype=float)
    return LevelCostModel(option_type, LinearPredictor(coefficients))


# Each level option lists its storage features. The access features are
# sequential reads, scatter reads, sequential writes, and scatter writes.
ITER_COST_MODEL = ByKind(
    {
        "dense": linear_level_model(DenseLevelOption, (0, 1)),
        "sparse_hash": linear_level_model(SparseHashLevelOption, (1, 1, 0, 0)),
        "element": linear_level_model(ElementLevelOption, (0, 0)),
    }
)
STORAGE_COST_MODEL = ByKind(
    {
        "dense": linear_level_model(DenseLevelOption, (0, 0)),
        "sparse_hash": linear_level_model(SparseHashLevelOption, (0, 0, 1, 1)),
        "element": linear_level_model(ElementLevelOption, (0, 1)),
    }
)


class CostFormatter(LogicFormatter):
    """
    CostFormatter chooses the format of each alias from the level options with
    the lowest cost under cost_model, memoizing the best format for each depth and
    input position count. The cost of a level is summed over the blocks of its
    alias in the program.
    """

    def __init__(
        self,
        loader: LogicLoader | None = None,
        cost_model: CostModel[LevelChoice] = STORAGE_COST_MODEL,
        options: Sequence[LevelOption] = LEVEL_OPTIONS,
    ):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader
        self.cost_model = cost_model
        self.options = options

    def get_tensor_ftype(
        self,
        alias: lgc.Alias,
        term: lgc.LogicStatement,
        machine: StatsMachine,
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
    ) -> FiberTensorFType:
        lvl = optimize_format(
            alias, shape_type, machine, fill_type, self.cost_model, term, self.options
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
        shape_types: dict[lgc.Alias, tuple[FType | None, ...]] = {
            var: val.shape_type for var, val in bindings.items()
        }
        fill_types = prgm.infer_fill_type(
            {var: val.fill_type for var, val in bindings.items()}
        )
        # The dimensions each new alias stores, at its first definition.
        stored_shape_types: dict[lgc.Alias, tuple[FType, ...]] = {}

        def formatter(node: lgc.LogicStatement):
            match node:
                case lgc.Plan(bodies):
                    for body in bodies:
                        formatter(body)
                case lgc.Fuse(_, body):
                    formatter(body)
                case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                    lgc.Table(lgc.Alias() as lhs, _), _, _
                ):
                    node.infer_shape_type(shape_types)
                    machine(node)
                    alias, nfused = lhs.unfused, lhs.nfused
                    if alias not in bindings and alias not in stored_shape_types:
                        stored_shape_types[alias] = tuple(
                            ftype(dim) if dim is not None else ftypes.intp
                            for dim in shape_types[alias][nfused:]
                        )
                case lgc.Produces():
                    return
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)
        # Each alias is formatted for all of its blocks in the program, which
        # are replayed from the stats before it.
        initial = StatsMachine(stats_factory, OrderedDict(stats))
        for alias, shape_type in stored_shape_types.items():
            bindings[alias] = self.get_tensor_ftype(
                alias, prgm, initial, fill_types[alias], shape_type
            )

        logger.debug(prgm)

        return self.ctx(prgm, bindings, machine.bindings, stats_factory)


class StorageCostFormatter(CostFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader, STORAGE_COST_MODEL)


class IterCostFormatter(CostFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader, ITER_COST_MODEL)
