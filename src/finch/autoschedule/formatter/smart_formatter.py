from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from itertools import product
from typing import TypeVar

import numpy as np

from finch import finch_logic as lgc
from finch.algebra import (
    AbstractFillFType,
    FType,
    TensorFType,
    ftype,
    ftypes,
)
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    CostModel,
    LevelOption,
    ProtocolComponent,
    allowed_options,
    level_formats,
    read_protocols,
    storage_choices,
    write_protocols,
)
from finch.autoschedule.cost_models.protocol_cost_models import KINDS, STORAGE_KINDS
from finch.autoschedule.protocolizer.protocolize import choose_protocols
from finch.autoschedule.tensor_stats import FDStats, StatsInterpreter
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import LogicLoader, MockLogicLoader, StatsFactory
from finch.finch_logic.tensor_stats import TensorStats
from finch.tensor import (
    FiberTensorFType,
    dense,
    element,
    fiber_tensor,
    sparse_bytemap,
    sparse_hash,
    sparse_list,
)
from finch.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


# The level options CostFormatter chooses from by default, and their builders.
# Of options which cost the same, the earliest is chosen, so a list is chosen
# over a hash wherever it can be written.
LEVEL_OPTIONS = KINDS
LEVEL_BUILDERS = {
    "dense": dense,
    "sparse_list": sparse_list,
    "sparse_hash": sparse_hash,
    "sparse_bytemap": sparse_bytemap,
}


def format_ftype(
    options: Sequence[LevelOption], fill_type: AbstractFillFType, shape_type
) -> FiberTensorFType:
    """The format of a tensor of the level options, fill_type, and shape_type."""
    lvl = element(fill_type, fill_type.element_type, ftypes.intp)
    for option, dim in reversed(list(zip(options, shape_type, strict=True))):
        lvl = LEVEL_BUILDERS[option.kind](lvl, dim)
    return fiber_tensor(lvl)


def optimize_formats(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    stored: Mapping[lgc.Alias, tuple[FType, ...]],
    fill_types: Mapping[lgc.Alias, AbstractFillFType],
    cost_model: CostModel[ProtocolComponent] = PROTOCOL_COST_MODEL,
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
) -> dict[lgc.Alias, TensorFType]:
    """
    The format of each alias of stored, whose levels have the shape types it
    gives and the fill of fill_types, under which prgm costs least by
    cost_model, with its other tensors formatted as bindings says, by
    coordinate descent: each alias in turn tries every product of the options
    its levels allow, re-choosing protocols only for the queries which touch
    it, for two sweeps or until none changes. Each starts from the first
    allowed option of each level which filters.
    """
    allowed = {
        alias: [
            allowed_options(alias, depth, prgm, options)
            for depth in range(len(shape_type))
        ]
        for alias, shape_type in stored.items()
    }
    for alias, levels in allowed.items():
        for depth, level_options in enumerate(levels):
            if not level_options:
                raise ValueError(
                    f"No level option supports reading level {depth} of {alias} "
                    f"with {read_protocols(alias, depth, prgm)} and writing it "
                    f"with {write_protocols(alias, depth, prgm)}."
                )

    def ftype_of(alias, kinds):
        return format_ftype(kinds, fill_types[alias], stored[alias])

    formats = {
        alias: ftype_of(
            alias,
            [
                next((option for option in level if option.filters), level[0])
                for level in levels
            ],
        )
        for alias, levels in allowed.items()
    }

    def storage_cost(alias, fmt):
        machine = StatsMachine(stats_factory, OrderedDict(stats))
        options = level_formats({alias: fmt})[alias]
        return sum(
            cost_model.predict_component_cost(choice, machine)
            for choice in storage_choices(alias, options, prgm, machine)
        )

    bindings = {**bindings, **formats}
    storage = {alias: storage_cost(alias, fmt) for alias, fmt in formats.items()}
    _, costs = choose_protocols(prgm, bindings, stats, stats_factory, cost_model)
    # A second sweep lets each alias answer the formats chosen after it.
    for _ in range(2):
        changed = False
        for alias, levels in allowed.items():
            best = (sum(costs) + storage[alias], bindings[alias], costs, storage[alias])
            for kinds in product(*levels):
                fmt = ftype_of(alias, kinds)
                if fmt == bindings[alias]:
                    continue
                _, costs_2 = choose_protocols(
                    prgm,
                    {**bindings, alias: fmt},
                    stats,
                    stats_factory,
                    cost_model,
                    only={alias},
                    costs=costs,
                )
                storage_2 = storage_cost(alias, fmt)
                if sum(costs_2) + storage_2 < best[0]:
                    best = (sum(costs_2) + storage_2, fmt, costs_2, storage_2)
            _, best_fmt, best_costs, best_storage = best
            if best_fmt != bindings[alias]:
                bindings[alias], costs, storage[alias] = (
                    best_fmt,
                    best_costs,
                    best_storage,
                )
                changed = True
        if not changed:
            break
    return {alias: bindings[alias] for alias in stored}


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


class CostFormatter(LogicFormatter):
    """
    CostFormatter chooses the format of each alias the program defines from
    the level options, as optimize_formats does under cost_model, a model of
    the protocol basis, which prices the loops of the program with the
    protocols chosen for each format. The inputs keep the formats of their
    bindings.
    """

    def __init__(
        self,
        loader: LogicLoader | None = None,
        cost_model: CostModel[ProtocolComponent] = PROTOCOL_COST_MODEL,
        options: Sequence[LevelOption] = LEVEL_OPTIONS,
    ):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader
        self.cost_model = cost_model
        self.options = options

    def choose_formats(
        self,
        prgm: lgc.LogicStatement,
        bindings: Mapping[lgc.Alias, TensorFType],
        stats: Mapping[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
        stored: Mapping[lgc.Alias, tuple[FType, ...]],
        fill_types: Mapping[lgc.Alias, AbstractFillFType],
    ) -> dict[lgc.Alias, TensorFType]:
        """The format of each alias of stored, whose levels have the shape
        types it gives, from the stats before prgm."""
        return optimize_formats(
            prgm,
            bindings,
            stats,
            stats_factory,
            stored,
            fill_types,
            self.cost_model,
            self.options,
        )

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
        # The dimensions each new alias stores, at its first definition, and
        # the rank each alias stores, which no definition may change.
        stored_shape_types: dict[lgc.Alias, tuple[FType, ...]] = {}
        ranks = {var: len(val.shape_type) for var, val in bindings.items()}

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
                    shape_type = tuple(
                        ftype(dim) if dim is not None else ftypes.intp
                        for dim in shape_types[alias][nfused:]
                    )
                    if ranks.setdefault(alias, len(shape_type)) != len(shape_type):
                        raise ValueError(
                            f"Cannot change the rank of allocated tensor {alias}."
                        )
                    if alias not in bindings and alias not in stored_shape_types:
                        stored_shape_types[alias] = shape_type
                case lgc.Produces():
                    return
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)
        bindings.update(
            self.choose_formats(
                prgm, bindings, stats, stats_factory, stored_shape_types, fill_types
            )
        )

        logger.debug(prgm)

        return self.ctx(prgm, bindings, machine.bindings, stats_factory)


# Costs are in resets of a dense slot, as fit to Gustavson SpGEMM on Finch.jl.
# A dense level zeroes each of its slots, and a workspace zeroes them again for
# each slice, which the copy out of it counts by following every slot of every
# slice; it takes writes in any order almost for free. A byte map allocates a
# flag and a child for each of its slots, like a dense level, but only clears
# and sorts the distinct entries written to each slice, as if each slice held
# 16 entries to sort. A hash pays for a probe at every write and lookup, and a
# sorted list appends cheaply but binary searches to look up out of order. A
# walk costs 1 for each entry it steps through, and a gallop 4 for each jump.
_WORKSPACE_STORAGE_COSTS: dict[str, tuple[float, ...]] = {
    "dense": (0, 1),
    "sparse_list": (0, 0, 0, 0),
    "sparse_hash": (0, 0, 0, 0),
    "sparse_bytemap": (0, 1, 0, 0),
    "element": (0, 0),
}
_BYTEMAP_CLEAR = 1 + 1.6 * np.log2(16)
# The follow, walk, gallop, extrude, and laminate visits and the distinct writes
# of each kind of level.
_WORKSPACE_ACCESS_COSTS: dict[str, tuple[float, ...]] = {
    "dense": (1 + 1, 1, 0, 0.1, 0.1, 0),
    "sparse_list": (8, 1, 4, 1.3, 1.3, 0),
    "sparse_hash": (15, 1, 0, 15, 15, 0),
    "sparse_bytemap": (1, 1, 4, 1, 1, _BYTEMAP_CLEAR),
}


def _workspace_access_costs(kind: str) -> tuple[float, ...]:
    follow, walk, gallop, extrude, laminate, writes = _WORKSPACE_ACCESS_COSTS[kind]
    return (0, 0, 0, 0, follow, walk, gallop, 0, 0, extrude, laminate, writes)


WORKSPACE_COST_MODEL = CostModel(
    PROTOCOL_BASIS,
    np.concatenate(
        [
            *(_WORKSPACE_STORAGE_COSTS[option.kind] for option in STORAGE_KINDS),
            *(_workspace_access_costs(option.kind) for option in KINDS),
            [0.0],
        ]
    ),
)


class WorkspaceCostFormatter(CostFormatter):
    """
    Choose each format, including the format of each workspace, pricing
    scattered writes and the reset of each slice by level.
    """

    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader, WORKSPACE_COST_MODEL)
