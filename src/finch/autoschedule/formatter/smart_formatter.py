from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
    PROTOCOL_COST_MODEL,
    STORAGE_BASIS,
    CostModel,
    DenseLevelOption,
    ElementLevelOption,
    LevelOption,
    ProtocolChoice,
    SparseListLevelOption,
    allowed_options,
    read_protocols,
    write_protocols,
)
from finch.autoschedule.cost_models.protocol_cost_models import FIELD_KINDS, accesses
from finch.autoschedule.protocolizer.protocolize import alias_costs, choose_protocols
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

from .formatter import LogicFormatter, scalar_ftype, stores_nothing

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


# The level options CostFormatter chooses from by default, and their builders.
# Of options which cost the same, the earliest is chosen, so a list is chosen
# over a hash wherever it can be written.
LEVEL_OPTIONS = FIELD_KINDS
LEVEL_BUILDERS = {
    "dense": dense,
    "sparse_list": sparse_list,
    "sparse_hash": sparse_hash,
    "sparse_bytemap": sparse_bytemap,
}

# CostFormatter stores no tensor with a dense or bytemap level, which
# allocate a slot for every index of their dimension, which it expects to take
# this many bytes or more.
MAX_MEMORY_UTILIZATION = 12 * 2**30
SLOTTED_KINDS = ("dense", "sparse_bytemap")

# CostFormatter stores no tensor of one level densely where it expects fewer
# than this fraction of its slots to have entries.
MIN_DENSE_DENSITY = 0.25

# The storage features of STORAGE_BASIS which count bytes: the pointers,
# indices and values of the levels, and the byte which flags each slot of a
# bytemap.
_BYTE_FEATURES = np.array(
    [
        float(name.endswith("_bytes") or name == "sparse_bytemap.slots")
        for name in STORAGE_BASIS.feature_names
    ]
)


def format_ftype(
    options: Sequence[LevelOption], fill_type: AbstractFillFType, shape_type
) -> FiberTensorFType:
    """The format of a tensor of the level options, fill_type, and shape_type."""
    lvl = element(fill_type, fill_type.element_type, ftypes.intp)
    for option, dim in reversed(list(zip(options, shape_type, strict=True))):
        lvl = LEVEL_BUILDERS[option.kind](lvl, dim)
    return fiber_tensor(lvl)


def expected_sizes(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    candidates: Mapping[lgc.Alias, Sequence[tuple[LevelOption, ...]]],
) -> dict[tuple[lgc.Alias, tuple[LevelOption, ...]], float]:
    """
    The bytes each alias of candidates is expected to take with the levels of
    each of its candidates, as STORAGE_BASIS prices the levels that the
    queries of prgm write, formatted as bindings says, in the largest of the
    queries which write it.
    """
    machine = StatsMachine(stats_factory, OrderedDict(stats))
    by_query: dict = defaultdict(float)
    for access in accesses(prgm, bindings, machine):
        alias = access.alias.unfused
        if not access.write or alias not in candidates:
            continue
        for kinds in candidates[alias]:
            choice = replace(access, options=[*kinds, ElementLevelOption()])
            features = STORAGE_BASIS.get_component_features(choice, machine)
            by_query[alias, kinds, access.prefix.expr] += float(
                features.sum(axis=0) @ _BYTE_FEATURES
            )
    sizes: dict[tuple[lgc.Alias, tuple[LevelOption, ...]], float] = {}
    for (alias, kinds, _), size in by_query.items():
        sizes[alias, kinds] = max(sizes.get((alias, kinds), 0.0), size)
    return sizes


def expected_densities(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    aliases: Sequence[lgc.Alias],
) -> dict[lgc.Alias, float]:
    """
    The fraction of the slots of the first level of each of aliases which
    STORAGE_BASIS expects the queries of prgm that write it, formatted as
    bindings says, to give entries, in the densest of them.
    """
    names = STORAGE_BASIS.feature_names
    entries, slots = names.index("sparse_list.entries"), names.index("dense.slots")
    sparse = [SparseListLevelOption(), ElementLevelOption()]
    dense = [DenseLevelOption(), ElementLevelOption()]
    machine = StatsMachine(stats_factory, OrderedDict(stats))
    densities: dict[lgc.Alias, float] = {}
    for access in accesses(prgm, bindings, machine):
        alias = access.alias.unfused
        if not access.write or access.depth != 0 or alias not in aliases:
            continue
        found = STORAGE_BASIS.get_component_features(
            replace(access, options=sparse), machine
        ).sum(axis=0)[entries]
        room = STORAGE_BASIS.get_component_features(
            replace(access, options=dense), machine
        ).sum(axis=0)[slots]
        if room > 0:
            densities[alias] = max(densities.get(alias, 0.0), found / room)
    return densities


def optimize_formats(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    stored: Mapping[lgc.Alias, tuple[FType, ...]],
    fill_types: Mapping[lgc.Alias, AbstractFillFType],
    cost_model: CostModel[ProtocolChoice] = PROTOCOL_COST_MODEL,
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
) -> tuple[dict[lgc.Alias, TensorFType], float]:
    """
    The format of each alias of stored, whose levels have the shape types it
    gives and the fill of fill_types, under which prgm costs least by
    cost_model, with its other tensors formatted as bindings says, and the cost
    of prgm with those formats, once choose_protocols chooses its protocols.
    The formats are found by coordinate descent: each alias in turn prices every
    product of the options its levels allow, as alias_costs prices them, for
    two sweeps or until none changes. Each starts from the first allowed option
    of each level which filters. A product with a level of SLOTTED_KINDS is
    refused where expected_sizes expects the alias to take
    MAX_MEMORY_UTILIZATION bytes or more with it, and a dense level is refused
    to an alias of one level which expected_densities expects to be less dense
    than MIN_DENSE_DENSITY.
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

    initial = {
        alias: tuple(
            next((option for option in level if option.filters), level[0])
            for level in levels
        )
        for alias, levels in allowed.items()
    }
    bindings = {
        **bindings,
        **{alias: ftype_of(alias, kinds) for alias, kinds in initial.items()},
    }
    slotted = {
        alias: [
            kinds
            for kinds in product(*levels)
            if any(option.kind in SLOTTED_KINDS for option in kinds)
        ]
        for alias, levels in allowed.items()
    }
    sizes = expected_sizes(prgm, bindings, stats, stats_factory, slotted)
    refused = {key for key, size in sizes.items() if size >= MAX_MEMORY_UTILIZATION}
    vectors = [alias for alias, levels in allowed.items() if len(levels) == 1]
    densities = expected_densities(prgm, bindings, stats, stats_factory, vectors)
    refused |= {
        (alias, kinds)
        for alias in vectors
        if densities.get(alias, 1.0) < MIN_DENSE_DENSITY
        for kinds in product(*allowed[alias])
        if kinds[0].kind == "dense"
    }
    # A refused start gives way to the first product which isn't refused.
    for alias, levels in allowed.items():
        if (alias, initial[alias]) in refused:
            kinds = next(
                (kinds for kinds in product(*levels) if (alias, kinds) not in refused),
                initial[alias],
            )
            bindings[alias] = ftype_of(alias, kinds)
    # Every format prices the same loops over the same stats.
    cache: dict = {}
    _, costs = choose_protocols(
        prgm, bindings, stats, stats_factory, cost_model, cache=cache
    )
    # A second sweep lets each alias answer the formats chosen after it.
    for _ in range(2):
        changed = False
        for alias, levels in allowed.items():
            best_fmt, best_costs = bindings[alias], costs
            candidates = {
                kinds: fmt
                for kinds in product(*levels)
                if (alias, kinds) not in refused
                and (fmt := ftype_of(alias, kinds)) != bindings[alias]
            }
            priced = alias_costs(
                prgm,
                bindings,
                stats,
                stats_factory,
                cost_model,
                alias,
                candidates,
                costs,
                cache,
            )
            for kinds, fmt in candidates.items():
                costs_2 = priced[kinds]
                if sum(costs_2) < sum(best_costs):
                    best_fmt, best_costs = fmt, costs_2
            if best_fmt != bindings[alias]:
                bindings[alias], costs = best_fmt, best_costs
                changed = True
        if not changed:
            break
    return {alias: bindings[alias] for alias in stored}, sum(costs)


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
                        if stores_nothing(lhs, shape_types[alias]):
                            bindings[alias] = scalar_ftype(fill_types[alias])
                        else:
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
        cost_model: CostModel[ProtocolChoice] = PROTOCOL_COST_MODEL,
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
        formats, _ = optimize_formats(
            prgm,
            bindings,
            stats,
            stats_factory,
            stored,
            fill_types,
            self.cost_model,
            self.options,
        )
        return formats

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
                        if stores_nothing(lhs, shape_types[alias]):
                            bindings[alias] = scalar_ftype(fill_types[alias])
                        else:
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
