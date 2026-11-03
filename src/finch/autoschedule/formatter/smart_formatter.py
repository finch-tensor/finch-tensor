from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
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
    LevelBasis,
    LevelChoice,
    LevelOption,
    LinearCostModel,
    SparseByteMapLevelOption,
    SparseHashLevelOption,
    SparseListLevelOption,
    allowed_options,
)
from finch.autoschedule.cost_models.format_cost_models import (
    alias_blocks,
    read_protocols,
    write_protocols,
)
from finch.autoschedule.tensor_stats import FDStats, StatsInterpreter
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import LogicLoader, MockLogicLoader, StatsFactory
from finch.finch_logic.tensor_stats import TensorStats
from finch.symbolic import Namespace
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
LEVEL_OPTIONS = (
    DenseLevelOption(),
    SparseListLevelOption(),
    SparseHashLevelOption(),
    SparseByteMapLevelOption(),
)
LEVEL_BUILDERS = {
    "dense": dense,
    "sparse_list": sparse_list,
    "sparse_hash": sparse_hash,
    "sparse_bytemap": sparse_bytemap,
}


def optimize_format(
    alias, shape_type, machine, fill_type, cost_model, term, options=LEVEL_OPTIONS
):
    blocks = alias_blocks(alias, term, machine)
    n = len(shape_type)
    allowed = [allowed_options(alias, depth, term, options) for depth in range(n)]
    for depth, level_options in enumerate(allowed):
        if not level_options:
            raise ValueError(
                f"No level option supports reading level {depth} of {alias} with "
                f"{read_protocols(alias, depth, term)} and writing it with "
                f"{write_protocols(alias, depth, term)}."
            )
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
        for option in allowed[level]:
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

    return rec(0, (1.0,) * len(blocks))


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


# The basis of the levels of each kind of option, which cost models of the
# formatter describe.
LEVEL_BASIS = ByKind(
    {
        option.kind: LevelBasis(type(option))
        for option in (*LEVEL_OPTIONS, ElementLevelOption())
    }
)


def linear_level_model(
    storage: Mapping[type[LevelOption], Sequence[float]],
    access: Mapping[type[LevelOption], Sequence[float]] | None = None,
) -> LinearCostModel[LevelChoice]:
    """
    A linear model of LEVEL_BASIS, from the coefficients of the storage features
    of each option type, and of its walks, lookups, sequential writes, and
    scatter writes. Accesses are free unless access gives their coefficients.
    """
    access = access or {}
    coefficients = [
        [*storage[option], *access.get(option, (0, 0, 0, 0))]
        for option in map(LevelOption.kinds.__getitem__, LEVEL_BASIS.by_kind)
    ]
    return LinearCostModel(LEVEL_BASIS, np.concatenate(coefficients))


# Each level option lists its storage features. The access features are
# sequential reads, scatter reads, sequential writes, and scatter writes.
ITER_COST_MODEL = linear_level_model(
    {
        DenseLevelOption: (0, 1, 0),
        SparseListLevelOption: (1, 1, 0, 0),
        SparseHashLevelOption: (1, 1, 0, 0),
        SparseByteMapLevelOption: (1, 0, 1, 0, 0),
        ElementLevelOption: (0, 0),
    }
)
STORAGE_COST_MODEL = linear_level_model(
    {
        DenseLevelOption: (0, 0, 0),
        SparseListLevelOption: (0, 0, 1, 1),
        SparseHashLevelOption: (0, 0, 1, 1),
        # Each slot is flagged with a byte.
        SparseByteMapLevelOption: (0, 1, 0, 1, 0),
        ElementLevelOption: (0, 1),
    }
)


# Dense and byte map levels pay to reset each slice, but take scatter writes
# as cheaply as sequential ones. Hash levels pay for every probe, and sorted
# lists take sequential writes cheaply but binary search to look up out of
# order.
WORKSPACE_COST_MODEL = linear_level_model(
    {
        DenseLevelOption: (0, 0, 1),
        SparseListLevelOption: (0, 1, 0, 0),
        SparseHashLevelOption: (0, 1, 0, 0),
        SparseByteMapLevelOption: (0, 0.125, 0, 0, 1),
        ElementLevelOption: (0, 0),
    },
    {
        DenseLevelOption: (1, 1, 1, 1),
        SparseListLevelOption: (1, 8, 1, 0),
        SparseHashLevelOption: (1, 4, 4, 4),
        SparseByteMapLevelOption: (1, 1, 2, 2),
    },
)


def workspace_depth(prgm: lgc.LogicStatement, alias: lgc.Alias) -> int | None:
    """
    Return the number of leading fields that the query defining alias writes in
    loop order, when it scatters into a later field. A workspace fused over
    those fields can then take the scattered writes. Only an alias which one
    Query defines, and nothing updates, has a workspace.
    """
    writes: list[tuple[lgc.LogicStatement, tuple[lgc.Field, ...]]] = []

    def visit(node: lgc.LogicStatement, loops: tuple[lgc.Field, ...]):
        match node:
            case lgc.Plan(bodies):
                for body in bodies:
                    visit(body, loops)
            case lgc.Fuse(idx, body):
                visit(body, (*loops, idx))
            case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                lgc.Table(lgc.Alias() as lhs, _), _, _
            ) if lhs.unfused == alias:
                writes.append((node, loops))

    visit(prgm, ())
    match writes:
        case [(lgc.Query(lgc.Table(lgc.HardAlias(), idxs), _), loops)]:
            depth = 0
            while loops[depth : depth + 1] == idxs[depth : depth + 1] != ():
                depth += 1
            if depth < len(idxs):
                return depth
    return None


def add_workspace(
    prgm: lgc.LogicStatement, alias: lgc.HardAlias, depth: int, ws: lgc.HardAlias
) -> lgc.LogicStatement:
    """
    Redirect the query which defines alias into ws, fused over its first depth
    fields. Right after the statement within depth Fuse nodes which holds the
    query, ws is copied into alias in order.
    """
    fused = lgc.FusedAlias(ws, depth)

    # Each rewrite returns the copy until the depth where it is inserted.
    def rewrite(
        node: lgc.LogicStatement, nloops: int
    ) -> tuple[lgc.LogicStatement, lgc.LogicStatement | None]:
        match node:
            case lgc.Plan(bodies):
                stmts: list[lgc.LogicStatement] = []
                pending = None
                for body in bodies:
                    body, copy = rewrite(body, nloops)
                    stmts.append(body)
                    if copy is not None and nloops == depth:
                        stmts.append(copy)
                    elif copy is not None:
                        pending = copy
                return lgc.Plan(tuple(stmts)), pending
            case lgc.Fuse(idx, body):
                body, copy = rewrite(body, nloops + 1)
                if copy is not None and nloops + 1 == depth:
                    return lgc.Fuse(idx, lgc.Plan((body, copy))), None
                return lgc.Fuse(idx, body), copy
            case lgc.Query(lgc.Table(lhs, idxs), rhs) if lhs == alias:
                copy = fuse_loop_order(
                    lgc.Query(lgc.Table(alias, idxs), lgc.Table(fused, idxs)),
                    idxs[depth:],
                )
                return lgc.Query(lgc.Table(fused, idxs), rhs), copy
            case _:
                return node, None

    prgm, copy = rewrite(prgm, 0)
    if copy is not None:
        prgm = lgc.Plan((prgm, copy))
    return prgm


class CostFormatter(LogicFormatter):
    """
    CostFormatter chooses the format of each alias from the level options with
    the lowest cost under cost_model, memoizing the best format for each depth and
    input position count. The cost of a level is summed over the blocks of its
    alias in the program. With workspaces, a query which scatters into a new
    alias may instead write a workspace fused over the fields it writes in
    order, which is then copied into the alias in order, when that costs less.
    """

    def __init__(
        self,
        loader: LogicLoader | None = None,
        cost_model: CostModel[LevelChoice] = STORAGE_COST_MODEL,
        options: Sequence[LevelOption] = LEVEL_OPTIONS,
        workspaces: bool = False,
    ):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader
        self.cost_model = cost_model
        self.options = options
        self.workspaces = workspaces

    def format_cost(
        self,
        alias: lgc.Alias,
        term: lgc.LogicStatement,
        machine: StatsMachine,
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
    ) -> float:
        cost, _ = optimize_format(
            alias, shape_type, machine, fill_type, self.cost_model, term, self.options
        )
        return cost

    def get_tensor_ftype(
        self,
        alias: lgc.Alias,
        term: lgc.LogicStatement,
        machine: StatsMachine,
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
    ) -> FiberTensorFType:
        _, lvl = optimize_format(
            alias, shape_type, machine, fill_type, self.cost_model, term, self.options
        )
        return fiber_tensor(lvl)

    def stored_shape_types(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        machine: StatsMachine,
    ) -> dict[lgc.Alias, tuple[FType, ...]]:
        """
        Return the dimensions each new alias of prgm stores, at its first
        definition, advancing machine past prgm.
        """
        shape_types: dict[lgc.Alias, tuple[FType | None, ...]] = {
            var: val.shape_type for var, val in bindings.items()
        }
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
        return stored_shape_types

    def add_workspaces(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TS],
        stats_factory: StatsFactory,
    ) -> lgc.LogicStatement:
        """
        Give each new alias that its query scatters into a workspace, when
        formatting the workspace and the alias it is copied into costs less
        than formatting the alias alone.
        """
        initial = StatsMachine(stats_factory, OrderedDict(stats))
        stored_shape_types = self.stored_shape_types(
            prgm, bindings, StatsMachine(stats_factory, OrderedDict(stats))
        )
        fill_types = prgm.infer_fill_type(
            {var: val.fill_type for var, val in bindings.items()}
        )
        spc = Namespace(prgm)
        for alias, shape_type in stored_shape_types.items():
            depth = workspace_depth(prgm, alias)
            if depth is None:
                continue
            assert isinstance(alias, lgc.HardAlias)
            ws = lgc.HardAlias(spc.freshen(alias.name, "ws"))
            staged = add_workspace(prgm, alias, depth, ws)
            fill_type = fill_types[alias]
            direct = self.format_cost(alias, prgm, initial, fill_type, shape_type)
            staged_cost = self.format_cost(
                alias, staged, initial, fill_type, shape_type
            ) + self.format_cost(ws, staged, initial, fill_type, shape_type[depth:])
            if staged_cost < direct:
                prgm = staged
        return prgm

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TS],
        stats_factory: StatsFactory,
    ):
        bindings = bindings.copy()
        if self.workspaces:
            prgm = self.add_workspaces(prgm, bindings, stats, stats_factory)
        machine = StatsMachine(stats_factory, OrderedDict(stats))
        stored_shape_types = self.stored_shape_types(prgm, bindings, machine)
        fill_types = prgm.infer_fill_type(
            {var: val.fill_type for var, val in bindings.items()}
        )
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


class WorkspaceCostFormatter(CostFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader, WORKSPACE_COST_MODEL, LEVEL_OPTIONS, workspaces=True)
