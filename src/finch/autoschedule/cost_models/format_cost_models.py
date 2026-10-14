from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from finch.algebra import ffuncs, ftypes, np_dtype
from finch.algebra.utils import is_subsequence
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicStatement,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    StatsFactory,
    Table,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS, CostModel
from .loop_cost_models import PigeonCostModel, Prefix


def nnz_after(stats: NS, stats_factory: StatsFactory[NS], depth: int):
    """Return the non-fill entries of the first depth + 1 levels of stats."""
    reduce_fields = tuple(stats.index_order[depth + 1 :])
    if reduce_fields:
        stats = stats_factory.aggregate(
            ffuncs.or_, False, reduce_fields, stats_factory.support(stats)
        )
    return stats.estimate_non_fill_values()


class LevelOption(ABC):
    """
    A kind of level, which describes the storage of a level with n_features
    storage features, and the number of positions it passes to its child. Both
    depend on the stats of the tensor, the zero-based depth of the level, and
    the num_pos positions entering it. Storage features also depend on the
    bytes of an index of the level, and of a position and a value of the tensor.
    """

    kind: ClassVar[str]
    n_features: ClassVar[int]

    @abstractmethod
    def next_num_pos(
        self, stats: NS, stats_factory: StatsFactory[NS], depth: int, num_pos
    ): ...

    @abstractmethod
    def get_storage_features(
        self,
        stats: NS,
        stats_factory: StatsFactory[NS],
        depth: int,
        num_pos,
        idx_size: int,
        pos_size: int,
        val_size: int,
    ) -> np.ndarray: ...


class DenseLevelOption(LevelOption):
    """Storage features are positions and slots."""

    kind = "dense"
    n_features = 2

    def next_num_pos(self, stats, stats_factory, depth, num_pos):
        return num_pos * stats.get_dim_size(stats.index_order[depth])

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size
    ):
        return np.array(
            [num_pos, self.next_num_pos(stats, stats_factory, depth, num_pos)]
        )


class SparseHashLevelOption(LevelOption):
    """Storage features are positions, entries, position bytes, and index bytes."""

    kind = "sparse_hash"
    n_features = 4

    def next_num_pos(self, stats, stats_factory, depth, num_pos):
        return nnz_after(stats, stats_factory, depth)

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size
    ):
        nnz_l = nnz_after(stats, stats_factory, depth)
        return np.array([num_pos, nnz_l, (num_pos + 1) * pos_size, nnz_l * idx_size])


class ElementLevelOption(LevelOption):
    """
    The element level at the rank of a tensor stores one value for each
    position. Storage features are values and value bytes.
    """

    kind = "element"
    n_features = 2

    def next_num_pos(self, stats, stats_factory, depth, num_pos):
        # The element level has no child, so its positions pass through.
        return num_pos

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size
    ):
        return np.array([num_pos, num_pos * val_size])


@dataclass(frozen=True)
class LevelChoice:
    """
    The choice of a level option at a zero-based depth of alias, with num_pos
    positions entering it, in a format whose positions take pos_size bytes.
    term is the program which reads and writes alias.
    """

    term: LogicStatement
    alias: Alias
    depth: int
    num_pos: float
    option: LevelOption
    pos_size: int

    @property
    def kind(self) -> str:
        return self.option.kind


class FormatCostModel(CostModel[LevelChoice]):
    """
    A cost model where each level of each formatted alias is a component.
    bindings lists the level options of an alias, outermost first and ending
    with its element level, and the alias is formatted when it is first written.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Sequence[LevelOption]],
        machine: StatsMachine[NS],
    ) -> Iterator[LevelChoice]:
        visited: set[Alias] = set()

        def visit_alias(alias: Alias, options: Sequence[LevelOption]):
            stats = machine(alias)
            # bindings name no position type, so positions are intp.
            pos_size = np_dtype(ftypes.intp).itemsize
            num_pos = 1.0
            for depth, option in enumerate(options):
                yield LevelChoice(term, alias, depth, num_pos, option, pos_size)
                num_pos = option.next_num_pos(
                    stats, machine.stats_factory, depth, num_pos
                )

        def visit_stmt(stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        yield from visit_stmt(body)
                case Query(Table(Alias() as lhs, _), _) | QueryInto(
                    Table(Alias() as lhs, _), _, _
                ):
                    machine(stmt)
                    if lhs in bindings and lhs not in visited:
                        yield from visit_alias(lhs, bindings[lhs])
                    visited.add(lhs)
                case Produces():
                    return
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        return visit_stmt(term)


class LevelCostModel(FormatCostModel, LoopOrderedForm):
    """
    The cost model of the levels of one type of option. Each level contributes
    an allocation row, and an access row for each query loop prefix that uses
    it. The features are the storage features of the option for the
    allocation, followed by sequential reads, scatter reads, sequential writes,
    and scatter writes for the accesses.
    """

    def __init__(
        self,
        option_type: type[LevelOption],
        predictor: Callable[[np.ndarray], float],
    ):
        super().__init__(predictor)
        self.option_type = option_type

    @property
    def n_features(self) -> int:
        return self.option_type.n_features + 4

    def get_access_features(
        self,
        alias: Alias,
        level: int,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the accesses to alias at a zero-based depth in each query of term:
        sequential reads, scatter reads, sequential writes, and scatter writes,
        with one row per loop prefix that accesses the level in each query.
        Each access is counted by the PigeonCostModel iterations of the loops
        from the root to the current level. The element level is accessed with
        its parent.
        """
        # A sequential write is when the prefix of loops from root to current
        # level matches the ordering of the access exactly.
        # A sequential read is when the prefix of loops from root to current
        # level is a supersequence of the ordering of the access.
        features = []
        stats_interpreter = StatsMachine(
            machine.stats_factory, OrderedDict(machine.bindings)
        )
        pigeon = PigeonCostModel()

        def visit_access(idxs, loop_order, rhs, lhs_idxs, is_write, prefixes):
            if not idxs:
                return
            depth = min(level, len(idxs) - 1)
            order = tuple(idxs[: depth + 1])
            loops = tuple(loop_order[: loop_order.index(idxs[depth]) + 1])
            iterations = pigeon.get_component_features(
                Prefix(loops, rhs, lhs_idxs), stats_interpreter
            )[0, 0]
            features = prefixes.setdefault(loops, np.zeros(4))
            if is_write:
                features[2 if loops == order else 3] += iterations
            else:
                features[0 if is_subsequence(order, loops) else 1] += iterations

        def visit_stmt(stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                    return
                case QueryInto(Table(), _, Literal()):
                    stats_interpreter(stmt)
                    return
                case Produces():
                    return
                case Query(Table(lhs, lhs_idxs), Table() as rhs):
                    arg = rhs
                    loop_order = lhs_idxs
                case Query(
                    Table(lhs, lhs_idxs),
                    Aggregate(_, _, Reorder(arg, loop_order), _) as rhs,
                ):
                    pass
                case QueryInto(Table(lhs, lhs_idxs), _, Reorder(arg, loop_order)):
                    rhs = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            prefixes = {}
            for node in PostOrderDFS(arg):
                match node:
                    case Table(tns, idxs) if tns == alias:
                        visit_access(idxs, loop_order, rhs, lhs_idxs, False, prefixes)
            if lhs == alias:
                visit_access(lhs_idxs, loop_order, rhs, lhs_idxs, True, prefixes)
            features.extend(prefixes.values())
            stats_interpreter(stmt)

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, 4)

    def get_component_features(
        self,
        component: LevelChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the allocation row of the level, followed by its access rows.
        """
        stats = machine(component.alias)
        shape_type = stats.shape_type
        # The element level has no dimension, and a broadcast dimension is intp.
        idx_size = 0
        if component.depth < len(shape_type):
            idx_size = np_dtype(shape_type[component.depth] or ftypes.intp).itemsize
        access_features = self.get_access_features(
            component.alias, component.depth, component.term, machine
        )
        n_storage = self.option_type.n_features
        features = np.zeros((1 + len(access_features), self.n_features))
        features[0, :n_storage] = component.option.get_storage_features(
            stats,
            machine.stats_factory,
            component.depth,
            component.num_pos,
            idx_size,
            component.pos_size,
            np_dtype(stats.element_type).itemsize,
        )
        features[1:, n_storage:] = access_features
        return features
