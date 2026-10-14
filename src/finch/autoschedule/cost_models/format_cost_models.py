from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from finch.algebra import ffuncs, ftype, np_dtype
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
from finch.tensor import element

from .cost_models import NS, CostModel
from .loop_cost_models import PigeonCostModel, Prefix


class LevelOption(ABC):
    """
    A kind of level the formatter may choose for a dimension, and the number
    of positions it passes to its child.
    """

    kind: str

    @abstractmethod
    def next_num_pos(self, num_pos, n_l, nnz_l): ...


class DenseLevelOption(LevelOption):
    kind = "dense"

    def next_num_pos(self, num_pos, n_l, nnz_l):
        return num_pos * n_l


class SparseHashLevelOption(LevelOption):
    kind = "sparse_hash"

    def next_num_pos(self, num_pos, n_l, nnz_l):
        return nnz_l


@dataclass(frozen=True)
class Level:
    """
    The level at a zero-based depth of alias, with num_pos positions entering
    it. kind names a level option, or is "element" for the element level at the
    tensor's rank. term is the program which reads and writes alias.
    """

    term: LogicStatement
    alias: Alias
    depth: int
    num_pos: float
    kind: str


def nnz_after(stats: NS, stats_factory: StatsFactory[NS], depth: int):
    """Return the non-fill entries of the first depth + 1 levels of stats."""
    reduce_fields = tuple(stats.index_order[depth + 1 :])
    if reduce_fields:
        stats = stats_factory.aggregate(ffuncs.or_, False, reduce_fields, stats)
    return stats.estimate_non_fill_values()


class FormatCostModel(CostModel[Level]):
    """
    A cost model where each level of each formatted alias is a component.
    bindings names the level options of an alias, outermost first, and the
    alias is formatted when it is first written.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Sequence[LevelOption]],
        machine: StatsMachine[NS],
    ) -> Iterator[Level]:
        visited: set[Alias] = set()

        def visit_alias(alias: Alias, options: Sequence[LevelOption]):
            stats = machine(alias)
            num_pos = 1.0
            for depth, option in enumerate(options):
                yield Level(term, alias, depth, num_pos, option.kind)
                n_l = stats.get_dim_size(stats.index_order[depth])
                nnz_l = nnz_after(stats, machine.stats_factory, depth)
                num_pos = option.next_num_pos(num_pos, n_l, nnz_l)
            yield Level(term, alias, len(options), num_pos, "element")

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
    Each level contributes an allocation row, and an access row for each query
    loop prefix that uses it. The features are positions, slots, entries,
    position bytes, index bytes, and value bytes for the allocation, followed by
    sequential reads, scatter reads, sequential writes, and scatter writes for
    the accesses.
    """

    @property
    def n_features(self) -> int:
        return 10

    def get_level_features(self, num_pos, n_l, nnz_l, val_size, pos_size):
        """
        Return the features of a level with num_pos positions entering it, a
        dimension of n_l, and nnz_l non-fill entries: positions, slots, entries,
        position bytes, index bytes, and value bytes.
        """
        return np.array(
            [
                num_pos,
                num_pos * n_l,
                nnz_l,
                (num_pos + 1) * pos_size,
                nnz_l * pos_size,
                num_pos * val_size,
            ]
        )

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
        component: Level,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the allocation row of the level, followed by its access rows.
        """
        alias, depth, num_pos = component.alias, component.depth, component.num_pos
        stats = machine(alias)
        fill_ftype = ftype(stats.fill_value)
        leaf = element(stats.fill_value, fill_ftype)
        assert leaf.position_type is not None
        val_size = np_dtype(fill_ftype).itemsize
        pos_size = np_dtype(leaf.position_type).itemsize

        access_features = self.get_access_features(
            alias, depth, component.term, machine
        )
        features = np.zeros((1 + len(access_features), self.n_features))
        if component.kind == "element":
            # The element level stores one value for each position.
            n_l, nnz_l = 1, num_pos
        else:
            n_l = stats.get_dim_size(stats.index_order[depth])
            nnz_l = nnz_after(stats, machine.stats_factory, depth)
        features[0, :6] = self.get_level_features(
            num_pos, n_l, nnz_l, val_size, pos_size
        )
        features[1:, 6:] = access_features
        return features
