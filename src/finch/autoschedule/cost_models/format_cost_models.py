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
    Field,
    Fuse,
    FusedAlias,
    Literal,
    LogicStatement,
    Plan,
    Produces,
    Query,
    QueryInto,
    StatsFactory,
    Table,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS, CostModel
from .loop_cost_models import PigeonCostModel, Prefix


def nnz_after(
    stats: NS,
    stats_factory: StatsFactory[NS],
    depth: int,
    over: tuple[Field, ...] = (),
):
    """
    Return the non-fill entries of the first depth + 1 levels of stats, in the
    largest slice which fixes the fields of over.
    """
    reduce_fields = tuple(stats.index_order[depth + 1 :])
    if reduce_fields:
        stats = stats_factory.aggregate(
            ffuncs.or_, False, reduce_fields, stats_factory.support(stats)
        )
    return stats.estimate_non_fill_values(over=over)


def fused_fields(alias: Alias, idxs: tuple[Field, ...]) -> tuple[Field, ...]:
    """
    The fields of a table of alias with idxs which a fused alias fixes in each
    slice it stores, which are its leading fields.
    """
    match alias:
        case FusedAlias(_, n):
            return idxs[:n]
        case _:
            return ()


def alias_blocks(
    alias: Alias, term: LogicStatement, machine: StatsMachine[NS]
) -> list[tuple[NS, tuple[Field, ...]]]:
    """
    Return the stats and fused fields of each block of alias in term, in program
    order. A block is a query which defines alias and the in-place updates which
    follow it, and its stats are the union of the supports of its results, with
    the fill of alias. Every block stores the same rank. term is evaluated from
    the bindings of machine.
    """
    stats_factory = machine.stats_factory
    stats_interpreter = StatsMachine(stats_factory, OrderedDict(machine.bindings))
    blocks: list[tuple[NS, tuple[Field, ...]]] = []

    def visit_stmt(stmt: LogicStatement):
        match stmt:
            case Plan(bodies):
                for body in bodies:
                    visit_stmt(body)
            case Fuse(_, body):
                visit_stmt(body)
            case Query(Table(Alias() as lhs, idxs), _) | QueryInto(
                Table(Alias() as lhs, idxs), _, _
            ):
                stats_interpreter(stmt)
                if lhs.unfused != alias.unfused:
                    return
                stats = stats_interpreter(lhs)
                if isinstance(stmt, Query) or not blocks:
                    blocks.append((stats, fused_fields(lhs, idxs)))
                    return
                block, over = blocks[-1]
                union = stats_factory.mapjoin(
                    ffuncs.or_,
                    stats_factory.support(block),
                    stats_factory.support(
                        stats_factory.relabel(stats, block.index_order)
                    ),
                )
                union.fill_type = block.fill_type
                blocks[-1] = (union, over)
            case Produces():
                return
            case _:
                raise ValueError(f"Unsupported statement type: {stmt}")

    visit_stmt(term)
    if len({len(stats.index_order) - len(over) for stats, over in blocks}) > 1:
        raise ValueError(f"Cannot change the rank of allocated tensor {alias}.")
    return blocks


class LevelOption(ABC):
    """
    A kind of level, which describes the storage of a level with n_features
    storage features, and the number of positions it passes to its child. Both
    depend on the stats of the tensor, the zero-based depth of the level, and
    the num_pos positions entering it. Storage features also depend on the
    bytes of an index of the level, and of a position and a value of the tensor.
    A tensor which stores one slice that fixes the fields of over is described
    by its largest slice.
    """

    kind: ClassVar[str]
    n_features: ClassVar[int]

    @abstractmethod
    def next_num_pos(
        self,
        stats: NS,
        stats_factory: StatsFactory[NS],
        depth: int,
        num_pos,
        over: tuple[Field, ...] = (),
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
        over: tuple[Field, ...] = (),
    ) -> np.ndarray: ...


class DenseLevelOption(LevelOption):
    """Storage features are positions and slots."""

    kind = "dense"
    n_features = 2

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over=()):
        return num_pos * stats.get_dim_size(stats.index_order[depth])

    def get_storage_features(
        self,
        stats,
        stats_factory,
        depth,
        num_pos,
        idx_size,
        pos_size,
        val_size,
        over=(),
    ):
        return np.array(
            [num_pos, self.next_num_pos(stats, stats_factory, depth, num_pos, over)]
        )


class SparseHashLevelOption(LevelOption):
    """Storage features are positions, entries, position bytes, and index bytes."""

    kind = "sparse_hash"
    n_features = 4

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over=()):
        return nnz_after(stats, stats_factory, depth, over)

    def get_storage_features(
        self,
        stats,
        stats_factory,
        depth,
        num_pos,
        idx_size,
        pos_size,
        val_size,
        over=(),
    ):
        nnz_l = nnz_after(stats, stats_factory, depth, over)
        return np.array([num_pos, nnz_l, (num_pos + 1) * pos_size, nnz_l * idx_size])


class ElementLevelOption(LevelOption):
    """
    The element level at the rank of a tensor stores one value for each
    position. Storage features are values and value bytes.
    """

    kind = "element"
    n_features = 2

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over=()):
        # The element level has no child, so its positions pass through.
        return num_pos

    def get_storage_features(
        self,
        stats,
        stats_factory,
        depth,
        num_pos,
        idx_size,
        pos_size,
        val_size,
        over=(),
    ):
        return np.array([num_pos, num_pos * val_size])


def next_block_num_pos(
    option: LevelOption,
    blocks: list[tuple[NS, tuple[Field, ...]]],
    stats_factory: StatsFactory[NS],
    depth: int,
    num_pos: tuple[float, ...],
) -> tuple[float, ...]:
    """The positions the level at a stored depth passes to its child in each block."""
    return tuple(
        option.next_num_pos(stats, stats_factory, len(over) + depth, n, over)
        for (stats, over), n in zip(blocks, num_pos, strict=True)
    )


@dataclass(frozen=True)
class LevelChoice:
    """
    The choice of a level option at a zero-based depth of the levels alias
    stores, in a format whose positions take pos_size bytes. term is the
    program which reads and writes alias, and num_pos lists the positions
    entering the level in each block of alias in term. A fused view of alias
    stores one slice for each value of its fused fields, so its blocks are
    described by their largest slice.
    """

    term: LogicStatement
    alias: Alias
    depth: int
    num_pos: tuple[float, ...]
    option: LevelOption
    pos_size: int

    @property
    def kind(self) -> str:
        return self.option.kind


class FormatCostModel(CostModel[LevelChoice]):
    """
    A cost model where each level of each formatted alias is a component.
    bindings lists the options of the levels an alias stores, outermost first
    and ending with its element level. Each level is priced over every block of
    the alias in the program.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Sequence[LevelOption]],
        machine: StatsMachine[NS],
    ) -> Iterator[LevelChoice]:
        machine(term)
        # bindings name no position type, so positions are intp.
        pos_size = np_dtype(ftypes.intp).itemsize
        for alias, options in bindings.items():
            blocks = alias_blocks(alias, term, machine)
            if not blocks:
                continue
            num_pos = (1.0,) * len(blocks)
            for depth, option in enumerate(options):
                yield LevelChoice(term, alias, depth, num_pos, option, pos_size)
                num_pos = next_block_num_pos(
                    option, blocks, machine.stats_factory, depth, num_pos
                )


class LevelCostModel(FormatCostModel, LoopOrderedForm):
    """
    The cost model of the levels of one type of option. Each level contributes
    an allocation row for each block of its alias, and an access row for each
    query loop prefix that uses it. The features are the storage features of
    the option for the allocation, followed by sequential reads, scatter reads,
    sequential writes, and scatter writes for the accesses.
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
        Return the accesses to alias at a zero-based stored depth in each query
        of term: sequential reads, scatter reads, sequential writes, and scatter
        writes, with one row per loop prefix that accesses the level in each
        query.
        Each access is counted by the PigeonCostModel iterations of the loops
        from the root to the current level, which are the fields of the
        enclosing Fuse nodes. Every view of alias accesses its levels. The
        element level is accessed with its parent.
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

        def visit_access(tns, idxs, loop_order, rhs, lhs_idxs, is_write, prefixes):
            if not idxs:
                return
            depth = min(len(fused_fields(tns, idxs)) + level, len(idxs) - 1)
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

        def visit_stmt(stmt: LogicStatement, loop_order: tuple[Field, ...]):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body, loop_order)
                    return
                case Fuse(idx, body):
                    visit_stmt(body, (*loop_order, idx))
                    return
                case QueryInto(Table(), _, Literal()):
                    stats_interpreter(stmt)
                    return
                case Produces():
                    return
                case Query(Table(Alias() as lhs, lhs_idxs), Table() as rhs):
                    arg = rhs
                case Query(
                    Table(Alias() as lhs, lhs_idxs), Aggregate(_, _, arg, _) as rhs
                ):
                    pass
                case QueryInto(Table(Alias() as lhs, lhs_idxs), _, arg):
                    rhs = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            prefixes = {}
            for node in PostOrderDFS(arg):
                match node:
                    case Table(Alias() as tns, idxs) if tns.unfused == alias.unfused:
                        visit_access(
                            tns, idxs, loop_order, rhs, lhs_idxs, False, prefixes
                        )
            if lhs.unfused == alias.unfused:
                visit_access(lhs, lhs_idxs, loop_order, rhs, lhs_idxs, True, prefixes)
            features.extend(prefixes.values())
            stats_interpreter(stmt)

        visit_stmt(term, ())
        return np.asarray(features, dtype=float).reshape(-1, 4)

    def get_component_features(
        self,
        component: LevelChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the allocation row of the level in each block of its alias,
        followed by its access rows in the whole program.
        """
        blocks = alias_blocks(component.alias, component.term, machine)
        access_features = self.get_access_features(
            component.alias, component.depth, component.term, machine
        )
        n_storage = self.option_type.n_features
        features = np.zeros((len(blocks) + len(access_features), self.n_features))
        for row, ((stats, over), num_pos) in enumerate(
            zip(blocks, component.num_pos, strict=True)
        ):
            depth = len(over) + component.depth
            shape_type = stats.shape_type
            # The element level has no dimension, and a broadcast dimension is
            # intp.
            idx_size = 0
            if depth < len(shape_type):
                idx_size = np_dtype(shape_type[depth] or ftypes.intp).itemsize
            features[row, :n_storage] = component.option.get_storage_features(
                stats,
                machine.stats_factory,
                depth,
                num_pos,
                idx_size,
                component.pos_size,
                np_dtype(stats.element_type).itemsize,
                over,
            )
        features[len(blocks) :, n_storage:] = access_features
        return features
