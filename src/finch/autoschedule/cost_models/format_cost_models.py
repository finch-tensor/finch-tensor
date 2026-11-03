from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from finch.algebra import ffuncs, ftypes, np_dtype
from finch.autoschedule import util
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Alias,
    DefaultProtocol,
    Extrude,
    Field,
    Follow,
    Fuse,
    Gallop,
    Laminate,
    Literal,
    LogicStatement,
    Plan,
    Produces,
    Protocol,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    Walk,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS, Basis


def nnz_after(stats: NS, stats_factory: StatsFactory[NS], depth: int, over: tuple = ()):
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


def alias_blocks(
    alias: Alias, term: LogicStatement, machine: StatsMachine[NS]
) -> list[tuple[NS, tuple[Field, ...]]]:
    """
    Return the stats and fused fields of each block of alias in term, in program
    order. A block is a query which defines alias and the in-place updates which
    follow it, and its stats are the union of the supports of its results, with
    the fill of alias. An alias which term reads but never defines, such as an
    input, is one block of its binding. Every block stores the same rank. term
    is evaluated from the bindings of machine.
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
            case QueryInto(Table(Alias() as lhs, _), _, _) if (
                lhs.unfused == alias.unfused and blocks
            ):
                stats_interpreter(stmt)
                block, over = blocks[-1]
                stats = stats_factory.relabel(stats_interpreter(lhs), block.index_order)
                union = stats_factory.mapjoin(
                    ffuncs.or_,
                    stats_factory.support(block),
                    stats_factory.support(stats),
                )
                union.fill_type = block.fill_type
                blocks[-1] = (union, over)
            case Query(Table(Alias() as lhs, idxs), _) | QueryInto(
                Table(Alias() as lhs, idxs), _, _
            ):
                stats_interpreter(stmt)
                if lhs.unfused == alias.unfused:
                    # A fused alias stores one slice for each value of its
                    # leading fields.
                    blocks.append((stats_interpreter(lhs), idxs[: lhs.nfused]))
            case Produces():
                return
            case _:
                raise ValueError(f"Unsupported statement type: {stmt}")

    visit_stmt(term)
    if not blocks and alias in machine.bindings:
        blocks.append((machine.bindings[alias], ()))
    if len({len(stats.index_order) - len(over) for stats, over in blocks}) > 1:
        raise ValueError(f"Cannot change the rank of allocated tensor {alias}.")
    return blocks


class LevelOption(ABC):
    """
    A kind of level, which describes the storage of a level with the storage
    features feature_names names, and the number of positions it passes to its
    child. Both depend on the stats of the tensor, the zero-based depth of the
    level, and the num_pos positions entering it. Storage features also depend
    on the bytes of an index of the level, and of a position and a value of the
    tensor. A tensor which stores one slice that fixes the fields of over is
    described by its largest slice. A level of the kind can only be read and
    written with its protocols.
    """

    kind: ClassVar[str]
    feature_names: ClassVar[tuple[str, ...]]
    protocols: ClassVar[tuple[Protocol, ...]]
    kinds: ClassVar[dict[str, type["LevelOption"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "kind" in cls.__dict__:
            LevelOption.kinds[cls.kind] = cls

    @abstractmethod
    def next_num_pos(
        self, stats: NS, stats_factory: StatsFactory[NS], depth: int, num_pos, over
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
        over: tuple[Field, ...],
    ) -> np.ndarray: ...


class DenseLevelOption(LevelOption):
    """Storage features are positions and slots."""

    kind = "dense"
    feature_names = ("positions", "slots")
    # A dense level is only looked up, so that it never decides where a loop
    # goes.
    protocols = (Follow(), Extrude(), Laminate())

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over):
        return num_pos * stats.get_dim_size(stats.index_order[depth])

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size, over
    ):
        return np.array(
            [num_pos, self.next_num_pos(stats, stats_factory, depth, num_pos, over)]
        )


class SparseHashLevelOption(LevelOption):
    """Storage features are positions, entries, position bytes, and index bytes."""

    kind = "sparse_hash"
    feature_names = ("positions", "entries", "position_bytes", "index_bytes")
    protocols = (DefaultProtocol(), Walk(), Follow(), Extrude(), Laminate())

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over):
        return nnz_after(stats, stats_factory, depth, over)

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size, over
    ):
        nnz_l = nnz_after(stats, stats_factory, depth, over)
        return np.array([num_pos, nnz_l, (num_pos + 1) * pos_size, nnz_l * idx_size])


class SparseListLevelOption(LevelOption):
    """Storage features are positions, entries, position bytes, and index bytes."""

    kind = "sparse_list"
    feature_names = ("positions", "entries", "position_bytes", "index_bytes")
    # A list appends each entry after the last, so it's only extruded.
    protocols = (DefaultProtocol(), Walk(), Gallop(), Follow(), Extrude())

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over):
        return nnz_after(stats, stats_factory, depth, over)

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size, over
    ):
        nnz_l = nnz_after(stats, stats_factory, depth, over)
        return np.array([num_pos, nnz_l, (num_pos + 1) * pos_size, nnz_l * idx_size])


class SparseByteMapLevelOption(LevelOption):
    """
    A bytemap flags each of its slots with a byte, and lists the slots which
    are occupied, so its children are allocated densely. Storage features are
    positions, slots, entries, and the bytes of its pointers and list.
    """

    kind = "sparse_bytemap"
    feature_names = ("positions", "slots", "entries", "position_bytes")
    protocols = (
        DefaultProtocol(),
        Walk(),
        Gallop(),
        Follow(),
        Extrude(),
        Laminate(),
    )

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over):
        return num_pos * stats.get_dim_size(stats.index_order[depth])

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size, over
    ):
        slots = self.next_num_pos(stats, stats_factory, depth, num_pos, over)
        nnz_l = nnz_after(stats, stats_factory, depth, over)
        return np.array([num_pos, slots, nnz_l, (num_pos + 1 + nnz_l) * pos_size])


class ElementLevelOption(LevelOption):
    """
    The element level at the rank of a tensor stores one value for each
    position. Storage features are values and value bytes.
    """

    kind = "element"
    feature_names = ("values", "value_bytes")
    protocols = (
        DefaultProtocol(),
        Walk(),
        Gallop(),
        Follow(),
        Extrude(),
        Laminate(),
    )

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over):
        # The element level has no child, so its positions pass through.
        return num_pos

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size, over
    ):
        return np.array([num_pos, num_pos * val_size])


def read_protocols(alias: Alias, depth: int, term: LogicStatement) -> set[Protocol]:
    """The protocols which term reads the level of alias at a depth with."""
    protocols = set()
    for node in PostOrderDFS(term):
        match node:
            case Query(_, rhs) | QueryInto(_, _, rhs):
                for table in PostOrderDFS(rhs):
                    match table:
                        case Table(Alias() as tns, idxs, protos) if (
                            tns.unfused == alias.unfused
                            and tns.nfused + depth < len(idxs)
                        ):
                            protocols.add(
                                protos[tns.nfused + depth]
                                if protos
                                else DefaultProtocol()
                            )
    return protocols


def write_protocol(
    lhs: Alias, idxs: tuple[Field, ...], depth: int, loop_order: tuple[Field, ...]
) -> Protocol:
    """
    The protocol which a query under loop_order writes the level of a table of
    lhs and idxs at a zero-based stored depth with, as util.write_protocols
    infers. The element level is written with its parent, and an alias which
    stores only its element is written once for each of its fused fields.
    """
    protocols = util.write_protocols(idxs, loop_order, lhs.nfused)
    return protocols[min(depth, len(protocols) - 1)] if protocols else Extrude()


def write_protocols(alias: Alias, depth: int, term: LogicStatement) -> set[Protocol]:
    """
    The protocols which term writes the level of alias at a depth with. The
    loops are the fields of the enclosing Fuse nodes. An update which reduces
    into alias laminates what it already stores, but one which overwrites it or
    sets it to a literal starts it again.
    """
    protocols = set()

    def visit(stmt: LogicStatement, loop_order: tuple[Field, ...]):
        match stmt:
            case Plan(bodies):
                for body in bodies:
                    visit(body, loop_order)
            case Fuse(idx, body):
                visit(body, (*loop_order, idx))
            case QueryInto(Table(), _, Literal()):
                return
            case QueryInto(Table(Alias() as lhs, _), Literal(op), _) if (
                lhs.unfused == alias.unfused and op != ffuncs.overwrite
            ):
                protocols.add(Laminate())
            case Query(Table(Alias() as lhs, idxs), _) | QueryInto(
                Table(Alias() as lhs, idxs), _, _
            ) if lhs.unfused == alias.unfused and idxs:
                protocols.add(write_protocol(lhs, idxs, depth, loop_order))

    visit(term, ())
    return protocols


def allowed_options(
    alias: Alias, depth: int, term: LogicStatement, options: Sequence[LevelOption]
) -> list[LevelOption]:
    """
    The options which can take the level of alias at a depth in term: those
    which support the protocols term reads and writes it with.
    """
    protocols = read_protocols(alias, depth, term) | write_protocols(alias, depth, term)
    return [option for option in options if protocols <= set(option.protocols)]


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


class FormatBasis(Basis[LevelChoice]):
    """
    A basis where each level of each formatted alias is a component.
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
        # bindings name no position type, so positions are intp.
        pos_size = np_dtype(ftypes.intp).itemsize
        for alias, options in bindings.items():
            blocks = alias_blocks(alias, term, machine)
            if not blocks:
                continue
            num_pos = (1.0,) * len(blocks)
            for depth, option in enumerate(options):
                yield LevelChoice(term, alias, depth, num_pos, option, pos_size)
                num_pos = tuple(
                    option.next_num_pos(
                        stats, machine.stats_factory, len(over) + depth, n, over
                    )
                    for (stats, over), n in zip(blocks, num_pos, strict=True)
                )
        # The levels are priced over the whole of term from the stats before
        # it, so the machine is advanced past term only after every level.
        machine(term)
