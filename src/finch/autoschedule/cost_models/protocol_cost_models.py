"""
The protocol cost model prices the loops of a query and the accesses of the
levels of its formatted tensors at them, so that the protocol optimizer can
choose the protocol of each read. Where the level cost model of
format_cost_models prices the levels a tensor stores, this prices each entry
of a loop, each read of a level with its protocol, and each write of a level
of the output, each kind of level differently.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from finch.algebra import ffuncs
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Extrude,
    Field,
    Follow,
    Gallop,
    Laminate,
    Literal,
    LogicStatement,
    Protocol,
    Table,
    Walk,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS, Basis, LinearCostModel
from .format_cost_models import (
    DenseLevelOption,
    LevelOption,
    SparseByteMapLevelOption,
    SparseHashLevelOption,
    SparseListLevelOption,
    followed,
    level_formats,
)
from .loop_cost_models import LoopSpaces, Prefix, prefixes

# The protocols which a table can read a level with, in the order ties between
# them are broken.
READ_PROTOCOLS: tuple[Protocol, ...] = (Walk(), Gallop(), Follow())


def read_options(
    option: LevelOption,
    idxs: tuple[Field, ...],
    nfused: int,
    depth: int,
    loop_order: tuple[Field, ...],
) -> list[Protocol]:
    """
    The protocols which a table of idxs, whose first nfused fields are fused,
    can read a level of option at a zero-based stored depth with under
    loop_order. A walk or a gallop steps through the level at the loop over its
    field, so that loop must come after the loops over the fields of the levels
    above it. Otherwise the level is reached inside the loop over its field,
    where it can only be followed.
    """
    pos = loop_order.index(idxs[nfused + depth])
    in_order = all(loop_order.index(idx) < pos for idx in idxs[: nfused + depth])
    return [
        proto
        for proto in READ_PROTOCOLS
        if proto in option.protocols and (in_order or proto == Follow())
    ]


@dataclass(frozen=True)
class LoopEntry:
    """The entry of the last loop of a nonempty loop prefix."""

    prefix: Prefix


@dataclass(frozen=True)
class ProtocolChoice:
    """
    An access of the level of a tensor at a zero-based stored depth by a query,
    at the last loop of prefix, which reaches the level: it ends at the last
    loop over a field of the levels from the root to it, since the level is
    reached once the positions of its parents are known. table is the read,
    in the expression of prefix, or, if write, the table of the output of the
    query, and option is the kind of the level. A read gives its protocol, or
    takes the default of option. A write laminates the level where the query
    reduces into its output, or where the loops up to the level aren't its
    fields in storage order, and otherwise extrudes it, as util.write_protocols
    infers. The kind of the access names the kind of the level and the
    protocol. parents is how many distinct fibers of the level the access
    enters, the positions of the level above it reaches: every slot of a
    dense level, which has a fiber at each, or only the entries of a sparse
    one, since a lookup which finds none runs nothing within it. It defaults
    to what the stats say, and is a feature of its own, while the entries the
    access reaches are the tensor's at the fields of the level, however many
    fibers hold them.
    """

    prefix: Prefix
    table: Table
    depth: int
    option: LevelOption
    write: bool = False
    parents: float | None = None

    @property
    def alias(self) -> Alias:
        match self.table.tns:
            case Alias() as tns:
                return tns
        raise ValueError(f"Expected a table of an alias, got {self.table}")

    @property
    def fields(self) -> tuple[Field, ...]:
        """The fields of the level and of the levels above it."""
        return self.table.idxs[: self.alias.nfused + self.depth + 1]

    @property
    def field(self) -> Field:
        return self.fields[-1]

    @property
    def protocol(self) -> Protocol:
        if self.write:
            # An update which reduces into its output has the output's table
            # as the init of its aggregate, as QueryInto.as_query writes it.
            match self.prefix.expr:
                case Aggregate(Literal(op), Table(), _, _) if op != ffuncs.overwrite:
                    return Laminate()
            return Extrude() if self.prefix.prefix == self.fields else Laminate()
        protocols = self.table.protocols
        proto = protocols[self.alias.nfused + self.depth] if protocols else None
        if proto is None or proto == DefaultProtocol():
            return self.option.default_protocol
        return proto

    @property
    def kind(self) -> str:
        return f"{self.option.kind}.{self.protocol}"


ProtocolComponent = LoopEntry | ProtocolChoice


def reached_depths(
    loops: tuple[Field, ...], idxs: tuple[Field, ...], nfused: int
) -> Iterator[int]:
    """
    The zero-based stored depths of the levels of a table of idxs, whose first
    nfused fields are fused, which are reached at the last of loops: those
    whose fields, and the fields of the levels above them, are all looped over
    by then, the last of them there.
    """
    for depth in range(len(idxs) - nfused):
        order = idxs[: nfused + depth + 1]
        if loops[-1] in order and set(order) <= set(loops):
            yield depth


def level_reads(
    prefix: Prefix, formats: Mapping[Alias, Sequence[LevelOption]]
) -> Iterator[ProtocolChoice]:
    """
    The levels which the tables of the query of prefix read at its last loop:
    those of the tensors of formats, whose options list the levels each alias
    stores, outermost first. A table which occurs several times reads each
    level once.
    """
    loops = prefix.prefix
    if not loops:
        return
    match prefix.expr:
        case Aggregate(arg=arg):
            pass
        case arg:
            pass
    for table in dict.fromkeys(PostOrderDFS(arg)):
        match table:
            case Table(Alias() as tns, idxs) if tns.unfused in formats:
                options = formats[tns.unfused]
                for depth in reached_depths(loops, idxs, tns.nfused):
                    yield ProtocolChoice(prefix, table, depth, options[depth])


def level_writes(
    prefix: Prefix, formats: Mapping[Alias, Sequence[LevelOption]]
) -> Iterator[ProtocolChoice]:
    """
    The levels of the output of the query of prefix which it writes at its
    last loop, if formats lists the options of its levels.
    """
    loops, idxs = prefix.prefix, prefix.output_vars
    match prefix.output:
        case Alias() as tns if loops and idxs is not None and tns.unfused in formats:
            options = formats[tns.unfused]
            table = Table(tns, idxs)
            for depth in reached_depths(loops, idxs, tns.nfused):
                yield ProtocolChoice(prefix, table, depth, options[depth], write=True)


def level_accesses(
    prefix: Prefix, formats: Mapping[Alias, Sequence[LevelOption]]
) -> Iterator[ProtocolChoice]:
    """The reads, then the writes, of levels at the last loop of prefix."""
    yield from level_reads(prefix, formats)
    yield from level_writes(prefix, formats)


def loop_components(
    prefix: Prefix, formats: Mapping[Alias, Sequence[LevelOption]]
) -> Iterator[ProtocolComponent]:
    """
    The components of the last loop of prefix: its entry, then the accesses
    at it. The empty prefix has none.
    """
    if not prefix.prefix:
        return
    yield LoopEntry(prefix)
    yield from level_accesses(prefix, formats)


KINDS: tuple[LevelOption, ...] = (
    DenseLevelOption(),
    SparseListLevelOption(),
    SparseHashLevelOption(),
    SparseByteMapLevelOption(),
)
LOOP_FEATURES = ("executions", "iterations")
FEATURES = (
    "seeks",
    "parents",
    "follow_visits",
    "walk_visits",
    "gallop_visits",
    "dense_visits",
    "distinct_reads",
    "extrude_visits",
    "laminate_visits",
    "distinct_writes",
)


class ProtocolBasis(Basis[ProtocolComponent], LoopOrderedForm):
    """
    A basis where each loop, and each access of a level at it, is a
    component, as loop_components lists them for each loop prefix. The row of
    a loop counts its executions, how often it is entered, and its
    iterations, in the first block of columns. The row of an access describes
    it in the block of columns of the kind of the level. Every access seeks a
    fiber of the level each time the loop is entered, and counts its parents,
    the distinct fibers it enters, as ProtocolChoice takes them. A read counts
    the visits of the loop where the level is followed, which look it up, the
    entries its walk steps through, or the jumps of its gallop, which are
    bounded by the visits the other tables impose and by the smallest
    galloped support, with the visits the loop would make if the level were
    dense, so that it couldn't restrict the loop, and the distinct entries of
    the level which the enclosing loops reach. A write counts the visits of
    the loop, as extrusions or laminations by its protocol, and the distinct
    entries they write, which are fewer where the loop laminates the level,
    reducing into it or writing it out of order; the level stores an entry
    wherever it is written, even with its fill, so they are the visits of the
    loop at the fields of the level. The loops within a loop are entered
    where the formats say, as LoopSpaces counts them, whatever the protocols
    at it, so the protocols at a loop only change its own rows. The formats
    of each prefix list the level options of each alias, as level_formats
    reads them from bindings.
    """

    class_key = "protocol"

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[ProtocolComponent]:
        formats = level_formats(bindings)
        for prefix in prefixes(term, machine, formats=formats):
            yield from loop_components(prefix, formats)

    @property
    def n_features(self) -> int:
        return len(LOOP_FEATURES) + len(KINDS) * len(FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *(f"loop.{name}" for name in LOOP_FEATURES),
            *(f"{option.kind}.{name}" for option in KINDS for name in FEATURES),
        )

    def get_component_features(
        self,
        component: ProtocolComponent,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        row = np.zeros((1, self.n_features))
        spaces = LoopSpaces(component.prefix, machine)
        match component:
            case LoopEntry():
                row[0, : len(LOOP_FEATURES)] = [
                    spaces.executions(),
                    spaces.iterations(),
                ]
                return row
            case ProtocolChoice(prefix=prefix, table=table):
                pass
        block = [option.kind for option in KINDS].index(component.option.kind)
        features = dict.fromkeys(FEATURES, 0.0)
        features["seeks"] = spaces.executions()
        visited = spaces.visited()
        # The fibers entered are at the positions the enclosing loops reach,
        # or, where a level above is reached at this loop, those it visits.
        above = component.fields[:-1]
        entered = visited if spaces.idx in above else spaces.exec_space
        features["parents"] = spaces.located(entered, above)
        if component.parents is not None:
            features["parents"] = component.parents
        if component.write:
            match component.protocol:
                case Extrude():
                    features["extrude_visits"] = spaces.count(visited)
                case Laminate():
                    features["laminate_visits"] = spaces.count(visited)
            features["distinct_writes"] = spaces.located(visited, component.fields)
        else:
            dense = LoopSpaces(
                replace(prefix, expr=followed(prefix.expr, table)), machine
            )
            dense_visits = dense.count(dense.visited())
            match component.protocol:
                case Follow():
                    features["follow_visits"] = spaces.count(visited)
                case Walk():
                    features["walk_visits"] = spaces.walk(table)
                case Gallop():
                    galloped = []
                    for node in dict.fromkeys(PostOrderDFS(spaces.arg)):
                        match node:
                            case Table() if spaces.protocol(node) == Gallop():
                                galloped.append(node)
                    smallest = min(
                        (spaces.walk(node) for node in galloped), default=0.0
                    )
                    features["gallop_visits"] = min(dense_visits, smallest)
            features["dense_visits"] = dense_visits
            features["distinct_reads"] = spaces.located(
                spaces.stepped(table), component.fields
            )
        start = len(LOOP_FEATURES) + block * len(FEATURES)
        row[0, start : start + len(FEATURES)] = list(features.values())
        return row


PROTOCOL_BASIS = ProtocolBasis()

# Until the model is fitted, entering a loop and each of its iterations cost
# 1, as does seeking a fiber. A walk costs 1 for each entry it steps through,
# a gallop costs 4 for each jump, and a follow costs a lookup for each visit:
# 1 in a dense level or a bytemap, 3 in a hash table, and 4 in a list, which
# is searched. The fibers entered, the visits a dense level would make, and
# the distinct entries read are free, since the seeks and visits already
# count them. A write costs a lookup, a lamination of a list costs a search
# like a follow, and each distinct entry written costs 1 more to insert,
# except in a dense level, which has a slot for each.
_LOOP_COSTS: tuple[float, ...] = (1, 1)
_PROTOCOL_COSTS: dict[str, tuple[float, ...]] = {
    "dense": (1, 0, 1, 0, 0, 0, 0, 1, 1, 0),
    "sparse_list": (1, 0, 4, 1, 4, 0, 0, 1, 4, 1),
    "sparse_hash": (1, 0, 3, 1, 0, 0, 0, 3, 3, 1),
    "sparse_bytemap": (1, 0, 1, 1, 4, 0, 0, 1, 1, 1),
}
PROTOCOL_COST_MODEL = LinearCostModel(
    PROTOCOL_BASIS,
    np.concatenate([_LOOP_COSTS, *(_PROTOCOL_COSTS[option.kind] for option in KINDS)]),
)
