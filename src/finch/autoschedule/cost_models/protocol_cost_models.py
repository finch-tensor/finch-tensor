"""
The protocol cost model prices the protocols which the tables of a query read
the levels of their tensors with, so that the protocol optimizer can choose
one for each read. Where the level cost model of format_cost_models prices the
levels a tensor stores, this prices the reads of them, loop by loop, each
protocol on each kind of level differently.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Field,
    Follow,
    Gallop,
    LogicStatement,
    Protocol,
    Table,
    Walk,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS, LinearCostModel
from .format_cost_models import (
    DenseLevelOption,
    LevelOption,
    SparseByteMapLevelOption,
    SparseHashLevelOption,
    SparseListLevelOption,
    followed,
    level_formats,
)
from .loop_cost_models import LoopBasis, LoopSpaces, Prefix, prefixes

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
class ProtocolChoice:
    """
    The protocol which a table of a query reads the level of its tensor at a
    zero-based stored depth with. prefix is the loop prefix of the query which
    reaches the level: it ends at the last loop over a field of the levels from
    the root to it, since the level is reached once the positions of its
    parents are known. table is the read, in the expression of prefix, and
    option is the kind of the level. The protocol is the one table gives the
    field of the level, or the default of option. The kind of the choice names
    the kind of the level and the protocol.
    """

    prefix: Prefix
    table: Table
    depth: int
    option: LevelOption

    @property
    def alias(self) -> Alias:
        match self.table.tns:
            case Alias() as tns:
                return tns
        raise ValueError(f"Expected a table of an alias, got {self.table}")

    @property
    def field(self) -> Field:
        return self.table.idxs[self.alias.nfused + self.depth]

    @property
    def protocol(self) -> Protocol:
        protocols = self.table.protocols
        proto = protocols[self.alias.nfused + self.depth] if protocols else None
        if proto is None or proto == DefaultProtocol():
            return self.option.default_protocol
        return proto

    @property
    def kind(self) -> str:
        return f"{self.option.kind}.{self.protocol}"


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
                for depth in range(len(idxs) - tns.nfused):
                    order = idxs[: tns.nfused + depth + 1]
                    if loops[-1] in order and set(order) <= set(loops):
                        yield ProtocolChoice(prefix, table, depth, options[depth])


KINDS: tuple[LevelOption, ...] = (
    DenseLevelOption(),
    SparseListLevelOption(),
    SparseHashLevelOption(),
    SparseByteMapLevelOption(),
)
FEATURES = ("seeks", "follow_visits", "walk_visits", "gallop_visits", "dense_visits")


class ProtocolBasis(LoopBasis, LoopOrderedForm):
    """
    A basis where each loop prefix is a component, whose row counts the reads
    of the levels reached at its last loop, as level_reads lists them, in a
    block of columns for each kind of level: the seeks of its fibers, one for
    each the loop enters, the visits of the loop where the level is followed,
    which look it up, the entries its walks step through, the jumps of its
    gallops, which are bounded by the visits the other tables impose and by
    the smallest galloped support, and the visits the loop would make if the
    level were dense, so that it couldn't restrict the loop. The formats of
    the prefix list the level options of each alias, as level_formats reads
    them from bindings.
    """

    class_key = "protocol"

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[Prefix]:
        return prefixes(term, machine, formats=level_formats(bindings))

    @property
    def n_features(self) -> int:
        return len(KINDS) * len(FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{option.kind}.{name}" for option in KINDS for name in FEATURES)

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        row = np.zeros((1, self.n_features))
        reads = list(level_reads(component, component.formats or {}))
        if not reads:
            return row
        spaces = LoopSpaces(component, machine)
        galloped = []
        for node in dict.fromkeys(PostOrderDFS(spaces.arg)):
            match node:
                case Table() if spaces.protocol(node) == Gallop():
                    galloped.append(node)
        smallest = min((spaces.walk(node) for node in galloped), default=0.0)
        kinds = [option.kind for option in KINDS]
        for read in reads:
            block = kinds.index(read.option.kind) * len(FEATURES)
            dense = LoopSpaces(
                replace(component, expr=followed(component.expr, read.table)), machine
            )
            dense_visits = dense.count(dense.visited())
            row[0, block] += spaces.executions()
            match read.protocol:
                case Follow():
                    row[0, block + 1] += spaces.count(spaces.visited())
                case Walk():
                    row[0, block + 2] += spaces.walk(read.table)
                case Gallop():
                    row[0, block + 3] += min(dense_visits, smallest)
            row[0, block + 4] += dense_visits
        return row


PROTOCOL_BASIS = ProtocolBasis()

# Until the model is fitted, seeking a fiber costs 1, a walk costs 1 for each
# entry it steps through, a gallop costs 4 for each jump, and a follow costs a
# lookup for each visit: 1 in a dense level or a bytemap, 3 in a hash table,
# and 4 in a list, which is searched. The visits a dense level would make are
# free, since the visits of a followed level already count them.
_PROTOCOL_COSTS: dict[str, tuple[float, ...]] = {
    "dense": (1, 1, 0, 0, 0),
    "sparse_list": (1, 4, 1, 4, 0),
    "sparse_hash": (1, 3, 1, 0, 0),
    "sparse_bytemap": (1, 1, 1, 4, 0),
}
PROTOCOL_COST_MODEL = LinearCostModel(
    PROTOCOL_BASIS,
    np.concatenate([_PROTOCOL_COSTS[option.kind] for option in KINDS]),
)
