"""
The physical cost model prices the levels of the formatted tensors of a
program, as StorageBasis does, and the accesses of them at the loops of each
query, as ProtocolBasis does, each kind of level differently, so that the
protocol optimizer can choose the protocol of each read, and the formatter the
kind of each level.
"""

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from finch.algebra import TensorFType, ffuncs, ftypes, np_dtype
from finch.autoschedule import util
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Extrude,
    Field,
    Follow,
    Fuse,
    Gallop,
    HardAlias,
    Laminate,
    Literal,
    LogicStatement,
    Plan,
    Protocol,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    Walk,
)
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite
from finch.tensor import (
    DenseLevelFType,
    ElementLevelFType,
    FiberTensorFType,
    SparseByteMapLevelFType,
    SparseHashLevelFType,
    SparseListLevelFType,
)

from .cost_models import NS, Affine, Basis, ByKind, Concat, CostModel
from .loop_cost_models import (
    LoopSpaces,
    Prefix,
    loop_arg,
    loop_spaces,
    prefixes,
    restrict,
)


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


class LevelOption(ABC):
    """
    A kind of level, which gives the storage features feature_names names, and
    the positions it passes to its child, from the stats of the tensor, the
    zero-based depth of the level, and the num_pos positions entering it. A
    tensor which stores one slice that fixes the fields of over is described by
    its largest slice. A level which filters has a fiber only where it has an
    entry, so a lookup which finds none runs nothing within it.
    """

    kind: ClassVar[str]
    feature_names: ClassVar[tuple[str, ...]]
    protocols: ClassVar[tuple[Protocol, ...]]
    default_protocol: ClassVar[Protocol]
    filters: ClassVar[bool]

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
    default_protocol = Follow()
    filters = False

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
    default_protocol = Walk()
    filters = True

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
    default_protocol = Walk()
    filters = True

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
    default_protocol = Walk()
    filters = True

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
    # A value is looked up at the position of its parent, and written with it.
    protocols = (Follow(), Extrude(), Laminate())
    default_protocol = Follow()
    # An element has no field, and a value at every position.
    filters = False

    def next_num_pos(self, stats, stats_factory, depth, num_pos, over):
        # The element level has no child, so its positions pass through.
        return num_pos

    def get_storage_features(
        self, stats, stats_factory, depth, num_pos, idx_size, pos_size, val_size, over
    ):
        return np.array([num_pos, num_pos * val_size])


# The option of each kind of level which a format can store.
LEVEL_FTYPES: dict[type, LevelOption] = {
    DenseLevelFType: DenseLevelOption(),
    SparseListLevelFType: SparseListLevelOption(),
    SparseHashLevelFType: SparseHashLevelOption(),
    SparseByteMapLevelFType: SparseByteMapLevelOption(),
}


def level_options(fmt: TensorFType) -> list[LevelOption] | None:
    """
    The options of the levels which a fiber tensor of the format fmt stores,
    outermost first and without its element level, or None where fmt isn't a
    fiber tensor or stores a level of no option.
    """
    match fmt:
        case FiberTensorFType(lvl_t=lvl):
            pass
        case _:
            return None
    options = []
    while True:
        match lvl:
            case ElementLevelFType():
                return options
            case _ if type(lvl) in LEVEL_FTYPES:
                options.append(LEVEL_FTYPES[type(lvl)])
                lvl = lvl.lvl_t
            case _:
                return None


def level_formats(bindings: Mapping[Alias, Any]) -> dict[Alias, Sequence[LevelOption]]:
    """
    The options of the levels each alias of bindings stores, outermost first
    and ending with its element level, as the bases of formatted programs take
    them. A binding is either such a list, or a format, whose options
    level_options lists. An alias whose format stores a level of no option is
    left out.
    """
    formats: dict[Alias, Sequence[LevelOption]] = {}
    for alias, fmt in bindings.items():
        match fmt:
            case TensorFType():
                options = level_options(fmt)
                if options is not None:
                    formats[alias] = [*options, ElementLevelOption()]
            case _:
                formats[alias] = fmt
    return formats


def read_protocols(alias: Alias, depth: int, term: LogicStatement) -> set[Protocol]:
    """
    The protocols which term reads the level of alias at a depth with. A read
    which gives no protocol takes whichever the level supports, which the
    protocol optimizer chooses, so it constrains nothing.
    """
    protocols = set()
    for node in PostOrderDFS(term):
        match node:
            case Query(_, rhs) | QueryInto(_, _, rhs):
                for table in PostOrderDFS(rhs):
                    match table:
                        case Table(Alias() as tns, idxs, (_, *_) as protos) if (
                            tns.unfused == alias.unfused
                            and tns.nfused + depth < len(idxs)
                            and protos[tns.nfused + depth] != DefaultProtocol()
                        ):
                            protocols.add(protos[tns.nfused + depth])
    return protocols


def write_protocols(alias: Alias, depth: int, term: LogicStatement) -> set[Protocol]:
    """
    The protocols which term writes the level of alias at a zero-based stored
    depth with, as util.write_protocols infers them under the loops of the
    enclosing Fuse nodes. An update which reduces into alias laminates what it
    already stores, but one which overwrites it or sets it to a literal starts
    it again.
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
                protocols.add(util.write_protocols(idxs, loop_order, lhs.nfused)[depth])

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


# bindings name no position type, so positions are intp.
POS_SIZE = np_dtype(ftypes.intp).itemsize

# The kinds of level which store a field, which a format chooses between, and
# every kind the bases price, the element level among them.
FIELD_KINDS: tuple[LevelOption, ...] = (
    DenseLevelOption(),
    SparseListLevelOption(),
    SparseHashLevelOption(),
    SparseByteMapLevelOption(),
)
KINDS: tuple[LevelOption, ...] = (*FIELD_KINDS, ElementLevelOption())


def option_kind(component: "ProtocolChoice") -> str:
    return component.option.kind


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
    An access of the level of a tensor at a zero-based stored depth by a query,
    at the last loop of prefix, which reaches the level: the last loop over a
    field of the levels from the root to it, or, for the element level at the
    rank of the tensor, to the last of them. table is the read, in the
    expression of prefix, or, if write, the table of the output of the query,
    and options are the kinds of the levels of the tensor, outermost first.
    """

    prefix: Prefix
    table: Table
    depth: int
    options: Sequence[LevelOption]
    write: bool = False

    @property
    def option(self) -> LevelOption:
        return self.options[self.depth]

    @property
    def alias(self) -> Alias:
        match self.table.tns:
            case Alias() as tns:
                return tns
        raise ValueError(f"Expected a table of an alias, got {self.table}")

    @property
    def parent_fields(self) -> tuple[Field, ...]:
        """The fields of the levels above the level."""
        return self.table.idxs[: self.alias.nfused + self.depth]

    @property
    def fields(self) -> tuple[Field, ...]:
        """The fields of the level and of the levels above it."""
        return self.table.idxs[: self.alias.nfused + self.depth + 1]

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
                    yield ProtocolChoice(prefix, table, depth, options)


def rescans_a_root(
    loops: tuple[Field, ...],
    reads: Sequence[ProtocolChoice],
    protocols: Sequence[Protocol],
) -> bool:
    """
    Whether reading reads with protocols at the last of loops, within the
    others, steps through the root level of a table, such as a vector, which
    has the same entries at every execution of the loop, where a sparse level
    of another read, whose fiber the enclosing loops choose, could lead the
    loop instead. A walk steps through the root again at every execution, and
    so does a gallop of it beside a walk, which ends no phase, or beside no
    other gallop; a gallop leapfrogging with that of another read doesn't.
    The length of the root is exact while DC stats bound the other fibers by
    the largest, so the model may price stepping through it below following
    it, which looks up only where the others lead. Reads whose protocols the
    program gives are not among reads.
    """
    roots, leads = [], []
    for read, protocol in zip(reads, protocols, strict=True):
        level = read.alias.nfused + read.depth
        above = read.table.idxs[:level]
        if read.table.idxs[level] != loops[-1]:
            continue
        if not above:
            roots.append(protocol)
        elif read.option.filters and set(above) <= set(loops):
            leads.append(protocol)
    leapfrogs = Gallop() in leads and Walk() not in protocols
    return (
        len(loops) > 1
        and bool(leads)
        and any(p == Walk() or (p == Gallop() and not leapfrogs) for p in roots)
    )


def level_accesses(
    prefix: Prefix, formats: Mapping[Alias, Sequence[LevelOption]]
) -> Iterator[ProtocolChoice]:
    """
    The reads of levels at the last loop of prefix, then the writes of the
    levels of its output, if formats lists the options of its levels, with the
    write of its element level where the last of them is reached.
    """
    yield from level_reads(prefix, formats)
    loops, idxs = prefix.prefix, prefix.output_vars
    match prefix.output:
        case Alias() as tns if loops and idxs is not None and tns.unfused in formats:
            options = formats[tns.unfused]
            table = Table(tns, idxs)
            rank = len(idxs) - tns.nfused
            for depth in reached_depths(loops, idxs, tns.nfused):
                yield ProtocolChoice(prefix, table, depth, options, write=True)
                if depth == rank - 1 and len(options) > rank:
                    yield ProtocolChoice(prefix, table, rank, options, write=True)


# The features of an access: the times its loop is entered, each a seek of a
# fiber, the points the loop visits, the visits of its protocol, which are the
# entries a walk steps through, the jumps of a gallop, or the lookups or
# writes of the rest, and the distinct fibers entered and entries touched.
FEATURES = ("executions", "iterations", "visits", "parents", "distinct")


def accesses(
    term: LogicStatement, bindings: Mapping[Alias, Any], machine: StatsMachine[NS]
) -> Iterator[ProtocolChoice]:
    """
    The accesses of the levels of the formatted tensors of bindings at the
    last loop of each loop prefix of term, advancing machine past each query.
    """
    formats = level_formats(bindings)
    for prefix in prefixes(term, machine, formats=formats):
        yield from level_accesses(prefix, formats)


def parents(access: ProtocolChoice, spaces: LoopSpaces) -> float:
    """
    The fibers of the level an access enters: those at the positions the
    enclosing loops reach, or, where a level above is reached at this loop,
    those it visits.
    """
    above = access.parent_fields
    entered = spaces.visited() if spaces.idx in above else spaces.exec_space
    return spaces.located(entered, above)


class AccessBasis(Basis[ProtocolChoice], LoopOrderedForm):
    """
    A basis whose components are the accesses of the levels of the formatted
    tensors at the last loop of each loop prefix, each a row of FEATURES. The
    loops within a loop are entered where the formats say, whatever the
    protocols at it, so the protocols at a loop only change its own rows, and
    choose_protocols picks each loop on its own.
    """

    @property
    def n_features(self) -> int:
        return len(FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURES

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[ProtocolChoice]:
        return accesses(term, bindings, machine)

    def get_component_features(
        self,
        component: ProtocolChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        prefix, table = component.prefix, component.table
        spaces = loop_spaces(prefix, machine)
        visited = spaces.visited()
        visits = spaces.count(visited)
        iterations = spaces.iterations()
        if component.write and component.alias.nfused:
            # There is some minimal work done every time a workspace is
            # declared or copied out, regardless of actual writes. We model
            # this as at least 1 write per declare.
            fused = table.idxs[: component.alias.nfused]
            last = max(prefix.prefix.index(idx) for idx in fused)
            outer = Prefix(
                prefix.prefix[: last + 1],
                prefix.expr,
                prefix.output_vars,
                prefix.formats,
                prefix.output,
            )
            iterations = max(iterations, loop_spaces(outer, machine).iterations())
            visits = max(visits, loop_spaces(outer, machine).iterations())
        if component.write:
            distinct = spaces.located(visited, component.fields)
        else:
            distinct = spaces.located(spaces.stepped(table), component.fields)
            arg = loop_arg(prefix.expr)
            galloped = []
            for node in dict.fromkeys(PostOrderDFS(arg)):
                match node:
                    case Table(Alias() as tns, idxs) if (
                        spaces.idx in idxs[tns.nfused :]
                        and spaces.protocol(node) == Gallop()
                    ):
                        galloped.append(node)
            match component.protocol:
                case Walk() if not galloped:
                    # With no gallop to lead it, the loop merges its walks,
                    # skipping a fiber only where they have no entries as the
                    # expression combines them, and otherwise stepping through
                    # every entry of each, not only those the others share.
                    stats_factory = machine.stats_factory
                    walked = spaces.walked(arg)
                    # The table of the access is among the walks.
                    assert walked is not None
                    visits = spaces.count(
                        stats_factory.mapjoin(
                            ffuncs.and_,
                            spaces.exec_space,
                            stats_factory.support(walked),
                            spaces.project(machine(table), (spaces.idx,)),
                        )
                    )
                case Walk():
                    visits = spaces.walk(table)
                case Gallop():
                    # A gallop jumps once for each entry of the smallest support
                    # among the arguments galloped at the loop, in the fibers it
                    # enters: gallops leapfrog, and walks are stepped within
                    # their phases.
                    visits = min(spaces.walk(node) for node in galloped)
        return np.array(
            [
                [
                    spaces.executions(),
                    iterations,
                    visits,
                    parents(component, spaces),
                    distinct,
                ]
            ]
        )


def protocol_name(protocol: Protocol) -> str:
    return type(protocol).__name__.lower()


def access_protocol(access: ProtocolChoice) -> str:
    return protocol_name(access.protocol)


def protocol_names(option: LevelOption) -> list[str]:
    """The protocols a level of option is accessed with, once chosen."""
    return [
        protocol_name(proto) for proto in option.protocols if proto != DefaultProtocol()
    ]


class ProtocolBasis(ByKind[ProtocolChoice, str]):
    """
    The accesses of the levels of each formatted tensor, each under the basis
    of the kind of its level and, within it, of its protocol, so that a
    linear model prices each feature of an access by both.
    """

    def __init__(self):
        super().__init__(
            {
                option.kind: ByKind(
                    dict.fromkeys(protocol_names(option), AccessBasis()),
                    access_protocol,
                )
                for option in KINDS
            },
            option_kind,
        )


def written(prefix: Prefix, machine: StatsMachine[NS]) -> NS:
    """
    The stats of where the query of prefix writes its output: those of its
    expression, with each table whose levels from the last which filters down
    don't filter spanning their fields wholly, as the formats of prefix give
    them. A read of such a level finds a value at every slot, so the query
    writes wherever it reads one, whatever the value.
    """
    formats = prefix.formats or {}
    stats_factory = machine.stats_factory
    bindings = OrderedDict(machine.bindings)

    def span(node):
        match node:
            case Table(Alias() as tns, idxs) if tns.unfused in formats:
                # The element level, the last option, has no field.
                levels = zip(idxs[tns.nfused :], formats[tns.unfused][:-1], strict=True)
                dense: list[Field] = []
                for idx, option in levels:
                    dense = [*dense, idx] if not option.filters else []
                if dense:
                    stat = machine(Table(tns, idxs))
                    kept = restrict(stats_factory, stat, set(idxs) - set(dense))
                    spanned = stats_factory.reorder(
                        stats_factory.mapjoin(
                            ffuncs.or_, stats_factory.support(stat), kept
                        ),
                        idxs,
                    )
                    spanned.fill_type = stat.fill_type
                    alias = HardAlias(f"{tns.name}[{','.join(map(str, idxs))}]")
                    bindings[alias] = spanned
                    return Table(alias, idxs)
        return None

    expr = Rewrite(PostWalk(span))(prefix.expr)
    if expr == prefix.expr:
        return machine(expr)
    return StatsMachine(stats_factory, bindings)(expr)


class LevelBasis(Basis[ProtocolChoice], LoopOrderedForm):
    """
    A basis whose components are the accesses, as AccessBasis lists them, and
    whose rows are the storage features of option at the accesses which write
    a level of it: the level stores what the query writes, as written gives
    it, entered at the fibers the access enters. The tensors a query reads are
    already stored, so a read costs no storage.
    """

    def __init__(self, option: LevelOption):
        self.option = option

    @property
    def n_features(self) -> int:
        return len(self.option.feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.option.feature_names

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[ProtocolChoice]:
        return accesses(term, bindings, machine)

    def get_component_features(
        self,
        component: ProtocolChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        if not component.write:
            return np.zeros((1, self.n_features))
        stats_factory = machine.stats_factory
        idxs = component.table.idxs
        stats = written(component.prefix, machine)
        if any(idx not in idxs for idx in stats.index_order):
            stats = restrict(stats_factory, stats, idxs)
        stats = stats_factory.reorder(stats, idxs)
        nfused = component.alias.nfused
        depth = nfused + component.depth
        # A fused alias stores one slice at a time, described by its largest,
        # and the positions entering a level are those its parent passes.
        over = idxs[:nfused]
        num_pos = 1.0
        for k, option in enumerate(component.options[: component.depth]):
            num_pos = option.next_num_pos(
                stats, stats_factory, nfused + k, num_pos, over
            )
        # The element level has no dimension, and a broadcast dimension is
        # intp.
        idx_size = 0
        if depth < len(stats.shape_type):
            idx_size = np_dtype(stats.shape_type[depth] or ftypes.intp).itemsize
        features = self.option.get_storage_features(
            stats,
            stats_factory,
            depth,
            num_pos,
            idx_size,
            POS_SIZE,
            np_dtype(stats.element_type).itemsize,
            over,
        )
        return np.asarray(features, dtype=float).reshape(1, self.n_features)


class StorageBasis(ByKind[ProtocolChoice, str]):
    """The storage of the levels each access writes, each under the basis of
    the kind of its level."""

    def __init__(self):
        super().__init__(
            {option.kind: LevelBasis(option) for option in KINDS}, option_kind
        )


STORAGE_BASIS = StorageBasis()

# The basis of the physical optimizers prices the storage and the accesses of
# the levels together, with a constant for each access.
PROTOCOL_BASIS = Affine(Concat(STORAGE_BASIS, ProtocolBasis()))

# The protocol cost model, which the formatter, the workspacer, and the
# protocol optimizer price with. Its coefficients are those scripts/fitting
# fitted to the timings of its protocol stage, which fit_physical_data writes,
# so that a basis which changes refuses them until it is fitted again.
PROTOCOL_COST_MODEL = CostModel(
    PROTOCOL_BASIS, np.zeros(PROTOCOL_BASIS.n_features)
).from_json(
    json.loads(Path(__file__).with_name("protocol_cost_model.json").read_text())
)
