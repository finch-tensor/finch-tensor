"""
The protocol cost model prices the levels of the formatted tensors of a program
and the accesses of them at the loops of each query, each kind of level
differently, so that the protocol optimizer can choose the protocol of each
read, and the formatter the kind of each level.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    Laminate,
    Literal,
    LogicExpression,
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
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite
from finch.tensor import (
    DenseLevelFType,
    ElementLevelFType,
    FiberTensorFType,
    SparseByteMapLevelFType,
    SparseHashLevelFType,
    SparseListLevelFType,
)

from .cost_models import NS, Affine, Basis, CostModel
from .loop_cost_models import LoopSpaces, Prefix, prefixes


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


def num_slices(stats: NS, over: tuple = ()) -> float:
    """Return the number of slices of stats which fix the fields of over."""
    return float(np.prod([stats.get_dim_size(idx) for idx in over]))


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
    A kind of level, which gives the storage features feature_names names, and
    the positions it passes to its child, from the stats of the tensor, the
    zero-based depth of the level, and the num_pos positions entering it. A
    tensor which stores one slice that fixes the fields of over is described by
    its largest slice, and is reset once for each slice. A level which filters
    has a fiber only where it has an entry, so a lookup which finds none runs
    nothing within it.
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
    """
    Storage features are positions, slots, and the slots zeroed by resetting
    every slice.
    """

    kind = "dense"
    feature_names = ("positions", "slots", "reset_slots")
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
        slots = self.next_num_pos(stats, stats_factory, depth, num_pos, over)
        return np.array([num_pos, slots, slots * num_slices(stats, over)])


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
    positions, slots, entries, the bytes of its pointers and list, the entries
    cleared by resetting every slice, and the comparisons which sort the
    entries of every slice when it is frozen.
    """

    kind = "sparse_bytemap"
    feature_names = (
        "positions",
        "slots",
        "entries",
        "position_bytes",
        "reset_entries",
        "sort_compares",
    )
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
        return np.array(
            [
                num_pos,
                slots,
                nnz_l,
                (num_pos + 1 + nnz_l) * pos_size,
                nnz_l * num_slices(stats, over),
                nnz_l * np.log2(max(nnz_l, 2)) * num_slices(stats, over),
            ]
        )


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
    # A value is looked up at the position of its parent.
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


def written_aliases(term: LogicStatement) -> set[Alias]:
    """The unfused aliases of the tables which the queries of term write."""
    written = set()
    for node in PostOrderDFS(term):
        match node:
            case Query(Table(Alias() as lhs, _), _) | QueryInto(
                Table(Alias() as lhs, _), _, _
            ):
                written.add(lhs.unfused)
    return written


def allowed_options(
    alias: Alias, depth: int, term: LogicStatement, options: Sequence[LevelOption]
) -> list[LevelOption]:
    """
    The options which can take the level of alias at a depth in term: those
    which support the protocols term reads and writes it with.
    """
    protocols = read_protocols(alias, depth, term) | write_protocols(alias, depth, term)
    return [option for option in options if protocols <= set(option.protocols)]


def followed(expr: LogicExpression, table: Table) -> LogicExpression:
    """expr, where each table of the tensor and fields of table follows every
    field it reads, as dense levels are read."""

    def read(node):
        match node:
            case Table(tns, idxs, _) if (tns, idxs) == (table.tns, table.idxs):
                return Table(tns, idxs, (Follow(),) * len(idxs))
        return None

    return Rewrite(PostWalk(read))(expr)


@dataclass(frozen=True)
class StorageChoice:
    """
    The level option at a zero-based depth of the levels alias stores, priced
    over the blocks of alias in the program, as alias_blocks lists them, with
    the positions entering the level in each.
    """

    alias: Alias
    depth: int
    option: LevelOption
    num_pos: tuple[float, ...]
    blocks: list[tuple[Any, tuple[Field, ...]]] = field(compare=False, hash=False)


# bindings name no position type, so positions are intp.
POS_SIZE = np_dtype(ftypes.intp).itemsize

KINDS: tuple[LevelOption, ...] = (
    DenseLevelOption(),
    SparseListLevelOption(),
    SparseHashLevelOption(),
    SparseByteMapLevelOption(),
)
STORAGE_KINDS: tuple[LevelOption, ...] = (*KINDS, ElementLevelOption())

STORAGE_FEATURES = sum(len(option.feature_names) for option in STORAGE_KINDS)
STORAGE_FEATURE_NAMES = tuple(
    f"{option.kind}.{name}" for option in STORAGE_KINDS for name in option.feature_names
)


def storage_choices(
    alias: Alias,
    options: Sequence[LevelOption],
    term: LogicStatement,
    machine: StatsMachine[NS],
) -> Iterator[StorageChoice]:
    """
    The levels alias stores in term, of options outermost first, each with
    the positions entering it in each block of alias, from the stats of
    machine, which is left where it is.
    """
    blocks = alias_blocks(alias, term, machine)
    if not blocks:
        return
    num_pos = (1.0,) * len(blocks)
    for depth, option in enumerate(options):
        yield StorageChoice(alias, depth, option, num_pos, blocks)
        num_pos = tuple(
            option.next_num_pos(
                stats, machine.stats_factory, len(over) + depth, n, over
            )
            for (stats, over), n in zip(blocks, num_pos, strict=True)
        )


def storage_features(
    component: StorageChoice, machine: StatsMachine[NS], n_features: int
) -> np.ndarray:
    """
    The storage features of component, one row for each block of its alias,
    in the columns of its kind of the first STORAGE_FEATURES of n_features,
    which STORAGE_FEATURE_NAMES names.
    """
    start = 0
    for option in STORAGE_KINDS:
        if option.kind == component.option.kind:
            break
        start += len(option.feature_names)
    features = np.zeros((len(component.blocks), n_features))
    for row, ((stats, over), num_pos) in enumerate(
        zip(component.blocks, component.num_pos, strict=True)
    ):
        depth = len(over) + component.depth
        shape_type = stats.shape_type
        # The element level has no dimension, and a broadcast dimension is
        # intp.
        idx_size = 0
        if depth < len(shape_type):
            idx_size = np_dtype(shape_type[depth] or ftypes.intp).itemsize
        storage = component.option.get_storage_features(
            stats,
            machine.stats_factory,
            depth,
            num_pos,
            idx_size,
            POS_SIZE,
            np_dtype(stats.element_type).itemsize,
            over,
        )
        features[row, start : start + len(storage)] = storage
    return features


class StorageBasis(Basis[StorageChoice], LoopOrderedForm):
    """
    A tensor-local basis where each level of each formatted alias is a
    component, whose rows are the storage features of its option, one for
    each block of the alias in the program, in the block of columns of its
    kind. The levels depend on nothing but the stats of their alias, so this
    basis prices the storage of a format on its own. bindings lists the
    options of the levels an alias stores, as level_formats reads them.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[StorageChoice]:
        for alias, options in level_formats(bindings).items():
            yield from storage_choices(alias, options, term, machine)
        # The levels are priced from the stats before term, so the machine is
        # advanced past it only after every level.
        machine(term)

    @property
    def n_features(self) -> int:
        return STORAGE_FEATURES

    @property
    def feature_names(self) -> tuple[str, ...]:
        return STORAGE_FEATURE_NAMES

    def get_component_features(
        self,
        component: StorageChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        return storage_features(component, machine, self.n_features)


STORAGE_BASIS = StorageBasis()


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
    field of the levels from the root to it. table is the read, in the
    expression of prefix, or, if write, the table of the output of the query,
    and option is the kind of the level.
    """

    prefix: Prefix
    table: Table
    depth: int
    option: LevelOption
    write: bool = False

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


ProtocolComponent = StorageChoice | ProtocolChoice


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


def level_accesses(
    prefix: Prefix, formats: Mapping[Alias, Sequence[LevelOption]]
) -> Iterator[ProtocolChoice]:
    """
    The reads of levels at the last loop of prefix, then the writes of the
    levels of its output, if formats lists the options of its levels.
    """
    yield from level_reads(prefix, formats)
    loops, idxs = prefix.prefix, prefix.output_vars
    match prefix.output:
        case Alias() as tns if loops and idxs is not None and tns.unfused in formats:
            options = formats[tns.unfused]
            table = Table(tns, idxs)
            for depth in reached_depths(loops, idxs, tns.nfused):
                yield ProtocolChoice(prefix, table, depth, options[depth], write=True)


FEATURES = (
    "executions",
    "iterations",
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
    A basis whose components are the levels of each formatted alias which term
    writes, since the tensors it reads are already stored, and each access of
    a level at the last loop of each loop prefix. A level has one row for each
    block of its alias, in the storage columns of its kind, which come first.
    An access has one row in the columns of the kind of its level, which count
    FEATURES. The loops within a loop are entered where the formats say,
    whatever the protocols at it, so the protocols at a loop only change its
    own rows, and choose_protocols picks each loop on its own.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[ProtocolComponent]:
        formats = level_formats(bindings)
        # The levels are priced from the stats before term, which listing
        # them leaves in place, and then the accesses advance past each query.
        written = written_aliases(term)
        for alias, options in formats.items():
            if alias.unfused in written:
                yield from storage_choices(alias, options, term, machine)
        for prefix in prefixes(term, machine, formats=formats):
            yield from level_accesses(prefix, formats)

    @property
    def n_features(self) -> int:
        return STORAGE_FEATURES + len(KINDS) * len(FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *STORAGE_FEATURE_NAMES,
            *(f"{option.kind}.{name}" for option in KINDS for name in FEATURES),
        )

    def get_component_features(
        self,
        component: ProtocolComponent,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        match component:
            case StorageChoice():
                return storage_features(component, machine, self.n_features)
            case ProtocolChoice(prefix=prefix, table=table):
                pass
        row = np.zeros((1, self.n_features))
        spaces = LoopSpaces(prefix, machine)
        block = [option.kind for option in KINDS].index(component.option.kind)
        features = dict.fromkeys(FEATURES, 0.0)
        features["executions"] = spaces.executions()
        features["iterations"] = spaces.iterations()
        features["seeks"] = spaces.executions()
        visited = spaces.visited()
        # The fibers entered are at the positions the enclosing loops reach,
        # or, where a level above is reached at this loop, those it visits.
        above = component.fields[:-1]
        entered = visited if spaces.idx in above else spaces.exec_space
        features["parents"] = spaces.located(entered, above)
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
        start = STORAGE_FEATURES + block * len(FEATURES)
        row[0, start : start + len(FEATURES)] = list(features.values())
        return row


# The basis of the physical optimizers adds a constant to the cost of each
# component.
PROTOCOL_BASIS = Affine(ProtocolBasis())

# Until the model is fitted, each byte a level stores costs 1, a byte of flags
# among them, and its positions and entries nothing more, nor resetting and
# sorting its slices. Seeking a fiber
# costs 1, and entering a loop or iterating it nothing more. A walk costs 1
# for each entry it steps through, a gallop costs 4 for each jump, and a
# follow costs a lookup for each visit: 1 in a dense level or a bytemap, 3 in
# a hash table, and 4 in a list, which is searched. The fibers entered, the
# visits a dense level would make, and the distinct entries read are free,
# since the seeks and visits already count them. A write costs a lookup, a
# lamination of a list costs a search like a follow, and each distinct entry
# written costs 1 more to insert, except in a dense level, which has a slot
# for each.
_STORAGE_COSTS: dict[str, tuple[float, ...]] = {
    "dense": (0, 0, 0),
    "sparse_list": (0, 0, 1, 1),
    "sparse_hash": (0, 0, 1, 1),
    "sparse_bytemap": (0, 1, 0, 1, 0, 0),
    "element": (0, 1),
}
_PROTOCOL_COSTS: dict[str, tuple[float, ...]] = {
    "dense": (0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0),
    "sparse_list": (0, 0, 1, 0, 4, 1, 4, 0, 0, 1, 4, 1),
    "sparse_hash": (0, 0, 1, 0, 3, 1, 0, 0, 0, 3, 3, 1),
    "sparse_bytemap": (0, 0, 1, 0, 1, 1, 4, 0, 0, 1, 1, 1),
}
PROTOCOL_COST_MODEL = CostModel(
    PROTOCOL_BASIS,
    np.concatenate(
        [
            *(_STORAGE_COSTS[option.kind] for option in STORAGE_KINDS),
            *(_PROTOCOL_COSTS[option.kind] for option in KINDS),
            [0.0],
        ]
    ),
)
