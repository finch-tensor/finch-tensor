from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

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

from .cost_models import NS, Basis
from .loop_cost_models import LoopSpaces, Prefix


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
    written with its protocols, and a read which gives none uses its
    default_protocol.
    """

    kind: ClassVar[str]
    feature_names: ClassVar[tuple[str, ...]]
    protocols: ClassVar[tuple[Protocol, ...]]
    default_protocol: ClassVar[Protocol]
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
    default_protocol = Follow()

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
    # A value is looked up at the position of its parent.
    default_protocol = Follow()

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
    entering the level in each block of alias in term. executions, when
    given, lists how many fibers of the level each access of alias in term
    enters, in the order LevelBasis.get_access_features lists the accesses,
    as the choice of the level above makes it: every position of a level
    which can't walk, or only those a walk steps to. Without it, the accesses
    enter the fibers the stats say the loops reach. A fused view of alias
    stores one slice for each value of its fused fields, so its blocks are
    described by their largest slice.
    """

    term: LogicStatement
    alias: Alias
    depth: int
    num_pos: tuple[float, ...]
    option: LevelOption
    pos_size: int
    executions: tuple[float, ...] | None = None

    @property
    def kind(self) -> str:
        return self.option.kind


class FormatBasis(Basis[LevelChoice]):
    """
    A basis where each level of each formatted alias is a component.
    bindings lists the options of the levels an alias stores, as level_formats
    reads them. Each level is priced over every block of the alias in the
    program.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[LevelChoice]:
        # bindings name no position type, so positions are intp.
        pos_size = np_dtype(ftypes.intp).itemsize
        for alias, options in level_formats(bindings).items():
            blocks = alias_blocks(alias, term, machine)
            if not blocks:
                continue
            num_pos = (1.0,) * len(blocks)
            executions = None
            for depth, option in enumerate(options):
                choice = LevelChoice(
                    term, alias, depth, num_pos, option, pos_size, executions
                )
                yield choice
                executions = level_executions(choice, machine)
                num_pos = tuple(
                    option.next_num_pos(
                        stats, machine.stats_factory, len(over) + depth, n, over
                    )
                    for (stats, over), n in zip(blocks, num_pos, strict=True)
                )
        # The levels are priced over the whole of term from the stats before
        # it, so the machine is advanced past term only after every level.
        machine(term)


def with_protocols(
    expr: LogicExpression, table: Table, protocols: tuple[Protocol, ...]
) -> LogicExpression:
    """expr, where each table of the tensor and fields of table reads with
    protocols, or without any if protocols is empty."""

    def read(node):
        match node:
            case Table(tns, idxs, _) if (tns, idxs) == (table.tns, table.idxs):
                return Table(tns, idxs, protocols)
        return None

    return Rewrite(PostWalk(read))(expr)


# A table which follows every field it reads reads them as dense levels do.
def followed(expr: LogicExpression, table: Table) -> LogicExpression:
    return with_protocols(expr, table, (Follow(),) * len(table.idxs))


def without_protocols(expr: LogicExpression) -> LogicExpression:
    def read(node):
        match node:
            case Table(tns, idxs, (_, *_)):
                return Table(tns, idxs)
        return None

    return Rewrite(PostWalk(read))(expr)


class LevelBasis(FormatBasis, LoopOrderedForm):
    """
    The basis of the levels of one type of option. Each level contributes
    an allocation row for each block of its alias, and an access row for each
    query loop prefix that uses it. The features are the storage features of
    the option for the allocation, followed by visits, dense visits, entries,
    lookups, sequential writes, and scatter writes for the accesses. The accesses
    don't depend on the protocols the program reads with, which the formatter
    prices before they are chosen, so a model of this basis learns the cost
    of a level under the protocols the protocol optimizer chooses for it.
    """

    class_key = "level"

    # The accesses are cached across the bases of every option type, so that
    # the fibers a level passes to the next are read from the rows priced.
    _accesses: OrderedDict[Any, Any] = OrderedDict()

    def __init__(self, option_type: type[LevelOption]):
        self.option_type = option_type

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *self.option_type.feature_names,
            "visits",
            "dense_visits",
            "entries",
            "lookups",
            "sequential_writes",
            "scatter_writes",
        )

    def to_json(self) -> dict[str, Any]:
        return {**super().to_json(), "option": self.option_type.kind}

    def _from_json(self, data: Mapping[str, Any]) -> Self:
        if data["option"] != self.option_type.kind:
            raise ValueError(
                f"Expected the levels of {self.option_type.kind}, "
                f"got {data['option']!r}"
            )
        return type(self)(self.option_type)

    def get_access_features(
        self,
        alias: Alias,
        level: int,
        term: LogicStatement,
        machine: StatsMachine[NS],
        executions: tuple[float, ...] | None = None,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """
        Return the accesses to alias at a zero-based stored depth in each query
        of term: visits, dense visits, entries, lookups, sequential writes, and
        scatter writes, with one row per loop prefix that accesses the level in
        each query, and the fibers of the level below which each access enters.
        The accesses of a query are the reads of alias in its right-hand side,
        in post order, then its write of alias, if any. A level is reached at
        the last loop over a field of the levels from the root to it, since it
        is reached only once the positions of its parents are known. The loops
        are the fields of the enclosing Fuse nodes. A read of a level at the
        loop over its own field visits the level wherever that loop goes, with
        every table walked, or wherever it would go if the level were dense, so
        that the level couldn't restrict it, and the level stores its entries
        in the fibers the loop enters: the steps of a walk, which a gallop or a
        follow may skip. Any other read looks the level up wherever the loop
        visits, as does a write. A write is sequential where it extrudes the
        level, and scatters where it laminates it, as write_protocols infers.
        Every view of alias accesses its levels. The element level is accessed
        with its parent, and looks up its value. The protocols which term gives
        its reads are ignored: alias walks the levels above the level, and
        looks the level up as its option reads it: following it if the option
        can't walk, and walking it otherwise. The loops reaching the level run
        as often as they enter its fibers, which executions gives for each
        access, rather than as often as the stats say they reach them with the
        levels above walked. The fibers entered beyond those hold no entries
        of alias, so they add to the visits the level would make if it were
        dense and to its lookups, which the other tables drive, but not to the
        visits or entries of its walks. The level passes on the fibers below
        it which each access enters: those the loop visits if the level walks,
        or every one the loop would visit if the level were dense otherwise,
        since a level which can't walk has a fiber at every position.
        """
        features = []
        next_executions: list[float] = []
        stats_interpreter = StatsMachine(
            machine.stats_factory, OrderedDict(machine.bindings)
        )

        def visit_access(table, loop_order, rhs, lhs_idxs, write, rows):
            tns, idxs = table.tns, table.idxs
            given = None if executions is None else executions[len(next_executions)]
            if not idxs:
                next_executions.append(1.0 if given is None else given)
                return
            # The levels of a fused alias follow its fused fields.
            own = tns.nfused + level
            depth = min(own, len(idxs) - 1)
            order = tuple(idxs[: depth + 1])
            loops = loop_order[: max(map(loop_order.index, order)) + 1]
            # The level above is reached at the loops over its fields, or at
            # the same loops as a level without a field, which is reached with
            # it.
            reached = order[:-1] if own < len(idxs) else order
            above = loop_order[: max(map(loop_order.index, reached), default=-1) + 1]

            def read_as(protocol):
                protocols = [DefaultProtocol()] * len(idxs)
                if own < len(idxs):
                    protocols[own] = protocol
                return with_protocols(rhs, table, tuple(protocols))

            def spaces(protocol, scaled=True):
                """The spaces of the loop reaching the level, reading it with
                protocol, whose enclosing loops run as often as the stats say
                they enter the fibers of the level, or, if scaled, as often as
                the fibers given, as the loops between the level above and
                this one run as often under both."""
                prefix = Prefix(loops, read_as(protocol), lhs_idxs)
                if scaled and given is not None:
                    entered = 1.0
                    if above:
                        entered = LoopSpaces(
                            replace(prefix, prefix=above), stats_interpreter
                        ).iterations()
                    if entered:
                        estimated = LoopSpaces(prefix, stats_interpreter).executions()
                        prefix = replace(prefix, executions=estimated * given / entered)
                return LoopSpaces(prefix, stats_interpreter)

            walks = Walk() in self.option_type.protocols
            walked = spaces(DefaultProtocol(), scaled=False)
            row = rows.setdefault(loops, np.zeros(6))
            if write is not None:
                visits = walked.count(walked.visited())
                row[4 if write == Extrude() else 5] += visits
                next_executions.append(visits)
            elif self.option_type is ElementLevelOption or loops[-1] != idxs[depth]:
                read = spaces(DefaultProtocol() if walks else Follow())
                visits = read.count(read.visited())
                row[3] += visits
                next_executions.append(visits)
            else:
                dense = spaces(Follow())
                visits = walked.count(walked.visited())
                dense_visits = dense.count(dense.visited())
                row[0] += visits
                row[1] += dense_visits
                row[2] += walked.walk(table)
                next_executions.append(visits if walks else dense_visits)

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

            # The accesses of each loop prefix of the query share a row. The
            # reads are priced as walks, whatever their protocols.
            rhs, arg = without_protocols(rhs), without_protocols(arg)
            rows: dict[tuple[Field, ...], np.ndarray] = {}
            for node in PostOrderDFS(arg):
                match node:
                    case Table(Alias() as tns, _) if tns.unfused == alias.unfused:
                        visit_access(node, loop_order, rhs, lhs_idxs, None, rows)
            if lhs.unfused == alias.unfused:
                write = write_protocol(lhs, lhs_idxs, level, loop_order)
                # An update which reduces into alias laminates what it stores.
                match stmt:
                    case QueryInto(_, Literal(op), _) if op != ffuncs.overwrite:
                        write = Laminate()
                table = Table(lhs, lhs_idxs)
                visit_access(table, loop_order, rhs, lhs_idxs, write, rows)
            features.extend(rows.values())
            stats_interpreter(stmt)

        visit_stmt(term, ())
        return (
            np.asarray(features, dtype=float).reshape(-1, 6),
            tuple(next_executions),
        )

    def accesses(
        self,
        alias: Alias,
        level: int,
        term: LogicStatement,
        machine: StatsMachine[NS],
        executions: tuple[float, ...] | None = None,
    ) -> tuple[list[tuple[NS, tuple[Field, ...]]], np.ndarray, tuple[float, ...]]:
        """
        The blocks of alias in term, the access rows of its level at a
        zero-based stored depth, whose accesses enter the fibers executions
        gives, and the fibers each access enters below it. The rows don't
        depend on the positions of the level, so they are found once for every
        choice of the level entered the same, while machine holds the same
        stats.
        """
        bindings = machine.bindings
        key = (
            self.option_type,
            alias,
            level,
            term,
            executions,
            tuple((var, id(s)) for var, s in bindings.items()),
        )
        if key not in self._accesses:
            # The stats are kept with their rows, so that their ids aren't reused.
            self._accesses[key] = (
                list(bindings.values()),
                alias_blocks(alias, term, machine),
                *self.get_access_features(alias, level, term, machine, executions),
            )
            if len(self._accesses) > 64:
                self._accesses.popitem(last=False)
        _, blocks, access_features, next_executions = self._accesses[key]
        return blocks, access_features, next_executions

    def get_component_features(
        self,
        component: LevelChoice,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the allocation row of the level in each block of its alias,
        followed by its access rows in the whole program.
        """
        blocks, access_features, _ = self.accesses(
            component.alias,
            component.depth,
            component.term,
            machine,
            component.executions,
        )
        n_storage = len(self.option_type.feature_names)
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


def level_executions(
    choice: LevelChoice, machine: StatsMachine[NS]
) -> tuple[float, ...]:
    """The fibers of the level below the level of choice which each access
    of its alias enters, as LevelChoice takes them, under choice."""
    return LevelBasis(type(choice.option)).accesses(
        choice.alias, choice.depth, choice.term, machine, choice.executions
    )[2]
