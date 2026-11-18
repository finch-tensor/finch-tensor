import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic

import numpy as np

from finch.algebra import ffuncs, is_annihilator
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Field,
    Follow,
    Fuse,
    Gallop,
    HardAlias,
    Literal,
    LogicExpression,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Protocol,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite, Term

from .cost_models import NS, Affine, Basis, Concat, CostModel, Stratify


@dataclass(frozen=True)
class Prefix:
    """
    An outermost-first loop prefix of expr, the right-hand side of a query.
    output_vars orders the output of the query, when there is one, and output
    is the tensor it writes. formats, which bases that price formatted tensors
    need, lists the level options each alias stores, outermost first, as
    level_formats lists them.
    """

    prefix: tuple[Field, ...]
    expr: LogicExpression
    output_vars: tuple[Field, ...] | None = None
    formats: Mapping[Alias, Any] | None = field(default=None, compare=False, hash=False)
    output: Alias | Literal | None = None


def prefixes(
    term: LogicStatement,
    machine: StatsMachine[NS],
    loop_order: tuple[Field, ...] = (),
    formats: Mapping[Alias, Any] | None = None,
) -> Iterator[Prefix]:
    """
    The loop prefixes of each query of term, including the empty prefix, in
    query order and from the shortest to the longest, advancing machine past
    each query after its prefixes. The loop order of a query is the fields of
    its enclosing Fuse nodes.
    """
    match term:
        case Plan(bodies):
            for body in bodies:
                yield from prefixes(body, machine, loop_order, formats)
            return
        case Fuse(idx, body):
            yield from prefixes(body, machine, (*loop_order, idx), formats)
            return
        case QueryInto(Table(), _, Literal()):
            machine(term)
            return
        case Produces():
            return
        case Query(Table(lhs, lhs_idxs), expr):
            pass
        case QueryInto(Table(lhs, lhs_idxs), _, _):
            expr = term.as_query().rhs
        case _:
            raise ValueError(f"Unsupported statement type: {term}")

    for n in range(len(loop_order) + 1):
        yield Prefix(loop_order[:n], expr, lhs_idxs, formats, output=lhs)
    machine(term)


def restrict(stats_factory: StatsFactory, stat: NS, fields) -> NS:
    """The support of stat over fields, reducing the rest away."""
    reduce_idxs = tuple(idx for idx in stat.index_order if idx not in fields)
    return stats_factory.aggregate(
        ffuncs.or_, False, reduce_idxs, stats_factory.support(stat)
    )


class LoopBasis(Basis[Prefix]):
    """
    A basis where each loop prefix, including the empty prefix, is a component,
    as prefixes yields them, and get_component_features returns one row for
    each prefix. Models used with branch-and-bound loop search must predict
    nonnegative component costs.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[Prefix]:
        return prefixes(term, machine, formats=bindings)


def loop_arg(expr: LogicExpression) -> LogicExpression:
    """The expression the loops of a query merge: its aggregate's argument."""
    match expr:
        case Aggregate(arg=arg):
            return arg
    return expr


def unprotocoled(expr: LogicExpression) -> LogicExpression:
    """expr, with every table reading its fields without protocols."""

    def strip(node):
        match node:
            case Table(tns, idxs, (_, *_)):
                return Table(tns, idxs)
        return None

    return Rewrite(PostWalk(strip))(expr)


class LoopNest(Generic[NS]):
    """
    Where the loops of a loop prefix of arg, an expression without protocols,
    are entered, as the formats of its tables allow, whatever protocols read
    them. The merge of a nest is where every fiber merged has positions at the
    fields of the prefix, so the loop over the next field is entered there,
    and the loop over the last field is entered where the merge of its parent,
    the nest of the prefix without it, has positions. The root, the nest of the
    empty prefix, keeps the stats of the inputs of arg, the extent of each
    field, and the field of each level of each table and whether it doesn't
    filter, outermost first, where the formats list level options. The spaces
    of the last loop under each set of followed tables are kept in spaces.
    """

    def __init__(
        self,
        arg: LogicExpression,
        loops: tuple[Field, ...],
        formats: Mapping[Alias, Any],
        machine: StatsMachine[NS],
        parent: "LoopNest[NS] | None" = None,
    ):
        self.arg = arg
        self.loops = loops
        self.parent = parent
        self.stats_factory = machine.stats_factory
        self.spaces: dict[frozenset[Table], LoopSpaces[NS]] = {}
        if parent is not None:
            self.stats, self.dims, self.levels = (
                parent.stats,
                parent.dims,
                parent.levels,
            )
        else:
            self.stats: dict[LogicExpression, NS] = {}
            self.dims: dict[Field, float] = {}
            self.levels: dict[Table, list[tuple[Field, bool]]] = {}

            def read(node):
                match node:
                    case MapJoin(Literal(), args):
                        for child in args:
                            read(child)
                    case _:
                        self.stats[node] = machine(node)

            read(arg)
            for node in PostOrderDFS(arg):
                match node:
                    case Table(Alias() as tns, idxs) if idxs:
                        stat = self.stats.setdefault(node, machine(node))
                        self.dims.update(stat.dim_sizes)
                        match formats.get(tns.unfused):
                            case [*options]:
                                self.levels[node] = [
                                    (idxs[tns.nfused + depth], not option.filters)
                                    for depth, option in enumerate(options)
                                    if tns.nfused + depth < len(idxs)
                                ]
        self.merge = self.merged(arg)
        if parent is not None:
            self.exec_space = self.stats_factory.support(parent.merge)

    def unrestricted(self, table: Table, follows: bool) -> set[Field]:
        """
        The fields of the prefix which table doesn't restrict: the last, where
        table follows it, since it is looked up wherever the others go, and
        those of the dense levels down to the deepest level of the rest, from
        the sparse level above them, which have a fiber at every slot of it,
        so the positions of the level are the entries of the sparse one, at
        every slot of the dense ones.
        """
        unrestricted = {self.loops[-1]} if follows else set()
        fields = set(self.loops)
        run: set[Field] = set()
        trailing: set[Field] = set()
        for idx, dense in self.levels.get(table, ()):
            run = run | {idx} if dense else set()
            if idx in fields and idx not in unrestricted:
                trailing = run
        return unrestricted | trailing

    def merged(
        self, node: LogicExpression, followed: frozenset[Table] = frozenset()
    ) -> NS:
        """
        Where a merge of the inputs of node visits the fields of the prefix:
        where every fiber it merges has positions, as the formats give them,
        and the tables of followed span the last field wholly.
        """
        # A merge coiterates the projections of the inputs, so it visits more
        # of the prefix than the projection of their result.
        match node:
            case MapJoin(Literal(op), args):
                return self.stats_factory.mapjoin(
                    op, *(self.merged(arg, followed) for arg in args)
                )
        unrestricted: set[Field] = set()
        match node:
            case Table():
                unrestricted = self.unrestricted(node, node in followed)
        stat = self.stats[node]
        projected = restrict(self.stats_factory, stat, set(self.loops) - unrestricted)
        # The fill decides whether a mapjoin joins or unions it.
        projected.fill_type = stat.fill_type
        return projected


class LoopSpaces(Generic[NS]):
    """
    The spaces of the last loop of a nonempty loop nest: where it is entered,
    what it visits, and where its walks step, where the tables of followed
    span the field of the loop wholly, as they are looked up wherever the
    others go. A table whose formats the nest doesn't list is taken to be
    sparse.
    """

    def __init__(self, nest: LoopNest[NS], followed: frozenset[Table]):
        self.nest = nest
        self.stats_factory = nest.stats_factory
        self.parent = set(nest.loops[:-1])
        self.idx = nest.loops[-1]
        self.dims = nest.dims
        self.exec_space = nest.exec_space
        # The merge steps only where every fiber it merges has positions at idx.
        self._merged_at_idx = (
            nest.merged(nest.arg, followed) if followed else nest.merge
        )
        self.merge_space = self.project(self._merged_at_idx)
        self._visited: NS | None = None

    def project(self, stat: NS, fields=(), followed=()) -> NS:
        """
        The support of stat over parent and fields, less the fields of
        followed, which it spans wholly.
        """
        return restrict(self.stats_factory, stat, {*self.parent, *fields} - {*followed})

    def protocol(self, node: Term) -> Protocol:
        """The protocol node reads the field of the loop with."""
        match node:
            case Table(_, idxs, (_, *_) as protocols) if self.idx in idxs:
                return protocols[idxs.index(self.idx)]
        return DefaultProtocol()

    def walked(self, node: LogicExpression) -> NS | None:
        """
        Where the walks of node at the loop have fibers with entries, over
        parent, combined as node combines them, or None where it walks
        nothing. A merge of walks skips a fiber only where this is empty:
        under a product, wherever one walked fiber is; under a sum, only where
        all are.
        """
        match node:
            case MapJoin(Literal(op), args):
                parts = [part for arg in args if (part := self.walked(arg)) is not None]
                if not parts:
                    return None
                joins = []
                unions = []
                for part in parts:
                    annihilates = is_annihilator(op.ftype, part.fill_type)
                    (joins if annihilates else unions).append(
                        self.stats_factory.support(part)
                    )
                if unions:
                    joins.append(self.stats_factory.mapjoin(ffuncs.or_, *unions))
                return self.stats_factory.mapjoin(ffuncs.and_, *joins)
            case Table(Alias() as tns, idxs) if self.idx in idxs[
                tns.nfused :
            ] and self.protocol(node) not in (Follow(), Gallop()):
                stat = self.nest.stats[Table(tns, idxs)]
                projected = self.project(stat)
                projected.fill_type = stat.fill_type
                return projected
        return None

    def extent(self, stat: NS, fields) -> float:
        """The points of stat over fields, spanning all of those it lacks."""
        count = stat.estimate_non_fill_values()
        for idx in fields:
            if idx not in stat.index_order:
                count *= self.dims.get(idx, 1.0)
        return count

    def count(self, stat: NS) -> float:
        return self.extent(stat, (*self.parent, self.idx))

    def executions(self) -> float:
        return self.extent(self.exec_space, self.parent)

    def iterations(self) -> float:
        return self.count(self.visited())

    def visited(self) -> NS:
        """Where the loop visits idx, which its walks and gallops restrict."""
        if self._visited is None:
            self._visited = self.stats_factory.mapjoin(
                ffuncs.and_,
                self.exec_space,
                self.stats_factory.support(self._merged_at_idx),
            )
        return self._visited

    def located(self, stat: NS, fields) -> float:
        """
        The distinct points of stat at fields: the positions of a level of
        those fields which the loop reaches in stat, however often it runs.
        """
        if any(idx not in fields for idx in stat.index_order):
            stat = restrict(self.stats_factory, stat, fields)
        return self.extent(stat, fields)

    def stepped(self, table: Table) -> NS:
        """
        The entries of table at idx which a walk of it steps through: those
        where the merge steps, since it steps through nothing where another
        fiber it merges has no positions.
        """
        stat = self.nest.stats[Table(table.tns, table.idxs)]
        return self.stats_factory.mapjoin(
            ffuncs.and_, self.merge_space, self.project(stat, (self.idx,))
        )

    def walk(self, table: Table) -> float:
        return self.count(self.stepped(table))


def loop_spaces(component: Prefix, machine: StatsMachine[NS]) -> LoopSpaces[NS]:
    """
    The LoopSpaces of the last loop of component, a nonempty prefix, on the
    nest of its loops, which extends the nests of their prefixes. The cache of
    machine keeps the nests by the expression they merge, the density of its
    levels and the stats it binds, and then by loop prefix, so the prefixes
    of a search share them however their tables read their fields, as do
    formats which agree on which levels are dense.
    """
    arg = loop_arg(component.expr)
    bare = unprotocoled(arg)
    formats = component.formats or {}
    density = []
    bound = []
    idx = component.prefix[-1]
    followed = set()
    for node in dict.fromkeys(PostOrderDFS(arg)):
        match node:
            case Table(Alias() as tns, idxs, protocols):
                bound.append(machine.bindings.get(tns))
                match formats.get(tns.unfused):
                    case [*options]:
                        density.append(
                            (tns, idxs, tuple(not opt.filters for opt in options))
                        )
                if idx in idxs and protocols and protocols[idxs.index(idx)] == Follow():
                    followed.add(Table(tns, idxs))
    by_loops = machine.cache.setdefault((bare, tuple(density), tuple(bound)), {})
    nest = None
    for n in range(len(component.prefix) + 1):
        loops = component.prefix[:n]
        if loops not in by_loops:
            by_loops[loops] = LoopNest(bare, loops, formats, machine, nest)
        nest = by_loops[loops]
    assert nest is not None
    key = frozenset(followed)
    if key not in nest.spaces:
        nest.spaces[key] = LoopSpaces(nest, key)
    return nest.spaces[key]


class PigeonBasis(LoopBasis, LoopOrderedForm):
    """
    PigeonBasis counts iteration, operation, output, lookup, and dense
    iteration costs across loop prefixes. The empty prefix charges the
    non-fill values of the output once, and a nonempty prefix charges its last
    loop: a walk steps through the stored indices of its table, a gallop or a
    follow instead visits the indices where the expression runs, charging a
    lookup at each, and the dense iterations of a loop are those it would make
    if its tables were dense, so that none could restrict it: its whole
    extent, wherever its enclosing loops run.
    """

    @property
    def n_features(self) -> int:
        return 5

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("iterations", "operations", "outputs", "lookups", "dense_iterations")

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        prefix, expr, output_vars = (
            component.prefix,
            component.expr,
            component.output_vars,
        )
        stats_factory = machine.stats_factory
        out_space = machine(expr)
        if output_vars is not None:
            if any(idx not in output_vars for idx in out_space.index_order):
                out_space = restrict(stats_factory, out_space, output_vars)
            out_space = stats_factory.reorder(out_space, output_vars)
        if not prefix:
            return np.array(
                [[0.0, 0.0, out_space.estimate_non_fill_values(), 0.0, 0.0]]
            )
        spaces = loop_spaces(component, machine)
        idx = spaces.idx
        exec_space = spaces.exec_space
        inputs, walks, lookups = [], [], 0
        for node in PostOrderDFS(loop_arg(expr)):
            match node:
                case Table(_, idxs) if idx in idxs:
                    stat = machine(node)
                    inputs.append(stat)
                    match spaces.protocol(node):
                        case Follow() | Gallop():
                            lookups += 1
                        case _:
                            walks.append(spaces.project(stat, (idx,)))
        iterations = spaces.executions()
        lookup_cost = 0.0
        if lookups:
            visited = spaces.visited()
            # The loop steps through the tables it walks, and visits the rest.
            steps = stats_factory.mapjoin(ffuncs.or_, visited, *walks)
            iterations = spaces.count(
                stats_factory.mapjoin(ffuncs.and_, exec_space, steps)
            )
            lookup_cost = spaces.count(visited) * lookups
        elif inputs:
            iterations = spaces.count(
                stats_factory.mapjoin(
                    ffuncs.and_, exec_space, stats_factory.mapjoin(ffuncs.or_, *walks)
                )
            )
        out_cost = 0.0
        if idx in out_space.index_order:
            out_cost = spaces.project(out_space).estimate_non_fill_values()
        return np.array(
            [
                [
                    iterations,
                    spaces.executions() * len(inputs),
                    out_cost,
                    lookup_cost,
                    spaces.count(exec_space),
                ]
            ]
        )


def transposed(
    component: Prefix, machine: StatsMachine[NS]
) -> Iterator[tuple[Alias, NS]]:
    """
    The tensor of each table which the last loop of component reads out of the
    order it is stored in, the first time a prefix of the query does, with the
    stats of the table.
    """
    prefix, expr = component.prefix, component.expr
    if not prefix:
        return
    match expr:
        case Aggregate(arg=arg):
            pass
        case _:
            arg = expr
    for node in set(PostOrderDFS(arg)):
        match node:
            case Table(Alias() as tns, idxs) if prefix[-1] in idxs:
                stat = machine(node)
                order = stat.index_order
                parent = tuple(idx for idx in prefix[:-1] if idx in order)
                if parent == order[: len(parent)] and prefix[-1] != order[len(parent)]:
                    yield tns, stat


def transpose_copy(tns: Alias, stat: TensorStats, prefix: tuple[Field, ...]) -> Plan:
    """
    The copy concordize makes of tns, of stats stat, read out of the order it
    is stored in under prefix, as with_copy_loop_orders orders it: in the order
    tns stores, into a tensor which stores the fields prefix visits first, in
    its order, and then the rest, as tns stores them.
    """
    order = stat.index_order
    target = (
        *(idx for idx in prefix if idx in order),
        *(idx for idx in order if idx not in prefix),
    )
    copy = HardAlias("transposed")
    query = Query(Table(copy, target), Table(tns, order))
    return Plan((fuse_loop_order(query, order), Produces((copy,))))


class WithCopies(Basis[Prefix]):
    """
    The rows of basis, where a prefix whose last loop reads a table out of the
    order it is stored in, the first time a prefix of the query does, also has
    the rows basis gives the copy concordize makes of it, so that a model of
    basis prices transposing the table as it prices any query which scatters.
    """

    def __init__(self, basis: Basis[Prefix]):
        self.basis = basis

    @property
    def n_features(self) -> int:
        return self.basis.n_features

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.basis.feature_names

    @property
    def ridge(self) -> np.ndarray:
        return self.basis.ridge

    def validate_inputs(self, *inputs):
        self.basis.validate_inputs(*inputs)

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[Prefix]:
        return self.basis.components(term, bindings, machine)

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        rows = [self.basis.get_component_features(component, machine)]
        for tns, stat in transposed(component, machine):
            copy = transpose_copy(tns, stat, component.prefix)
            rows.append(
                self.basis.get_features(copy, {}, {tns: stat}, machine.stats_factory)
            )
        return np.concatenate(rows)


# The most entries a slice of a workspace may have, so that it can be reset and
# copied out cheaply.
class WriteBasis(LoopBasis, LoopOrderedForm):
    """
    The writes to the levels of the output of a query which the last loop of
    each prefix reaches: those whose fields, and those of the levels above
    them, the prefix loops over, the last of them last. Where the loops up to a
    level are its fields in storage order, the level is written once for each
    of its indices, in order, and otherwise its writes scatter, as
    util.write_protocols infers. The element level is written with its parent.
    Besides the writes, a row has the entries of the region of the output the
    writes scatter into, which WriteClasses classes it by, a slice over the
    fields after the leading ones the loops visit in storage order, which a
    workspace fused over those could hold, or none where nothing scatters, and
    the distinct entries of the levels reached. Sizes are products of the
    extents of the fields, where a field which no table has, such as a unit
    dimension of the output alone, has one index.
    """

    @property
    def n_features(self) -> int:
        return 4

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "sequential_writes",
            "scatter_writes",
            "scattered_region",
            "distinct_outputs",
        )

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        prefix, output_vars = component.prefix, written_fields(component)
        features = np.zeros((1, self.n_features))
        reached = [
            output_vars[: depth + 1]
            for depth in range(len(output_vars))
            if set(output_vars[: depth + 1]) <= set(prefix)
            and prefix[-1] in output_vars[: depth + 1]
        ]
        if not reached:
            return features
        dims: dict[Field, float] = {}
        for node in PostOrderDFS(component.expr):
            match node:
                case Table():
                    dims.update(machine(node).dim_sizes)

        def size(fields) -> float:
            return math.prod(dims.get(idx, 1.0) for idx in fields)

        spaces = loop_spaces(component, machine)
        visits = spaces.count(spaces.visited())
        output = machine(component.expr)
        for order in reached:
            features[0, 0 if prefix == order else 1] += visits
            distinct = restrict(machine.stats_factory, output, order)
            features[0, 3] += distinct.estimate_non_fill_values()
        if features[0, 1]:
            fused = 0
            while (
                fused < len(prefix)
                and prefix[fused : fused + 1] == output_vars[fused : fused + 1]
            ):
                fused += 1
            features[0, 2] = size(output_vars[fused:])
        return features


def written_fields(component: Prefix) -> tuple[Field, ...]:
    """The fields of the output of the query of component, in the order it
    stores them, which are those of its expression without output_vars."""
    if component.output_vars is None:
        return component.expr.fields()
    return component.output_vars


N_WRITE_CLASSES = 2


@dataclass(frozen=True)
class WriteClasses:
    """
    The class of each row of WriteBasis, whether the region its writes
    scatter into exceeds the region cutoff. The default cutoff is the most
    entries a workspace can reset and copy out cheaply.
    """

    region: float = 100_000

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        _, _, region, _ = rows.T
        return (region > self.region).astype(int)


write_class = WriteClasses()


class WriteStrata(Stratify):
    """
    The rows of WriteBasis in the classes of classes. A write which scatters
    into a region over the cutoff isn't staged in a workspace, but written
    into the output where it goes, so its region costs nothing.
    """

    def __init__(self, classes: WriteClasses):
        super().__init__(WriteBasis(), N_WRITE_CLASSES, classes)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        features = super().transform(rows)
        region = self.basis.feature_names.index("scattered_region")
        features[:, self.basis.n_features + region] = 0.0
        return features


def loop_basis(classes: WriteClasses = write_class) -> Basis[Prefix]:
    """The basis of the loop model, whose writes are of the classes of the
    region cutoff of classes."""
    return Affine(Concat(PigeonBasis(), WriteStrata(classes)))


def loop_cost_model(data: Mapping[str, Any]) -> CostModel[Prefix]:
    """The loop model of data, which fit_physical_data wrote, whose writes are
    classed by its region cutoff if it was tuned. The fit times each loop order
    with its inputs stored in that order, so a table read out of the order it
    is stored in costs, besides, what the model prices the copy concordize
    makes of it at."""
    basis = WithCopies(
        loop_basis(WriteClasses(data.get("region_cutoff", write_class.region)))
    )
    return CostModel(basis, np.zeros(basis.n_features)).from_json(data)


# The loop model, which the loop orderers price with. Its coefficients are
# those scripts/fitting fitted to the timings of its loop stage, which runs the
# workspacer, the formatter, and the protocol optimizer after each loop order,
# so a write which scatters costs what it does once it is staged.
LOOP_COST_MODEL = loop_cost_model(
    json.loads(Path(__file__).with_name("loop_cost_model.json").read_text())
)
