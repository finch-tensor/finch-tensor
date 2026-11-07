import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic

import numpy as np

from finch.algebra import ffuncs
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Field,
    Follow,
    Fuse,
    Gallop,
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
)
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite, Term

from .cost_models import NS, Basis, Concat, CostModel, Stratify


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


class TransposeBasis(LoopBasis, LoopOrderedForm):
    """
    Count dense and sparse transposed entries, charging each input once per
    query, at the first prefix whose last loop reads it out of its stored
    order.
    """

    @property
    def n_features(self) -> int:
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("dense_transposes", "sparse_transposes")

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        prefix, expr = component.prefix, component.expr
        features = np.zeros((1, self.n_features))
        if not prefix:
            return features

        match expr:
            case Aggregate(arg=arg):
                pass
            case _:
                arg = expr
        for node in set(PostOrderDFS(arg)):
            match node:
                case Table(_, idxs) if prefix[-1] in idxs:
                    stat = machine(node)
                case _:
                    continue
            order = stat.index_order
            parent = tuple(idx for idx in prefix[:-1] if idx in order)
            if parent == order[: len(parent)] and prefix[-1] != order[len(parent)]:
                nnz = stat.estimate_non_fill_values()
                space = stat.get_dim_space_size(order)
                dense = 0 < space < float("inf") and nnz / space > 0.1
                features[0, 0 if dense else 1] += nnz
        return features


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
    Besides the writes, a row has the measures which WriteClasses classes it
    by: the entries of the region of the output the writes scatter into, a
    slice over the fields after the leading ones the loops visit in storage
    order, which a workspace fused over those could hold, or none where
    nothing scatters, the distinct entries of the levels reached, those
    levels would have if dense, and the points of the loops if dense. Sizes
    are products of the extents of the fields, where a field which no table
    has, such as a unit dimension of the output alone, has one index.
    """

    @property
    def n_features(self) -> int:
        return 6

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "sequential_writes",
            "scatter_writes",
            "scattered_region",
            "distinct_outputs",
            "dense_outputs",
            "dense_loops",
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
            features[0, 4] += size(order)
        if features[0, 1]:
            fused = 0
            while (
                fused < len(prefix)
                and prefix[fused : fused + 1] == output_vars[fused : fused + 1]
            ):
                fused += 1
            features[0, 2] = size(output_vars[fused:])
        features[0, 5] = size(prefix)
        return features


def written_fields(component: Prefix) -> tuple[Field, ...]:
    """The fields of the output of the query of component, in the order it
    stores them, which are those of its expression without output_vars."""
    if component.output_vars is None:
        return component.expr.fields()
    return component.output_vars


N_WRITE_CLASSES = 8


@dataclass(frozen=True)
class WriteClasses:
    """
    The class of each row of WriteBasis, as four times whether the region
    its writes scatter into, plus two times whether the density of the levels
    of the output it reaches, its distinct outputs over its dense ones, plus
    whether the density of its loops, its writes over its dense loops,
    exceeds its cutoff. The default region cutoff is the most entries a
    workspace can reset and copy out cheaply.
    """

    region: float = 100_000
    output_density: float = 0.5
    loop_density: float = 0.5

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        sequential, scattered, region, distinct, dense_outputs, dense_loops = rows.T
        output_density = np.divide(
            distinct, dense_outputs, out=np.zeros(len(rows)), where=dense_outputs > 0
        )
        loop_density = np.divide(
            sequential + scattered,
            dense_loops,
            out=np.zeros(len(rows)),
            where=dense_loops > 0,
        )
        return (
            4 * (region > self.region)
            + 2 * (output_density > self.output_density)
            + (loop_density > self.loop_density)
        )


write_class = WriteClasses()
WRITE_BASIS = Stratify(WriteBasis(), N_WRITE_CLASSES, write_class)


def write_coefficients(scatter: float, workspace: float) -> list[float]:
    """The coefficients of WRITE_BASIS which charge a write which scatters
    workspace in the classes whose scattered region is within its cutoff,
    scatter in the rest, and nothing else."""
    return [
        cost
        for cls in range(N_WRITE_CLASSES)
        for cost in (0.0, scatter if cls >= 4 else workspace, 0.0, 0.0, 0.0, 0.0)
    ]


# Allocations are weighted 10x reads and operations, a lookup costs as much as
# an iteration, dense iterations are free until the model is fitted, and a
# write which scatters costs 10 iterations more than one in order, as it does
# into a hash level, which most sparse outputs need for it.
LOOP_COST_MODEL = CostModel(
    Concat(PigeonBasis(), TransposeBasis(), WRITE_BASIS),
    [1.0, 1.0, 10.0, 1.0, 0.0, 0.5, 60.0, *write_coefficients(10.0, 10.0)],
)
# A small workspace takes the writes of a query which scatter, as a dense level
# or a byte map does, almost as cheaply as those in order. Otherwise, they cost
# as much as they do without workspaces.
WORKSPACE_LOOP_COST_MODEL = CostModel(
    Concat(PigeonBasis(), TransposeBasis(), WRITE_BASIS),
    [1.0, 1.0, 10.0, 1.0, 0.0, 0.5, 60.0, *write_coefficients(10.0, 1.0)],
)
