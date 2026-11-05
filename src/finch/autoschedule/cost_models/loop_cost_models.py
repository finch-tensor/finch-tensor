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
from finch.symbolic import PostOrderDFS, Term

from .cost_models import NS, Basis, Concat, CostModel


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


class LoopSpaces(Generic[NS]):
    """
    The spaces of the last loop of a nonempty loop prefix: where it is
    entered, what it visits, and where its walks step, from the stats of the
    tables it merges, restricted as their formats and protocols allow, as
    unrestricted says. A table whose formats the prefix doesn't list is taken
    to be sparse, and a table without protocols walks the field of the loop.
    """

    def __init__(self, component: Prefix, machine: StatsMachine[NS]):
        self.machine = machine
        self.stats_factory = machine.stats_factory
        self.parent = set(component.prefix[:-1])
        self.idx = component.prefix[-1]
        match component.expr:
            case Aggregate(arg=arg):
                self.arg = arg
            case expr:
                self.arg = expr
        # The extent of each field, from the tables which read it, and the
        # field of each level of each and whether it doesn't filter, outermost
        # first, where the formats list level options.
        self.dims: dict[Field, float] = {}
        self.levels: dict[Table, list[tuple[Field, bool]]] = {}
        formats = component.formats or {}
        for node in PostOrderDFS(self.arg):
            match node:
                case Table(Alias() as tns, idxs) if idxs:
                    self.dims.update(self.machine(node).dim_sizes)
                    match formats.get(tns.unfused):
                        case [*options]:
                            self.levels[node] = [
                                (idxs[tns.nfused + depth], not option.filters)
                                for depth, option in enumerate(options)
                                if tns.nfused + depth < len(idxs)
                            ]
        # The loop is entered wherever the formats let the enclosing loops go,
        # and its merge steps only where every fiber it merges has positions
        # at idx.
        self.exec_space = self.stats_factory.support(self.merged(self.arg))
        self.merge_space = self.project(self.merged(self.arg, (self.idx,)))

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

    def unrestricted(self, table: Table, fields) -> set[Field]:
        """
        The fields of fields which table doesn't restrict: the field of the
        loop where table follows it, since it is looked up wherever the others
        go, and those of the dense levels down to the deepest level of the
        rest, from the sparse level above them, which have a fiber at every
        slot of it, so the positions of the level are the entries of the
        sparse one, at every slot of the dense ones.
        """
        unrestricted = set()
        if self.idx in fields and self.protocol(table) == Follow():
            unrestricted.add(self.idx)
        run: set[Field] = set()
        trailing: set[Field] = set()
        for idx, dense in self.levels.get(table, ()):
            run = run | {idx} if dense else set()
            if idx in fields and idx not in unrestricted:
                trailing = run
        return unrestricted | trailing

    def merged(self, node: LogicExpression, fields=()) -> NS:
        """
        Where a merge of the inputs of node visits parent and fields: where
        every fiber it merges has positions, as the formats give them.
        """
        # A merge coiterates the projections of the inputs, so it visits more
        # of parent than the projection of their result.
        followed: set[Field] = set()
        match node:
            case MapJoin(Literal(op), args):
                return self.stats_factory.mapjoin(
                    op, *(self.merged(arg, fields) for arg in args)
                )
            case Table():
                followed = self.unrestricted(node, {*self.parent, *fields})
        stat = self.machine(node)
        projected = self.project(stat, fields, followed)
        # The fill decides whether a mapjoin joins or unions it.
        projected.fill_type = stat.fill_type
        return projected

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
        return self.stats_factory.mapjoin(
            ffuncs.and_,
            self.exec_space,
            self.stats_factory.support(self.merged(self.arg, (self.idx,))),
        )

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
        return self.stats_factory.mapjoin(
            ffuncs.and_,
            self.merge_space,
            self.project(self.machine(table), (self.idx,)),
        )

    def walk(self, table: Table) -> float:
        return self.count(self.stepped(table))


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
        spaces = LoopSpaces(component, machine)
        idx = spaces.idx
        exec_space = spaces.exec_space
        inputs, walks, lookups = [], [], 0
        for node in PostOrderDFS(spaces.arg):
            match node:
                case Table(_, idxs) if idx in idxs:
                    stat = machine(node)
                    inputs.append(stat)
                    match spaces.protocol(node):
                        case Follow() | Gallop() | DefaultProtocol():
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


class WriteBasis(LoopBasis, LoopOrderedForm):
    """
    Count the writes to each level of the output of a query, at the loop which
    reaches that level: the last loop over the fields of the levels from the
    root to it. Where the loops up to it are those fields in storage order, the
    level is written once for each of its indices, in order, and otherwise its
    writes scatter, as util.write_protocols infers. The element level is written
    with its parent.

    A workspace fused over the leading fields of the output which the loops
    visit in storage order can take the writes which scatter, and stores one
    slice of the rest of the output. The writes which scatter are counted
    apart when that slice has at most workspace_limit entries, since only a
    workspace so small can be reset and copied out cheaply.
    """

    def __init__(self, workspace_limit: float = 100_000):
        self.workspace_limit = workspace_limit

    @property
    def n_features(self) -> int:
        return 3

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("sequential_writes", "scatter_writes", "workspace_writes")

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        prefix, expr = component.prefix, component.expr
        # Without output_vars, the output stores the fields of expr in order.
        output_vars = component.output_vars
        if output_vars is None:
            output_vars = expr.fields()
        features = np.zeros((1, self.n_features))
        if not prefix or not output_vars:
            return features
        visits = None
        for depth in range(len(output_vars)):
            order = output_vars[: depth + 1]
            # The level is reached by this prefix only if its last loop ends it.
            if not set(order) <= set(prefix) or prefix[-1] not in order:
                continue
            if visits is None:
                spaces = LoopSpaces(component, machine)
                visits = spaces.count(spaces.visited())
            if prefix == order:
                features[0, 0] += visits
            elif self.workspace_size(prefix, output_vars, expr, machine) <= (
                self.workspace_limit
            ):
                features[0, 2] += visits
            else:
                features[0, 1] += visits
        return features

    @staticmethod
    def workspace_size(prefix, output_vars, expr, machine) -> float:
        """
        The entries of a slice of a workspace fused over the leading fields of
        output_vars which prefix visits in order: the product of the dimensions
        of the rest. A field which no table has, such as a unit dimension of the
        output alone, has one index.
        """
        fused = 0
        while (
            fused < len(prefix)
            and prefix[fused : fused + 1] == output_vars[fused : fused + 1]
        ):
            fused += 1
        dims = {}
        for node in PostOrderDFS(expr):
            match node:
                case Table():
                    dims.update(machine(node).dim_sizes)
        return float(np.prod([dims.get(idx, 1) for idx in output_vars[fused:]]))


# Allocations are weighted 10x reads and operations, a lookup costs as much as
# an iteration, dense iterations are free until the model is fitted, and a
# write which scatters costs 10 iterations more than one in order, as it does
# into a hash level, which most sparse outputs need for it.
LOOP_COST_MODEL = CostModel(
    Concat(PigeonBasis(), TransposeBasis(), WriteBasis()),
    [1.0, 1.0, 10.0, 1.0, 0.0, 0.5, 60.0, 0.0, 10.0, 10.0],
)
# A small workspace takes the writes of a query which scatter, as a dense level
# or a byte map does, almost as cheaply as those in order. Without one, they
# cost as much as they do without workspaces.
WORKSPACE_LOOP_COST_MODEL = CostModel(
    Concat(PigeonBasis(), TransposeBasis(), WriteBasis()),
    [1.0, 1.0, 10.0, 1.0, 0.0, 0.5, 60.0, 0.0, 10.0, 1.0],
)
