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
    Table,
)
from finch.symbolic import PostOrderDFS, Term

from .cost_models import NS, Basis, Concat, LinearCostModel


@dataclass(frozen=True)
class Prefix:
    """
    An outermost-first loop prefix of expr, the right-hand side of a query.
    output_vars orders the output of the query, when there is one, and
    formats, which bases that price formatted tensors need, lists the level
    options each alias stores, outermost first, as FormatBasis takes them.
    executions, when given, estimates how many times the enclosing loops run,
    as the protocol optimizer computes it from its choices at them, and the
    counts of the last loop are scaled to it. output, when given, is the tensor
    the query writes, with the fields of output_vars.
    """

    prefix: tuple[Field, ...]
    expr: LogicExpression
    output_vars: tuple[Field, ...] | None = None
    formats: Mapping[Alias, Any] | None = field(default=None, compare=False, hash=False)
    executions: float | None = None
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
    its enclosing Fuse nodes. Each prefix carries formats, and the alias its
    query writes.
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
    The spaces of the last loop of a nonempty loop prefix. Execution is where a
    merge of the inputs, which can't look ahead, visits the enclosing loops.
    The formats of the prefix decide where the loop is entered: a sparse level
    has a fiber only where it has an entry, and a lookup which finds none runs
    nothing within it, so a table restricts the enclosing loops to its support
    over the fields of its sparse levels, whatever protocols it reads them
    with, while a dense level has a fiber at every slot, so its field never
    restricts, and a table whose formats the prefix doesn't list is taken to
    be sparse. A table walks the field of the loop unless it gallops or
    follows it, as its protocol for that field says, and a followed table is
    looked up wherever the others go, so it doesn't restrict the loop, and a
    loop whose tables all follow its field spans all of it. The counts are
    taken over the executions of the enclosing loops which the stats
    estimate, and are scaled to the executions the prefix gives, when it
    does, as the level basis gives them.
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
        # fields of the dense levels of each, where the formats list levels.
        self.dims: dict[Field, float] = {}
        self.dense: dict[Table, set[Field]] = {}
        formats = component.formats or {}
        for node in PostOrderDFS(self.arg):
            match node:
                case Table(Alias() as tns, idxs) if idxs:
                    self.dims.update(self.machine(node).dim_sizes)
                    match formats.get(tns.unfused):
                        case [*options]:
                            self.dense[node] = {
                                idxs[tns.nfused + depth]
                                for depth, option in enumerate(options)
                                if tns.nfused + depth < len(idxs)
                                and getattr(option, "kind", None) == "dense"
                            }
        self.exec_space = self.stats_factory.support(self.merged(self.arg))
        self.scale = 1.0
        if component.executions is not None:
            estimated = self.extent(self.exec_space, self.parent)
            if estimated:
                self.scale = component.executions / estimated
        self._executions = component.executions

    def project(self, stat: NS, fields=(), followed=()) -> NS:
        """
        The support of stat over parent and fields, less the fields of
        followed, which it spans wholly.
        """
        fields = {*self.parent, *fields} - {*followed}
        reduce_idxs = tuple(idx for idx in stat.index_order if idx not in fields)
        return self.stats_factory.aggregate(
            ffuncs.or_, False, reduce_idxs, self.stats_factory.support(stat)
        )

    def protocol(self, node: Term, idx: Field | None = None) -> Protocol:
        """The protocol node reads idx with, by default the field of the loop."""
        if idx is None:
            idx = self.idx
        match node:
            case Table(_, idxs, (_, *_) as protocols) if idx in idxs:
                return protocols[idxs.index(idx)]
        return DefaultProtocol()

    def unrestricted(self, table: Table, fields) -> set[Field]:
        """
        The fields of fields which table doesn't restrict: those of its dense
        levels, which have a fiber at every slot, and the field of the loop
        where table follows it, since it is looked up wherever the others go.
        """
        unrestricted = set(self.dense.get(table, ()))
        if self.idx in fields and self.protocol(table) == Follow():
            unrestricted.add(self.idx)
        return unrestricted

    def merged(self, node: LogicExpression, fields=()) -> NS:
        # A merge coiterates the projections of the inputs, so it visits more
        # of parent than the projection of their result.
        match node:
            case MapJoin(Literal(op), args):
                return self.stats_factory.mapjoin(
                    op, *(self.merged(arg, fields) for arg in args)
                )
            case Table():
                stat = self.machine(node)
                projected = self.project(
                    stat, fields, self.unrestricted(node, {*self.parent, *fields})
                )
                # The fill decides whether a mapjoin joins or unions it.
                projected.fill_type = stat.fill_type
                return projected
            case _:
                stat = self.machine(node)
                projected = self.project(stat, fields)
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
        """The points of stat over the loops of the prefix."""
        return self.extent(stat, (*self.parent, self.idx)) * self.scale

    def executions(self) -> float:
        """The points the enclosing loops visit."""
        if self._executions is not None:
            return self._executions
        return self.extent(self.exec_space, self.parent)

    def iterations(self) -> float:
        """The points the loop visits, at each of which its body runs."""
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
        those fields which the loop reaches in stat. They aren't scaled to the
        executions of the prefix, since a loop which runs more often reaches
        the same positions.
        """
        reduce_idxs = tuple(idx for idx in stat.index_order if idx not in fields)
        if reduce_idxs:
            stat = self.stats_factory.aggregate(
                ffuncs.or_, False, reduce_idxs, self.stats_factory.support(stat)
            )
        return self.extent(stat, fields)

    def stepped(self, table: Table) -> NS:
        """The entries of table at idx which the enclosing loops reach."""
        return self.stats_factory.mapjoin(
            ffuncs.and_,
            self.exec_space,
            self.project(self.machine(table), (self.idx,)),
        )

    def walk(self, table: Table) -> float:
        """The steps of a walk of table through idx."""
        return self.count(self.stepped(table))


class PigeonBasis(LoopBasis, LoopOrderedForm):
    """
    PigeonBasis counts iteration, operation, output, lookup, and dense
    iteration costs across loop prefixes. Each table reads the field of a loop
    with its protocol for that field, and a table without protocols walks it.
    The dense iterations of a loop are those it would make if its tables were
    dense, so that none could restrict it: its whole extent, wherever its
    enclosing loops run.
    """

    class_key = "pigeon"

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
        """Return the incremental features for an outermost-first loop prefix.

        A nonempty prefix charges its last index, projecting execution and
        output onto prefix[:-1], as in the inside-out calculation. Execution is
        where a merge of the inputs, which can't look ahead, visits prefix[:-1].
        The empty prefix charges the output's non-fill values once. Summing all
        prefixes, including the empty one, gives the query's features.

        A walk steps through the stored indices of its table. A gallop or a
        follow instead visits the indices where the expression runs, which a
        follow doesn't restrict, and charges a lookup at each of them. A loop
        whose tables all follow its field visits all of it.
        """
        prefix, expr, output_vars = (
            component.prefix,
            component.expr,
            component.output_vars,
        )
        stats_factory = machine.stats_factory
        out_space = machine(expr)
        if output_vars is not None:
            reduce_idxs = tuple(
                idx for idx in out_space.index_order if idx not in output_vars
            )
            if reduce_idxs:
                out_space = stats_factory.aggregate(
                    ffuncs.or_, False, reduce_idxs, stats_factory.support(out_space)
                )
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
    Count dense and sparse transposed entries, charging each input once per query.
    """

    class_key = "transpose"

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
        """Charge only the input transpositions newly forced by this prefix.

        Deduplicate input tables within this prefix. Comparing with the parent
        prefix avoids charging them again at deeper loop levels. Empty prefixes
        have no transposition cost.
        """
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


# Allocations are weighted 10x reads and operations, a lookup costs as much as
# an iteration, and dense iterations are free until the model is fitted.
LOOP_COST_MODEL = LinearCostModel(
    Concat(PigeonBasis(), TransposeBasis()), [1.0, 1.0, 10.0, 1.0, 0.0, 0.5, 60.0]
)
