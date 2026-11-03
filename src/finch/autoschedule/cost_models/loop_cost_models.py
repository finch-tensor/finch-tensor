from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Generic

import numpy as np

from finch.algebra import ffuncs
from finch.algebra.tensor import TensorFType
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
    output_vars orders the output of the query, when there is one.
    """

    prefix: tuple[Field, ...]
    expr: LogicExpression
    output_vars: tuple[Field, ...] | None = None


class LoopBasis(Basis[Prefix]):
    """
    A basis where each loop prefix, including the empty prefix, is a component.
    components yields prefixes in query order, from the shortest prefix to the
    longest, and get_component_features returns one row for each prefix. The
    loop order of a query is the fields of its enclosing Fuse nodes. Models
    used with branch-and-bound loop search must predict nonnegative component
    costs.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, TensorFType],
        machine: StatsMachine[NS],
        loop_order: tuple[Field, ...] = (),
    ) -> Iterator[Prefix]:
        match term:
            case Plan(bodies):
                for body in bodies:
                    yield from self.components(body, bindings, machine, loop_order)
                return
            case Fuse(idx, body):
                loop_order = (*loop_order, idx)
                yield from self.components(body, bindings, machine, loop_order)
                return
            case QueryInto(Table(), _, Literal()):
                machine(term)
                return
            case Produces():
                return
            case Query(Table(_, lhs_idxs), expr):
                pass
            case QueryInto(Table(_, lhs_idxs), _, _):
                expr = term.as_query().rhs
            case _:
                raise ValueError(f"Unsupported statement type: {term}")

        for n in range(len(loop_order) + 1):
            yield Prefix(loop_order[:n], expr, lhs_idxs)
        machine(term)


class LoopSpaces(Generic[NS]):
    """
    The spaces of the last loop of a nonempty loop prefix. Execution is where a
    merge of the inputs, which can't look ahead, visits the enclosing loops. A
    table walks the field of the loop unless it gallops or follows it, as its
    protocol for that field says.
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
        self.exec_space = self.stats_factory.support(self.merged(self.arg))

    def project(self, stat: NS, fields=()) -> NS:
        fields = {*self.parent, *fields}
        reduce_idxs = tuple(idx for idx in stat.index_order if idx not in fields)
        return self.stats_factory.aggregate(
            ffuncs.or_, False, reduce_idxs, self.stats_factory.support(stat)
        )

    def protocol(self, node: Term) -> Protocol:
        match node:
            case Table(_, idxs, (_, *_) as protocols) if self.idx in idxs:
                return protocols[idxs.index(self.idx)]
        return DefaultProtocol()

    def merged(self, node: LogicExpression, fields=()) -> NS:
        # A merge coiterates the projections of the inputs, so it visits more
        # of parent than the projection of their result.
        match node:
            case MapJoin(Literal(op), args):
                return self.stats_factory.mapjoin(
                    op, *(self.merged(arg, fields) for arg in args)
                )
            case _:
                stat = self.machine(node)
                # A followed table is looked up wherever the others go.
                followed = any(
                    self.protocol(table) == Follow() for table in PostOrderDFS(node)
                )
                projected = self.project(stat, () if followed else fields)
                # The fill decides whether a mapjoin joins or unions it.
                projected.fill_type = stat.fill_type
                return projected

    def count(self, stat: NS) -> float:
        # A space without idx spans all of it.
        if self.idx in stat.index_order:
            return stat.estimate_non_fill_values()
        for node in PostOrderDFS(self.arg):
            match node:
                case Table(_, idxs) if self.idx in idxs:
                    dim = self.machine(node).dim_sizes[self.idx]
                    return stat.estimate_non_fill_values() * dim
        return stat.estimate_non_fill_values()

    def visited(self) -> NS:
        """Where the loop visits idx, which its walks and gallops restrict."""
        return self.stats_factory.mapjoin(
            ffuncs.and_,
            self.exec_space,
            self.stats_factory.support(self.merged(self.arg, (self.idx,))),
        )

    def walk(self, table: Table) -> float:
        """The steps of a walk of table through idx."""
        return self.count(
            self.stats_factory.mapjoin(
                ffuncs.and_,
                self.exec_space,
                self.project(self.machine(table), (self.idx,)),
            )
        )


class PigeonBasis(LoopBasis, LoopOrderedForm):
    """
    PigeonBasis counts iteration, operation, output, and lookup costs across
    loop prefixes. Each table reads the field of a loop with its protocol for
    that field, and a table without protocols walks it.
    """

    @property
    def n_features(self) -> int:
        return 4

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("iterations", "operations", "outputs", "lookups")

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
            return np.array([[0.0, 0.0, out_space.estimate_non_fill_values(), 0.0]])
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
        iterations = exec_space.estimate_non_fill_values()
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
            iterations = stats_factory.mapjoin(
                ffuncs.and_, exec_space, stats_factory.mapjoin(ffuncs.or_, *walks)
            ).estimate_non_fill_values()
        out_cost = 0.0
        if idx in out_space.index_order:
            out_cost = spaces.project(out_space).estimate_non_fill_values()
        return np.array(
            [
                [
                    iterations,
                    exec_space.estimate_non_fill_values() * len(inputs),
                    out_cost,
                    lookup_cost,
                ]
            ]
        )


class TransposeBasis(LoopBasis, LoopOrderedForm):
    """
    Count dense and sparse transposed entries, charging each input once per query.
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


# Allocations are weighted 10x reads and operations, and a lookup costs as
# much as an iteration.
LOOP_COST_MODEL = LinearCostModel(
    Concat(PigeonBasis(), TransposeBasis()), [1.0, 1.0, 10.0, 1.0, 0.5, 60.0]
)
