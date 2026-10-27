from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from finch.algebra import ffuncs
from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Fuse,
    Literal,
    LogicExpression,
    LogicStatement,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
)
from finch.symbolic import PostOrderDFS

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


class PigeonBasis(LoopBasis, LoopOrderedForm):
    """
    PigeonBasis counts iteration, operation, and output costs across loop
    prefixes.
    """

    @property
    def n_features(self) -> int:
        return 3

    def get_component_features(
        self,
        component: Prefix,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """Return the incremental features for an outermost-first loop prefix.

        A nonempty prefix charges its last index, projecting execution and
        output onto prefix[:-1], as in the inside-out calculation. The empty
        prefix charges the output's non-fill values once. Summing all prefixes,
        including the empty one, gives the query's features.
        """
        prefix, expr, output_vars = (
            component.prefix,
            component.expr,
            component.output_vars,
        )
        stats_factory = machine.stats_factory
        match expr:
            case Aggregate(arg=arg):
                pass
            case _:
                arg = expr
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
            return np.array([[0.0, 0.0, out_space.estimate_non_fill_values()]])

        def project(stat: NS) -> NS:
            reduce_idxs = tuple(idx for idx in stat.index_order if idx not in parent)
            return stats_factory.aggregate(
                ffuncs.or_, False, reduce_idxs, stats_factory.support(stat)
            )

        parent = set(prefix[:-1])
        idx = prefix[-1]
        exec_space = project(machine(arg))
        inputs = []
        for node in PostOrderDFS(arg):
            match node:
                case Table(_, idxs) if idx in idxs:
                    inputs.append(stats_factory.support(machine(node)))
        iter_space = exec_space
        if inputs:
            iter_space = stats_factory.mapjoin(
                ffuncs.and_, exec_space, stats_factory.mapjoin(ffuncs.or_, *inputs)
            )
        out_cost = 0.0
        if idx in out_space.index_order:
            out_cost = project(out_space).estimate_non_fill_values()
        return np.array(
            [
                [
                    iter_space.estimate_non_fill_values(),
                    exec_space.estimate_non_fill_values() * len(inputs),
                    out_cost,
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


# Allocations are weighted 10x reads and operations.
LOOP_COST_MODEL = LinearCostModel(
    Concat(PigeonBasis(), TransposeBasis()), [1.0, 1.0, 10.0, 0.5, 60.0]
)
