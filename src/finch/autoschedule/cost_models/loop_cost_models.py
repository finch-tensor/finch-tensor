from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from finch.algebra import StaticFill, ffuncs
from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicStatement,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    Table,
)
from finch.symbolic import PostOrderDFS

from .cost_models import NS, CostModel, LinearPredictor


@dataclass(frozen=True)
class Prefix:
    """
    An outermost-first loop prefix of expr, the right-hand side of a query.
    output_vars orders the output of the query, when there is one.
    """

    prefix: tuple[Field, ...]
    expr: LogicExpression
    output_vars: tuple[Field, ...] | None = None


class LoopCostModel(CostModel[Prefix]):
    """
    A cost model where each loop prefix, including the empty prefix, is a component.
    components yields prefixes in query order, from the shortest prefix to the
    longest, and get_component_features returns one row for each prefix.
    Predictors used with branch-and-bound loop search must predict nonnegative
    component costs.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, TensorFType],
        machine: StatsMachine[NS],
    ) -> Iterator[Prefix]:
        match term:
            case Plan(bodies):
                for body in bodies:
                    yield from self.components(body, bindings, machine)
                return
            case QueryInto(Table(), _, Literal()):
                machine(term)
                return
            case Produces():
                return
            case Query(Table(_, lhs_idxs), Table() as expr):
                idxs = lhs_idxs
            case Query(
                Table(_, lhs_idxs),
                Aggregate(_, _, Reorder(_, idxs), _) as expr,
            ):
                pass
            case QueryInto(Table(_, lhs_idxs), _, Reorder(_, idxs)):
                expr = term.as_query().rhs
            case _:
                raise ValueError(f"Unsupported statement type: {term}")

        for n in range(len(idxs) + 1):
            yield Prefix(idxs[:n], expr, lhs_idxs)
        machine(term)


class PigeonCostModel(LoopCostModel, LoopOrderedForm):
    """
    PigeonCostModel counts iteration, operation, and output costs across loop
    prefixes. By default, allocations are weighted 10x reads and operations.
    """

    @property
    def n_features(self) -> int:
        return 3

    def __init__(self, predictor: Callable[[np.ndarray], float] | None = None):
        super().__init__(
            predictor
            if predictor is not None
            else LinearPredictor(np.array([1.0, 1.0, 10.0]))
        )

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
                out_space = stats_factory.copy(out_space)
                out_space.fill_value = StaticFill(False)
                out_space = stats_factory.aggregate(
                    ffuncs.or_, False, reduce_idxs, out_space
                )
            out_space = stats_factory.reorder(out_space, output_vars)
        if not prefix:
            return np.array([[0.0, 0.0, out_space.estimate_non_fill_values()]])

        def support(stat: NS) -> NS:
            stat = stats_factory.copy(stat)
            stat.fill_value = StaticFill(False)
            return stat

        def project(stat: NS) -> NS:
            reduce_idxs = tuple(idx for idx in stat.index_order if idx not in parent)
            return stats_factory.aggregate(
                ffuncs.or_, False, reduce_idxs, support(stat)
            )

        parent = set(prefix[:-1])
        idx = prefix[-1]
        exec_space = project(machine(arg))
        inputs = []
        for node in PostOrderDFS(arg):
            match node:
                case Table(_, idxs) if idx in idxs:
                    inputs.append(support(machine(node)))
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


class TransposeCostModel(LoopCostModel, LoopOrderedForm):
    """
    Count dense and sparse transposed entries, charging each input once per query.
    """

    @property
    def n_features(self) -> int:
        return 2

    def __init__(self, predictor: Callable[[np.ndarray], float] | None = None):
        super().__init__(
            predictor
            if predictor is not None
            else LinearPredictor(np.array([0.5, 60.0]))
        )

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
