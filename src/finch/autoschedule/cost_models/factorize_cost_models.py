from abc import abstractmethod
from collections.abc import Callable, Mapping

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import AliasedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicExpression,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Relabel,
    Reorder,
    StatsFactory,
    Table,
)

from .cost_models import NS, CostModel, LinearPredictor


class FactorizeCostModel(CostModel):
    """
    A cost model where each query statement is a component.
    get_statement_features returns one feature vector for a Query or QueryInto,
    given the stats of the aliases defined before it; get_features stacks these
    vectors in program order.
    """

    @abstractmethod
    def get_statement_features(
        self,
        stmt: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray: ...

    def predict_statement_cost(
        self,
        stmt: LogicStatement,
        machine: StatsMachine[NS],
    ) -> float:
        features = self.get_statement_features(stmt, machine)
        return self.predictor(features[None, :])

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        features = []
        stats_interpreter = StatsMachine(stats_factory, dict(stats))

        def visit_stmt(stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                case Query() | QueryInto():
                    features.append(
                        self.get_statement_features(stmt, stats_interpreter)
                    )
                    stats_interpreter(stmt)
                case Produces():
                    return
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, self.n_features)


class FlopsCostModel(FactorizeCostModel, AliasedForm):
    """
    FlopsCostModel counts the non-fill values in the iteration space of each
    query, and in the output it materializes. Each query is a component.
    By default, allocations are weighted 10x iterations.
    """

    @property
    def n_features(self) -> int:
        return 2

    def __init__(self, predictor: Callable[[np.ndarray], float] | None = None):
        super().__init__(
            predictor
            if predictor is not None
            else LinearPredictor(np.array([1.0, 10.0]))
        )

    def get_statement_features(
        self,
        stmt: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        stats_factory = machine.stats_factory

        def visit_expr(expr: LogicExpression) -> tuple[NS, NS]:
            """Return the stats of the output of expr, and of its iteration space."""
            match expr:
                case Aggregate(Literal(op), Literal(init), arg, idxs):
                    iter_stats, _ = visit_expr(arg)
                    out_stats = stats_factory.aggregate(op, init, idxs, iter_stats)
                    return out_stats, iter_stats
                case Aggregate(Literal(op), init, arg, idxs):
                    # A tensor init is folded into the reduction.
                    iter_stats, _ = visit_expr(arg)
                    init_stats, _ = visit_expr(init)
                    out_stats = stats_factory.aggregate(op, None, idxs, iter_stats)
                    out_stats = stats_factory.mapjoin(op, init_stats, out_stats)
                    return out_stats, iter_stats
                case MapJoin(Literal(op), (arg,)):
                    return visit_expr(arg)
                case MapJoin(Literal(op), args):
                    out_stats = stats_factory.mapjoin(
                        op, *(visit_expr(arg)[0] for arg in args)
                    )
                    return out_stats, out_stats
                case Reorder(arg, idxs):
                    out_stats, iter_stats = visit_expr(arg)
                    return stats_factory.reorder(out_stats, idxs), iter_stats
                case Relabel(arg, idxs):
                    out_stats, iter_stats = visit_expr(arg)
                    return stats_factory.relabel(out_stats, idxs), iter_stats
                case Table() | Literal():
                    out_stats = machine(expr)
                    return out_stats, out_stats
                case _:
                    raise ValueError(f"Unsupported expression type: {expr}")

        match stmt:
            case Query(Table(Alias()), rhs):
                out_stats, iter_stats = visit_expr(rhs)
                return np.array(
                    [
                        iter_stats.estimate_non_fill_values(),
                        out_stats.estimate_non_fill_values(),
                    ],
                    dtype=float,
                )
            case QueryInto():
                return self.get_statement_features(stmt.as_query(), machine)
            case _:
                raise ValueError(f"Unsupported statement type: {stmt}")
