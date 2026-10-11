from abc import abstractmethod
from collections.abc import Mapping

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import AliasedForm
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
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
    TensorStats,
)
from finch.symbolic import Form
from finch.tensor import Scalar


class CostModel(Form):
    """
    A CostModel predicts the cost of a logic statement as a linear function
    of its features. Like a Stage, a CostModel inherits the Form of the
    statements it accepts.
    """

    @property
    @abstractmethod
    def coefficients(self) -> np.ndarray: ...

    @abstractmethod
    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> np.ndarray: ...

    def predict_cost(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> float:
        features = self.get_features(term, bindings, stats, stats_factory)
        return float(np.dot(self.coefficients, features))


class FlopsCostModel(CostModel, AliasedForm):
    """
    FlopsCostModel counts the non-fill values in the iteration space of each
    query, and in the output it materializes. Writes are weighted 10x reads.
    """

    def __init__(self):
        pass

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([1.0, 10.0])

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NumericStats],
        stats_factory: StatsFactory[NumericStats],
    ) -> np.ndarray:
        stats = dict(stats)
        features = np.zeros(len(self.coefficients))

        def visit_expr(expr: LogicExpression) -> tuple[NumericStats, NumericStats]:
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
                case Table(Alias() as tns, idxs):
                    out_stats = stats_factory.relabel(stats[tns], idxs)
                    return out_stats, out_stats
                case Table(Literal(tns), idxs):
                    out_stats = stats_factory(tns, idxs)
                    return out_stats, out_stats
                case Literal(val):
                    out_stats = stats_factory(Scalar(val), ())
                    return out_stats, out_stats
                case _:
                    raise ValueError(f"Unsupported expression type: {expr}")

        def visit_stmt(stmt: LogicStatement):
            nonlocal features
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                case Query(Table(Alias() as lhs, idxs), rhs):
                    out_stats, iter_stats = visit_expr(rhs)
                    stats[lhs] = stats_factory.reorder(out_stats, idxs)
                    features += [
                        iter_stats.estimate_non_fill_values(),
                        out_stats.estimate_non_fill_values(),
                    ]
                case QueryInto():
                    visit_stmt(stmt.as_query())
                case Produces():
                    return
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        visit_stmt(term)
        return features

class PigeonCostModel(CostModel, LoopOrderedForm):
    """
    PigeonCostModel counts the non-fill values in the iteration space of each
    query, and in the output it materializes. Writes are weighted 10x reads.
    """

    def __init__(self):
        pass

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([1.0, 10.0])

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NumericStats],
        stats_factory: StatsFactory[NumericStats],
    ) -> np.ndarray:
        stats = dict(stats)
        features = np.zeros(len(self.coefficients))

        def visit_stmt(stmt: LogicStatement):
            nonlocal features
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                    return
                case QueryInto(Table(lhs), _, Literal()):
                    return
                case Query(Table(lhs), Table(rhs))|
                 QueryInto(Table(lhs, lhs_idxs), _, Reorder(Table(rhs), idxs)):
                    init = literal(lhs.fill_value)
                case Query(
                    Table(lhs, lhs_idxs), Aggregate(_, init, Reorder(arg, idxs), _) as out
                ) | QueryInto(Table(_, lhs_idxs), init, Reorder(arg, idxs)):
                    exec_space = stats_interpreter(rhs)
                    args = {[] for idx in rhs.idxs}
                    for node in PostOrderDFS(rhs):
                        match node:
                            case Table(arg, idxs):
                                for idx in idxs:
                                    args[idx].append((arg, idxs))
                    out_space = stats_interpreter(out)
                    out_cost += out_space.estimate_non_fill_values()
                    op_cost = 0
                    iter_cost = 0
                    for n in range(len(rhs.idxs), 1, -1):
                        idx = rhs.idxs[n]
                        exec_space = stats_factory.aggregate(exec_space, [idx])
                        out_space = stats_factory.aggregate(out_space, [idx])
                        if idx in out_idxs:
                            out_cost += out_space.estimate_non_fill_values()
                        iter_space = stats_factory.mapjoin(ffuncs.and, exec_space, stats_factory.mapjoin(ffuncs.or, *args[idx]))
                        iter_cost += iter_space.estimate_non_fill_values()
                        op_cost += exec_space.estimate_non_fill_values() * len(args[idx])
                    features += [iter_cost, op_cost, out_cost]
                    return None
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        visit_stmt(term)
        return features