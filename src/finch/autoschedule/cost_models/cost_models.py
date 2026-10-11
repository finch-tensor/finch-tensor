from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np

from finch.algebra import StaticFill, ffuncs
from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import AliasedForm, LoopOrderedForm
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
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
from finch.symbolic import Form, PostOrderDFS
from finch.tensor import Scalar

NS = TypeVar("NS", bound=NumericStats)


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


class LoopCostModel(CostModel):
    """A cost model with features for individual loop prefixes."""

    @abstractmethod
    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        exec_space: NS,
        out_space: NS,
        args: Mapping[Field, Sequence[NS]],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray: ...

    def predict_prefix_cost(
        self,
        prefix: tuple[Field, ...],
        exec_space: NS,
        out_space: NS,
        args: Mapping[Field, Sequence[NS]],
        stats_factory: StatsFactory[NS],
    ) -> float:
        features = self.get_prefix_features(
            prefix, exec_space, out_space, args, stats_factory
        )
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


class PigeonCostModel(LoopCostModel, LoopOrderedForm):
    """
    PigeonCostModel counts iteration, operation, and output costs across loop
    prefixes. Writes are weighted 10x reads and operations.
    """

    def __init__(self):
        pass

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([1.0, 1.0, 10.0])

    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        exec_space: NS,
        out_space: NS,
        args: Mapping[Field, Sequence[NS]],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        """Return the incremental features for an outermost-first loop prefix.

        The statistics are unprojected: exec_space describes the query's RHS,
        out_space describes its output in lhs order, and args groups input Table
        occurrences by field. They can be reused across candidate prefixes.

        A nonempty prefix charges its last index, projecting execution and
        output onto prefix[:-1], as in the inside-out calculation. The empty
        prefix charges the output's non-fill values once. Summing all prefixes,
        including the empty one, gives the query's features.
        """
        if not prefix:
            return np.array([0.0, 0.0, out_space.estimate_non_fill_values()])

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
        exec_space = project(exec_space)
        inputs = [support(stat) for stat in args.get(idx, ())]
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
                iter_space.estimate_non_fill_values(),
                exec_space.estimate_non_fill_values() * len(inputs),
                out_cost,
            ]
        )

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NumericStats],
        stats_factory: StatsFactory[NumericStats],
    ) -> np.ndarray:
        stats = dict(stats)
        features = np.zeros(len(self.coefficients))
        stats_interpreter = StatsMachine(stats_factory, stats)

        def visit_stmt(stmt: LogicStatement):
            nonlocal features
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                    return
                case QueryInto(Table(), _, Literal()):
                    stats_interpreter(stmt)
                    return
                case Produces():
                    return
                case Query(Table(Alias() as lhs, lhs_idxs), Table() as rhs):
                    idxs = lhs_idxs
                    out = rhs
                case Query(
                    Table(Alias() as lhs, lhs_idxs),
                    Aggregate(_, _, Reorder(_, idxs) as rhs, _) as out,
                ):
                    pass
                case QueryInto(
                    Table(Alias() as lhs, lhs_idxs), _, Reorder(_, idxs) as rhs
                ):
                    out = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            exec_space = stats_interpreter(rhs)
            args: dict[Field, list[NumericStats]] = {idx: [] for idx in idxs}
            for node in PostOrderDFS(rhs):
                match node:
                    case Table(_, arg_idxs):
                        arg_stats = stats_interpreter(node)
                        for idx in arg_idxs:
                            args[idx].append(arg_stats)
            out_space = stats_factory.reorder(stats_interpreter(out), lhs_idxs)
            stats[lhs] = out_space
            for n in range(len(idxs) + 1):
                features += self.get_prefix_features(
                    idxs[:n], exec_space, out_space, args, stats_factory
                )

        visit_stmt(term)
        return features
