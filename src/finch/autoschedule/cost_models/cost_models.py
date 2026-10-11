from abc import abstractmethod
from collections.abc import Mapping
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


class MultiCostModel(CostModel):
    """A cost model that sums the costs of its submodels."""

    def __init__(self, *submodels: CostModel):
        self.submodels = submodels

    def validate_inputs(self, *inputs):
        for model in self.submodels:
            model.validate_inputs(*inputs)

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> np.ndarray:
        return np.concatenate(
            [
                model.get_features(term, bindings, stats, stats_factory)
                for model in self.submodels
            ]
        )

    @property
    def coefficients(self) -> np.ndarray:
        return np.concatenate([model.coefficients for model in self.submodels])


class LoopCostModel(CostModel):
    """A cost model with features for individual loop prefixes."""

    @abstractmethod
    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray: ...

    def predict_prefix_cost(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> float:
        features = self.get_prefix_features(prefix, expr, machine, output_vars)
        return float(np.dot(self.coefficients, features))

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
                case Query(Table(_, lhs_idxs), Table() as out):
                    idxs = lhs_idxs
                case Query(
                    Table(_, lhs_idxs),
                    Aggregate(_, _, Reorder(_, idxs), _) as out,
                ):
                    pass
                case QueryInto(Table(_, lhs_idxs), _, Reorder(_, idxs)):
                    out = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            for n in range(len(idxs) + 1):
                features += self.get_prefix_features(
                    idxs[:n], out, stats_interpreter, lhs_idxs
                )
            stats_interpreter(stmt)

        visit_stmt(term)
        return features


class MultiLoopCostModel(LoopCostModel, MultiCostModel):
    """A loop cost model that sums the costs of its submodels."""

    def __init__(self, *submodels: LoopCostModel):
        self.submodels: tuple[LoopCostModel, ...] = submodels

    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray:
        return np.concatenate(
            [
                model.get_prefix_features(prefix, expr, machine, output_vars)
                for model in self.submodels
            ]
        )

class LevelCostModel(CostModel):
    """A cost model for a specific level of tensor formats."""

    @property
    @abstractmethod
    def coefficients(self):
        ...


class LevelOption(ABC):
    level_type: type

    @abstractmethod
    def build(self, lvl, dim_type): ...

    @abstractmethod
    def next_num_pos(self, num_pos, n_l, nnz_l): ...
    @abstractmethod
    def cost_fn(self, num_pos, n_l, nnz_l, val_size, pos_size): ...


class DenseLevelOption(LevelOption):
    level_type = DenseLevelFType

    def build(self, lvl, dim_type):
        return dense(lvl, dim_type)

    def next_num_pos(self, num_pos, n_l, nnz_l):
        return num_pos * n_l


class SparseHashLevelOption(LevelOption):
    level_type = SparseHashLevelFType

    def build(self, lvl, dim_type):
        return sparse_hash(lvl, dim_type, single_writer=True)

    def next_num_pos(self, num_pos, n_l, nnz_l):
        return nnz_l


class IterCostDenseLevelOption(DenseLevelOption):
    def cost_fn(self, num_pos, n_l, nnz_l, val_size, pos_size):
        return num_pos * n_l


class StorageCostDenseLevelOption(DenseLevelOption):
    def cost_fn(self, num_pos, n_l, nnz_l, val_size, pos_size):
        return 0.0


class IterCostSparseHashLevelOption(SparseHashLevelOption):
    def cost_fn(self, num_pos, n_l, nnz_l, val_size, pos_size):
        return num_pos + nnz_l


class StorageCostSparseHashLevelOption(SparseHashLevelOption):
    def cost_fn(self, num_pos, n_l, nnz_l, val_size, pos_size):
        return (num_pos + 1) * pos_size + nnz_l * pos_size

class FormatCostModel(CostModel):
    """A cost model with features for tensor format prefixes."""

    def __init__(self, formats:dict[str, LevelCostModel]):
        self.formats = formats

    @property
    def coefficients(self) -> np.ndarray:
        return np.concatenate([format.coefficients for format in self.formats.values()])

    @abstractmethod
    def get_format_features(
        self,
        alias: HardAlias,
        prefix: tuple[str, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray: ...

    def predict_format_cost(
        self,
        alias: HardAlias,
        prefix: tuple[str, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> float:
        features = self.get_format_features(alias, prefix, expr, machine, output_vars)
        return float(np.dot(self.coefficients, features))

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, list[str]],
        stats: Mapping[Alias, NumericStats],
        stats_factory: StatsFactory[NumericStats],
    ) -> np.ndarray:
        stats = dict(stats)
        features = {key: format.coefficients for key, format in self.formats.items()}
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
                case Query(Table(_, lhs_idxs), Table() as out):
                    idxs = lhs_idxs
                case Query(
                    Table(_, lhs_idxs),
                    Aggregate(_, _, Reorder(_, idxs), _) as out,
                ):
                    pass
                case QueryInto(Table(_, lhs_idxs), _, Reorder(_, idxs)):
                    out = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            for n in range(len(idxs) + 1):
                features += self.get_format_features(
                    idxs[:n], out, stats_interpreter, lhs_idxs
                )
            stats_interpreter(stmt)

        visit_stmt(term)
        return features


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
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray:
        """Return the incremental features for an outermost-first loop prefix.

        A nonempty prefix charges its last index, projecting execution and
        output onto prefix[:-1], as in the inside-out calculation. The empty
        prefix charges the output's non-fill values once. Summing all prefixes,
        including the empty one, gives the query's features.
        """
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
                iter_space.estimate_non_fill_values(),
                exec_space.estimate_non_fill_values() * len(inputs),
                out_cost,
            ]
        )


class TransposeCostModel(LoopCostModel, LoopOrderedForm):
    """
    Count dense and sparse transposed entries, charging each input once per query.
    """

    def __init__(self):
        pass

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([0.5, 60.0])

    def get_prefix_features(
        self,
        prefix: tuple[Field, ...],
        expr: LogicExpression,
        machine: StatsMachine[NS],
        output_vars: tuple[Field, ...] | None = None,
    ) -> np.ndarray:
        """Charge only the input transpositions newly forced by this prefix.

        Deduplicate input tables within this prefix. Comparing with the parent
        prefix avoids charging them again at deeper loop levels. Empty prefixes
        have no transposition cost.
        """
        features = np.zeros(len(self.coefficients))
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
                features[0 if dense else 1] += nnz
        return features
