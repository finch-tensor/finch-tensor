from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import TypeVar

import numpy as np

from finch.algebra import StaticFill, ffuncs, ftype, np_dtype
from finch.algebra.tensor import TensorFType
from finch.algebra.utils import is_subsequence
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
from finch.tensor import Scalar, element

NS = TypeVar("NS", bound=NumericStats)


class LinearPredictor:
    def __init__(self, coefficients: np.ndarray):
        self.coefficients: np.ndarray = coefficients

    def __call__(self, features: np.ndarray) -> float:
        return float(features.sum(axis=0) @ self.coefficients)


class QuadraticPredictor:
    """Sum row-wise quadratic forms, with flattened row-major coefficients."""

    def __init__(self, coefficients: np.ndarray):
        self.coefficients: np.ndarray = coefficients

    def __call__(self, features: np.ndarray) -> float:
        x = np.einsum("ij,ik->ijk", features, features)
        return float(x.sum(axis=0).ravel() @ self.coefficients)


class CostModel(Form):
    """
    A cost model predicts the cost of a logic statement as a sum of modeled pieces:

        sum_i predictor(X_i1, ..., X_in)

    get_features returns X with shape (n_components, n_features). n_features is
    independent of the number of predictor parameters. Each CostModel defines its
    own components. The predictor callable takes the whole matrix and
    returns a scalar sum of independently modeled rows, including zero for an
    empty matrix. Like a Stage, a CostModel inherits the form of statements it accepts.
    """

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    def __init__(self, predictor: Callable[[np.ndarray], float]):
        self.predictor = predictor

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
        return self.predictor(features)


class MultiCostModel(CostModel):
    """
    A cost model that composes the observations of its submodels. Note that
    submodels must describe the same pieces in the same order. Their feature
    columns are concatenated. By default, predictions of the submodels are
    summed; a supplied predictor instead predicts from the combined features.
    """

    def __init__(
        self,
        *submodels: CostModel,
        predictor: Callable[[np.ndarray], float] | None = None,
    ):
        if not submodels:
            raise ValueError("MultiCostModel requires at least one submodel")
        self.submodels = submodels
        super().__init__(predictor if predictor is not None else self._predict)

    @property
    def n_features(self) -> int:
        return sum(submodel.n_features for submodel in self.submodels)

    def _predict(self, features: np.ndarray) -> float:
        cost = 0.0
        start = 0
        for submodel in self.submodels:
            stop = start + submodel.n_features
            cost += submodel.predictor(features[:, start:stop])
            start = stop
        return cost

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
            ],
            axis=1,
        )


class LoopCostModel(CostModel):
    """
    A cost model where each loop prefix, including the empty prefix, is a component.
    get_prefix_features returns one feature vector; get_features stacks these
    vectors in query order, from the shortest prefix to the longest. Predictors used
    with branch-and-bound loop search must predict nonnegative component costs.
    """

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
        return self.predictor(features[None, :])

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        stats = dict(stats)
        features = []
        stats_interpreter = StatsMachine(stats_factory, stats)

        def visit_stmt(stmt: LogicStatement):
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

            features.extend(
                self.get_prefix_features(idxs[:n], out, stats_interpreter, lhs_idxs)
                for n in range(len(idxs) + 1)
            )
            stats_interpreter(stmt)

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, self.n_features)


class MultiLoopCostModel(LoopCostModel, MultiCostModel):
    """Combine feature columns for matching loop prefixes."""

    def __init__(
        self,
        *submodels: LoopCostModel,
        predictor: Callable[[np.ndarray], float] | None = None,
    ):
        super().__init__(*submodels, predictor=predictor)
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


class LevelOption(ABC):
    """
    A level the formatter may choose for a dimension. Its coefficients weigh
    the level features of a FormatCostModel.
    """

    def __init__(self, coefficients):
        self._coefficients = np.array(coefficients, dtype=float)

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @abstractmethod
    def next_num_pos(self, num_pos, n_l, nnz_l): ...


class DenseLevelOption(LevelOption):
    def next_num_pos(self, num_pos, n_l, nnz_l):
        return num_pos * n_l


class SparseHashLevelOption(LevelOption):
    def next_num_pos(self, num_pos, n_l, nnz_l):
        return nnz_l


class FormatCostModel(CostModel, LoopOrderedForm):
    """
    Each tensor level contributes an allocation component and an access component
    for each query loop prefix that uses it. Each entry of formats names a level
    option, whose coefficients supply the default linear weights for that format.
    """

    def __init__(
        self,
        formats: Mapping[str, LevelOption],
        element_coefficients,
        predictor: Callable[[np.ndarray], float] | None = None,
    ):
        self.formats = formats
        self.element_coefficients = np.array(element_coefficients, dtype=float)
        coefficients = np.concatenate(
            [
                *(format.coefficients for format in self.formats.values()),
                self.element_coefficients,
            ]
        )
        super().__init__(
            predictor if predictor is not None else LinearPredictor(coefficients)
        )

    @property
    def n_features(self) -> int:
        return 10 * (len(self.formats) + 1)

    def nnz_after(self, stats: NS, stats_factory: StatsFactory[NS], level: int):
        """Return the non-fill entries of the first level + 1 levels of stats."""
        reduce_fields = tuple(stats.index_order[level + 1 :])
        if reduce_fields:
            stats = stats_factory.aggregate(ffuncs.or_, False, reduce_fields, stats)
        return stats.estimate_non_fill_values()

    def get_level_features(self, num_pos, n_l, nnz_l, val_size, pos_size):
        """
        Return the features of a level with num_pos positions entering it, a
        dimension of n_l, and nnz_l non-fill entries: positions, slots, entries,
        position bytes, index bytes, and value bytes.
        """
        return np.array(
            [
                num_pos,
                num_pos * n_l,
                nnz_l,
                (num_pos + 1) * pos_size,
                nnz_l * pos_size,
                num_pos * val_size,
            ]
        )

    def get_access_features(
        self,
        alias: Alias,
        level: int,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """
        Return the accesses to alias at a zero-based depth in each query of term:
        sequential reads, scatter reads, sequential writes, and scatter writes,
        with one row per loop prefix that accesses the level in each query.
        Each access is counted by the PigeonCostModel iterations of the loops
        from the root to the current level. The element level is accessed with
        its parent.
        """
        # A sequential write is when the prefix of loops from root to current
        # level matches the ordering of the access exactly.
        # A sequential read is when the prefix of loops from root to current
        # level is a supersequence of the ordering of the access.
        features = []
        stats_interpreter = StatsMachine(
            machine.stats_factory, OrderedDict(machine.bindings)
        )
        pigeon = PigeonCostModel()

        def visit_access(idxs, loop_order, rhs, lhs_idxs, is_write, prefixes):
            if not idxs:
                return
            depth = min(level, len(idxs) - 1)
            order = tuple(idxs[: depth + 1])
            loops = tuple(loop_order[: loop_order.index(idxs[depth]) + 1])
            iterations = pigeon.get_prefix_features(
                loops, rhs, stats_interpreter, lhs_idxs
            )[0]
            features = prefixes.setdefault(loops, np.zeros(4))
            if is_write:
                features[2 if loops == order else 3] += iterations
            else:
                features[0 if is_subsequence(order, loops) else 1] += iterations

        def visit_stmt(stmt: LogicStatement):
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
                case Query(Table(lhs, lhs_idxs), Table() as rhs):
                    arg = rhs
                    loop_order = lhs_idxs
                case Query(
                    Table(lhs, lhs_idxs),
                    Aggregate(_, _, Reorder(arg, loop_order), _) as rhs,
                ):
                    pass
                case QueryInto(Table(lhs, lhs_idxs), _, Reorder(arg, loop_order)):
                    rhs = stmt.as_query().rhs
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

            prefixes = {}
            for node in PostOrderDFS(arg):
                match node:
                    case Table(tns, idxs) if tns == alias:
                        visit_access(idxs, loop_order, rhs, lhs_idxs, False, prefixes)
            if lhs == alias:
                visit_access(lhs_idxs, loop_order, rhs, lhs_idxs, True, prefixes)
            features.extend(prefixes.values())
            stats_interpreter(stmt)

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, 4)

    def get_format_features(
        self,
        alias: Alias,
        level: int,
        num_pos: float,
        format_key: str | None,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        """Return the features at a zero-based depth with num_pos input positions.

        format_key selects a level option; None selects the element level at
        the tensor's rank. term is the LoopOrderedForm program which reads and
        writes alias. Each format has a block of ten columns: six allocation
        features followed by four access features. The first row describes the
        allocation; subsequent rows describe accesses at individual loop prefixes.
        """
        stats = machine(alias)
        fields = stats.index_order
        fill_ftype = ftype(stats.fill_value)
        leaf = element(stats.fill_value, fill_ftype)
        assert leaf.position_type is not None
        val_size = np_dtype(fill_ftype).itemsize
        pos_size = np_dtype(leaf.position_type).itemsize

        access_features = self.get_access_features(alias, level, term, machine)
        features = np.zeros((1 + len(access_features), self.n_features))
        if format_key is None:
            # The element level stores one value for each position.
            block = len(self.formats)
            n_l, nnz_l = 1, num_pos
        else:
            block = list(self.formats).index(format_key)
            n_l = stats.get_dim_size(fields[level])
            nnz_l = self.nnz_after(stats, machine.stats_factory, level)
        features[0, 10 * block : 10 * block + 6] = self.get_level_features(
            num_pos, n_l, nnz_l, val_size, pos_size
        )
        features[1:, 10 * block + 6 : 10 * block + 10] = access_features
        return features

    def predict_format_cost(
        self,
        alias: Alias,
        level: int,
        num_pos: float,
        format_key: str | None,
        term: LogicStatement,
        machine: StatsMachine[NS],
    ) -> float:
        features = self.get_format_features(
            alias, level, num_pos, format_key, term, machine
        )
        return self.predictor(features)

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, list[str]],
        stats: Mapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        """
        Return the features of the formats in bindings, which name the level
        option of each level of an alias, outermost first.
        """
        features = []
        stats_interpreter = StatsMachine(stats_factory, dict(stats))
        visited: set[Alias] = set()

        def visit_stmt(stmt: LogicStatement):
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                case Query(Table(Alias() as lhs, _), _) | QueryInto(
                    Table(Alias() as lhs, _), _, _
                ):
                    stats_interpreter(stmt)
                    # Each alias is formatted once, when it is first written.
                    match bindings.get(lhs):
                        case list() as format_keys if lhs not in visited:
                            lhs_stats = stats_interpreter(lhs)
                            num_pos = 1.0
                            for level, key in enumerate((*format_keys, None)):
                                features.append(
                                    self.get_format_features(
                                        lhs,
                                        level,
                                        num_pos,
                                        key,
                                        term,
                                        stats_interpreter,
                                    )
                                )
                                if key is not None:
                                    n_l = lhs_stats.get_dim_size(
                                        lhs_stats.index_order[level]
                                    )
                                    nnz_l = self.nnz_after(
                                        lhs_stats, stats_factory, level
                                    )
                                    num_pos = self.formats[key].next_num_pos(
                                        num_pos, n_l, nnz_l
                                    )
                    visited.add(lhs)
                case Produces():
                    return
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        visit_stmt(term)
        if not features:
            return np.empty((0, self.n_features))
        return np.concatenate(features, axis=0)


class FlopsCostModel(CostModel, AliasedForm):
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

    def get_features(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: Mapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> np.ndarray:
        stats = dict(stats)
        features = []

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
            match stmt:
                case Plan(bodies):
                    for body in bodies:
                        visit_stmt(body)
                case Query(Table(Alias() as lhs, idxs), rhs):
                    out_stats, iter_stats = visit_expr(rhs)
                    stats[lhs] = stats_factory.reorder(out_stats, idxs)
                    features.append(
                        (
                            iter_stats.estimate_non_fill_values(),
                            out_stats.estimate_non_fill_values(),
                        )
                    )
                case QueryInto():
                    visit_stmt(stmt.as_query())
                case Produces():
                    return
                case _:
                    raise ValueError(f"Unsupported statement type: {stmt}")

        visit_stmt(term)
        return np.asarray(features, dtype=float).reshape(-1, self.n_features)


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

    @property
    def n_features(self) -> int:
        return 2

    def __init__(self, predictor: Callable[[np.ndarray], float] | None = None):
        super().__init__(
            predictor
            if predictor is not None
            else LinearPredictor(np.array([0.5, 60.0]))
        )

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
        features = np.zeros(self.n_features)
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
