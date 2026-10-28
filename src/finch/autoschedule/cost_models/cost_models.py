from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from operator import attrgetter
from typing import Any, Generic, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize, nnls

from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import Alias, LogicStatement, StatsFactory, TensorStats
from finch.symbolic import Form

NS = TypeVar("NS", bound=NumericStats)
C = TypeVar("C")
K = TypeVar("K", bound=Hashable)


class Basis(Form, Generic[C]):
    """
    A basis describes a logic statement by the features of its components,
    such as queries, loop prefixes, or tensor levels. Each stage decomposes
    programs into its own kind of component, and its searches price one
    component at a time. components yields the components of a program,
    advancing the StatsMachine past each statement after yielding its
    components. get_component_features returns X_c with shape
    (n_rows, n_features), with one row for each independently modeled part of
    the component. Like a Stage, a Basis inherits the form of statements it
    accepts.

    A basis also describes the coefficients of a linear model of it: ridge
    weights the penalty on the square of each coefficient, and constraints is
    a matrix whose product with the coefficients must be nonnegative.
    """

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    @property
    def ridge(self) -> np.ndarray:
        return np.ones(self.n_features)

    @property
    def constraints(self) -> np.ndarray:
        return np.empty((0, self.n_features))

    @abstractmethod
    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[C]: ...

    @abstractmethod
    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray: ...

    def get_features(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> np.ndarray:
        """Return the rows of every component of term, in order."""
        machine = StatsMachine(stats_factory, dict(stats))
        features = [
            self.get_component_features(component, machine)
            for component in self.components(term, bindings, machine)
        ]
        return np.concatenate([np.empty((0, self.n_features)), *features])


class Concat(Basis[C]):
    """
    The features of several bases side by side. The bases must describe the
    same kind of component. A linear model of the concatenation is the sum of
    linear models of each basis.
    """

    def __init__(self, *bases: Basis[C]):
        if not bases:
            raise ValueError(f"{type(self).__name__} requires at least one basis")
        self.bases = bases

    @property
    def n_features(self) -> int:
        return sum(basis.n_features for basis in self.bases)

    @property
    def ridge(self) -> np.ndarray:
        return np.concatenate([basis.ridge for basis in self.bases])

    @property
    def constraints(self) -> np.ndarray:
        return block_diag(*[basis.constraints for basis in self.bases])

    def validate_inputs(self, *inputs):
        for basis in self.bases:
            basis.validate_inputs(*inputs)

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[C]:
        return self.bases[0].components(term, bindings, machine)

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        return np.concatenate(
            [basis.get_component_features(component, machine) for basis in self.bases],
            axis=1,
        )


class ByKind(Concat[C], Generic[C, K]):
    """
    The features of each component under the basis of its kind, which key
    returns, by default the kind attribute of the component. The features of a
    component fill the columns of its basis, and the other columns are zero,
    so a linear model has separate coefficients for each kind.
    """

    def __init__(
        self,
        bases: Mapping[K, Basis[C]],
        key: Callable[[C], K] = attrgetter("kind"),
    ):
        super().__init__(*bases.values())
        self.by_kind = dict(bases)
        self.key = key

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        kind = self.key(component)
        start = 0
        for other, basis in self.by_kind.items():
            if other == kind:
                rows = basis.get_component_features(component, machine)
                features = np.zeros((len(rows), self.n_features))
                features[:, start : start + basis.n_features] = rows
                return features
            start += basis.n_features
        raise KeyError(kind)


class RowMap(Basis[C]):
    """
    A basis whose rows are a function, transform, of the rows of another
    basis for the same components.
    """

    def __init__(self, basis: Basis[C]):
        self.basis = basis

    def validate_inputs(self, *inputs):
        self.basis.validate_inputs(*inputs)

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[C]:
        return self.basis.components(term, bindings, machine)

    @abstractmethod
    def transform(self, rows: np.ndarray) -> np.ndarray: ...

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        return self.transform(self.basis.get_component_features(component, machine))


class Quadratic(RowMap[C]):
    """
    The products of each pair of features of each row of a basis, flattened
    row-major, so a linear model of them sums a quadratic form of each row.
    """

    @property
    def n_features(self) -> int:
        return self.basis.n_features**2

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ik->ijk", rows, rows).reshape(len(rows), -1)


class KNearest(RowMap[C]):
    """
    The weights of the k landmark rows nearest each row of a basis by
    Euclidean distance, in the columns of those landmarks, and zero
    elsewhere. The weights are inversely proportional to distance and sum to
    one, so a row on a landmark takes all of its weight. A linear model of
    this basis is a k nearest neighbors model whose coefficients are the
    costs of the landmarks, so fitting it by least squares apportions the
    runtime of each program among the landmarks its rows are near. The
    landmarks are usually rows of the basis from the programs the model is
    fitted to.
    """

    def __init__(self, basis: Basis[C], landmarks: ArrayLike, k: int = 1):
        super().__init__(basis)
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(
            -1, basis.n_features
        )
        if not 1 <= k <= len(self.landmarks):
            raise ValueError(f"Expected 1 <= k <= {len(self.landmarks)}, got {k}")
        self.k = k

    @property
    def n_features(self) -> int:
        return len(self.landmarks)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(rows[:, None] - self.landmarks[None], axis=2)
        nearest = np.argsort(distances, axis=1, kind="stable")[:, : self.k]
        distances = np.take_along_axis(distances, nearest, axis=1)
        # Rows on a landmark weight the landmarks at distance zero equally.
        on_landmark = distances == 0
        weights = np.where(
            on_landmark.any(axis=1, keepdims=True),
            on_landmark,
            1 / np.where(on_landmark, 1, distances),
        )
        weights = weights / weights.sum(axis=1, keepdims=True)
        features = np.zeros((len(rows), len(self.landmarks)))
        np.put_along_axis(features, nearest, weights, axis=1)
        return features


class Stratify(RowMap[C]):
    """
    The features of each row of a basis, in the block of columns of the
    regime of the row, with zeros in the blocks of the other regimes. The
    regime of a row is the number of thresholds its feature at index feature
    meets or exceeds. A linear model of this basis has separate coefficients
    for each regime, so a feature can cost differently once, say, the working
    set exceeds a cache. increasing constrains each coefficient to be
    nondecreasing from each regime to the next.
    """

    def __init__(
        self,
        basis: Basis[C],
        feature: int,
        thresholds: ArrayLike,
        increasing: bool = False,
    ):
        super().__init__(basis)
        if not 0 <= feature < basis.n_features:
            raise ValueError(
                f"Expected 0 <= feature < {basis.n_features}, got {feature}"
            )
        self.feature = feature
        self.thresholds = np.asarray(thresholds, dtype=float)
        if self.thresholds.ndim != 1 or not np.all(np.diff(self.thresholds) > 0):
            raise ValueError(f"Expected increasing thresholds, got {thresholds}")
        self.increasing = increasing

    @property
    def n_regimes(self) -> int:
        return len(self.thresholds) + 1

    @property
    def n_features(self) -> int:
        return self.n_regimes * self.basis.n_features

    @property
    def ridge(self) -> np.ndarray:
        return np.tile(self.basis.ridge, self.n_regimes)

    @property
    def constraints(self) -> np.ndarray:
        constraints = block_diag(*[self.basis.constraints] * self.n_regimes)
        if not self.increasing:
            return constraints
        steps = np.diff(np.eye(self.n_regimes), axis=0)
        return np.concatenate(
            [constraints, np.kron(steps, np.eye(self.basis.n_features))]
        )

    def transform(self, rows: np.ndarray) -> np.ndarray:
        n = self.basis.n_features
        regime = np.searchsorted(self.thresholds, rows[:, self.feature], side="right")
        columns = regime[:, None] * n + np.arange(n)
        features = np.zeros((len(rows), self.n_features))
        np.put_along_axis(features, columns, rows, axis=1)
        return features


class CostModel(Form, Generic[C]):
    """
    A cost model predicts the cost of a logic statement from the features its
    basis gives the components of the statement. predict takes the features
    of a component, or of a whole program, and returns a sum over their rows,
    including zero for an empty matrix, so the cost of a program is the sum of
    the costs of its components:

        sum_c predict_component_cost(c)

    Runtimes are measured for whole programs rather than components, so fit
    takes the features of each program, as get_features returns them, with
    its runtime, and returns a model fitted to all of them at once, with
    alpha scaling the ridge penalty of the basis. Fitted models predict
    nonnegative row costs, as branch-and-bound searches require.
    """

    def __init__(self, basis: Basis[C]):
        self.basis = basis

    @abstractmethod
    def predict(self, features: np.ndarray) -> float: ...

    @abstractmethod
    def fit(
        self,
        features: Sequence[np.ndarray],
        runtimes: ArrayLike,
        alpha: float = 0.0,
    ) -> Self: ...

    def validate_inputs(self, *inputs):
        self.basis.validate_inputs(*inputs)

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        return self.basis.get_component_features(component, machine)

    def predict_component_cost(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> float:
        return self.predict(self.get_component_features(component, machine))

    def get_features(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> np.ndarray:
        return self.basis.get_features(term, bindings, stats, stats_factory)

    def predict_cost(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> float:
        return self.predict(self.get_features(term, bindings, stats, stats_factory))


class LinearCostModel(CostModel[C]):
    """
    Weight the features of each row by coefficients, and sum over the rows.
    fit finds nonnegative coefficients by least squares, penalized by alpha
    times the ridge weights of the basis and subject to its constraints. The
    coefficients are nonnegative so that the costs of rows of nonnegative
    features are.
    """

    def __init__(self, basis: Basis[C], coefficients: ArrayLike):
        super().__init__(basis)
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != (basis.n_features,):
            raise ValueError(
                f"Expected {basis.n_features} coefficients, "
                f"got shape {self.coefficients.shape}"
            )

    def predict(self, features: np.ndarray) -> float:
        return float(features.sum(axis=0) @ self.coefficients)

    def fit(
        self,
        features: Sequence[np.ndarray],
        runtimes: ArrayLike,
        alpha: float = 0.0,
    ) -> Self:
        n = self.basis.n_features
        # The ridge penalty is least squares against extra rows of zeros.
        design = np.concatenate(
            [
                np.stack([rows.sum(axis=0) for rows in features]),
                np.diag(np.sqrt(alpha * self.basis.ridge)),
            ]
        )
        target = np.concatenate([np.asarray(runtimes, dtype=float), np.zeros(n)])
        constraints = self.basis.constraints
        if not len(constraints):
            coefficients, _ = nnls(design, target)
            return type(self)(self.basis, coefficients)
        result = minimize(
            lambda c: 0.5 * np.sum((design @ c - target) ** 2),
            np.zeros(n),
            jac=lambda c: design.T @ (design @ c - target),
            method="SLSQP",
            constraints=[
                LinearConstraint(np.concatenate([constraints, np.eye(n)]), 0, np.inf)
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise ValueError(f"Constrained fit failed: {result.message}")
        return type(self)(self.basis, np.maximum(result.x, 0))
