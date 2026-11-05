from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from operator import attrgetter
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import nnls

from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import Alias, LogicStatement, StatsFactory, TensorStats
from finch.symbolic import Form

NS = TypeVar("NS", bound=NumericStats)
by_kind = attrgetter("kind")
C = TypeVar("C")
K = TypeVar("K", bound=Hashable)


class Basis(Form, Generic[C]):
    """
    A basis describes a logic statement by the features of its components,
    such as queries, loop prefixes, or tensor levels. components yields the
    components of a program, advancing the StatsMachine past each statement
    after yielding its components, and get_component_features gives a
    component one row of n_features for each independently priced part of
    it. Like a Stage, a Basis inherits the form of statements it accepts.
    ridge weights the penalty on the square of each coefficient of a linear
    model of the basis, and feature_names names each feature, by default by
    its index.
    """

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(self.n_features))

    @property
    def ridge(self) -> np.ndarray:
        return np.ones(self.n_features)

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
        machine = StatsMachine(stats_factory, dict(stats))
        features = [
            self.get_component_features(component, machine)
            for component in self.components(term, bindings, machine)
        ]
        return np.concatenate([np.empty((0, self.n_features)), *features])


class Concat(Basis[C]):
    """
    The features of several bases side by side, over the components of the
    first. A basis prices a component of another kind at zero, so bases of
    different kinds of component can be concatenated over the union of them.
    """

    def __init__(self, *bases: Basis[C]):
        if not bases:
            raise ValueError(f"{type(self).__name__} requires at least one basis")
        self.bases = bases

    @property
    def n_features(self) -> int:
        return sum(basis.n_features for basis in self.bases)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name for basis in self.bases for name in basis.feature_names)

    @property
    def ridge(self) -> np.ndarray:
        return np.concatenate([basis.ridge for basis in self.bases])

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
    so a linear model has separate coefficients for each kind. A component of
    a kind the basis doesn't list is zero in every column.
    """

    def __init__(
        self,
        bases: Mapping[K, Basis[C]],
        key: Callable[[C], K] = by_kind,
    ):
        super().__init__(*bases.values())
        self.by_kind = dict(bases)
        self.key = key

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"{kind}.{name}"
            for kind, basis in self.by_kind.items()
            for name in basis.feature_names
        )

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
        return np.zeros((1, self.n_features))


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

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = self.basis.feature_names
        return tuple(f"{a}*{b}" for a in names for b in names)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ik->ijk", rows, rows).reshape(len(rows), -1)


class Affine(RowMap[C]):
    """
    The features of each row of a basis followed by a 1, so a linear model of
    them adds a constant to the cost of each row.
    """

    @property
    def n_features(self) -> int:
        return self.basis.n_features + 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.basis.feature_names, "constant")

    @property
    def ridge(self) -> np.ndarray:
        return np.append(self.basis.ridge, 1.0)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return np.hstack([rows, np.ones((len(rows), 1))])


class Stratify(RowMap[C]):
    """
    The features of each row of a basis, in the block of columns of the
    regime of the row, with zeros in the blocks of the other regimes. The
    regime of a row is the number of thresholds its feature at index feature
    meets or exceeds. A linear model of this basis has separate coefficients
    for each regime, so a feature can cost differently once, say, the working
    set exceeds a cache.
    """

    def __init__(self, basis: Basis[C], feature: int, thresholds: ArrayLike):
        super().__init__(basis)
        if not 0 <= feature < basis.n_features:
            raise ValueError(
                f"Expected 0 <= feature < {basis.n_features}, got {feature}"
            )
        self.feature = feature
        self.thresholds = np.asarray(thresholds, dtype=float)
        if self.thresholds.ndim != 1 or not np.all(np.diff(self.thresholds) > 0):
            raise ValueError(f"Expected increasing thresholds, got {thresholds}")

    @property
    def n_regimes(self) -> int:
        return len(self.thresholds) + 1

    @property
    def n_features(self) -> int:
        return self.n_regimes * self.basis.n_features

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"regime{regime}.{name}"
            for regime in range(self.n_regimes)
            for name in self.basis.feature_names
        )

    @property
    def ridge(self) -> np.ndarray:
        return np.tile(self.basis.ridge, self.n_regimes)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        n = self.basis.n_features
        regime = np.searchsorted(self.thresholds, rows[:, self.feature], side="right")
        columns = regime[:, None] * n + np.arange(n)
        features = np.zeros((len(rows), self.n_features))
        np.put_along_axis(features, columns, rows, axis=1)
        return features


class CostModel(Form, Generic[C]):
    """
    A linear cost model weights the features its basis gives the components
    of a logic statement by its coefficients, and sums over their rows, so
    the cost of a program is the sum of the costs of its components, which
    predict_component_cost prices one at a time. Runtimes are measured for
    whole programs, so fit takes the features of each program, as
    get_features returns them, with its runtime, and finds the nonnegative
    coefficients which minimize the sum of the squared relative errors of the
    predicted runtimes, so that fast programs count as much as slow ones, and
    the cost of a row of nonnegative features is nonnegative, penalized by
    alpha times the ridge weights of the basis. The runtimes must be
    positive, and a fit to no programs is zero. to_json writes the
    coefficients with the names of their features, which from_json checks
    against the basis of the model it reads them into.
    """

    def __init__(self, basis: Basis[C], coefficients: ArrayLike):
        self.basis = basis
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != (basis.n_features,):
            raise ValueError(
                f"Expected {basis.n_features} coefficients, "
                f"got shape {self.coefficients.shape}"
            )

    def validate_inputs(self, *inputs):
        self.basis.validate_inputs(*inputs)

    def predict(self, features: np.ndarray) -> float:
        return float(features.sum(axis=0) @ self.coefficients)

    def predict_component_cost(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> float:
        return self.predict(self.basis.get_component_features(component, machine))

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

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.basis.feature_names),
            "coefficients": self.coefficients.tolist(),
        }

    def from_json(self, data: Mapping[str, Any]) -> "CostModel[C]":
        names = list(self.basis.feature_names)
        if data["feature_names"] != names:
            raise ValueError(
                f"Expected the coefficients of {names}, got {data['feature_names']}"
            )
        return type(self)(self.basis, data["coefficients"])

    def fit(
        self,
        features: Sequence[np.ndarray],
        runtimes: ArrayLike,
        alpha: float = 0.0,
    ) -> "CostModel[C]":
        n = self.basis.n_features
        runtimes = np.asarray(runtimes, dtype=float)
        if np.any(runtimes <= 0):
            raise ValueError(f"Expected positive runtimes, got {runtimes.min()}")
        # Dividing the features of each program by its runtime makes the
        # residuals relative errors, which least squares fits to a target of
        # ones. The ridge penalty is least squares against extra rows of zeros.
        sums = np.array([rows.sum(axis=0) for rows in features]).reshape(-1, n)
        design = np.concatenate(
            [sums / runtimes[:, None], np.diag(np.sqrt(alpha * self.basis.ridge))]
        )
        target = np.concatenate([np.ones(len(runtimes)), np.zeros(n)])
        coefficients, _ = nnls(design, target)
        return type(self)(self.basis, coefficients)
