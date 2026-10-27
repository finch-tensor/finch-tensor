from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from operator import attrgetter
from typing import Any, Generic, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike

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
    """

    @property
    @abstractmethod
    def n_features(self) -> int: ...

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


class Quadratic(Basis[C]):
    """
    The products of each pair of features of each row of a basis, flattened
    row-major, so a linear model of them sums a quadratic form of each row.
    """

    def __init__(self, basis: Basis[C]):
        self.basis = basis

    @property
    def n_features(self) -> int:
        return self.basis.n_features**2

    def validate_inputs(self, *inputs):
        self.basis.validate_inputs(*inputs)

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[C]:
        return self.basis.components(term, bindings, machine)

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        rows = self.basis.get_component_features(component, machine)
        return np.einsum("ij,ik->ijk", rows, rows).reshape(len(rows), -1)


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
    its runtime, and returns a model fitted to all of them at once.
    """

    def __init__(self, basis: Basis[C]):
        self.basis = basis

    @abstractmethod
    def predict(self, features: np.ndarray) -> float: ...

    @abstractmethod
    def fit(
        self, features: Sequence[np.ndarray], runtimes: Sequence[float]
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
    fit finds the coefficients by least squares.
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

    def fit(self, features: Sequence[np.ndarray], runtimes: Sequence[float]) -> Self:
        design = np.stack([rows.sum(axis=0) for rows in features])
        coefficients, *_ = np.linalg.lstsq(design, np.asarray(runtimes, dtype=float))
        return type(self)(self.basis, coefficients)
