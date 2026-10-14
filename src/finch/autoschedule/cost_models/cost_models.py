from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping
from operator import attrgetter
from typing import Any, Generic, TypeVar

import numpy as np

from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import Alias, LogicStatement, StatsFactory, TensorStats
from finch.symbolic import Form

NS = TypeVar("NS", bound=NumericStats)
C = TypeVar("C")
K = TypeVar("K", bound=Hashable)


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


class CostModel(Form, Generic[C]):
    """
    A cost model predicts the cost of a logic statement as a sum over its
    components, such as queries, loop prefixes, or tensor levels:

        sum_c predict_component_cost(c)

    Each stage decomposes programs into its own kind of component, and its
    searches price one component at a time. components yields the components
    of a program, advancing the StatsMachine past each statement after yielding
    its components. get_component_features returns X_c with shape
    (n_rows, n_features). By default, the predictor callable takes X_c and
    returns a scalar sum of independently modeled rows, including zero for an
    empty matrix. Like a Stage, a CostModel inherits the form of statements it
    accepts.
    """

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    def __init__(self, predictor: Callable[[np.ndarray], float]):
        self.predictor = predictor

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

    def predict_component_cost(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> float:
        return self.predictor(self.get_component_features(component, machine))

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

    def predict_cost(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        stats: Mapping[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> float:
        machine = StatsMachine(stats_factory, dict(stats))
        return sum(
            (
                self.predict_component_cost(component, machine)
                for component in self.components(term, bindings, machine)
            ),
            0.0,
        )


class Combination(CostModel[C]):
    """
    A cost model of the components of its submodels, which must describe the
    same kind of component. The feature columns of the submodels are
    concatenated.
    """

    def __init__(self, *models: CostModel[C]):
        if not models:
            raise ValueError(f"{type(self).__name__} requires at least one model")
        self.models = models

    @property
    def n_features(self) -> int:
        return sum(model.n_features for model in self.models)

    def validate_inputs(self, *inputs):
        for model in self.models:
            model.validate_inputs(*inputs)

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, Any],
        machine: StatsMachine[NS],
    ) -> Iterator[C]:
        return self.models[0].components(term, bindings, machine)

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        return np.concatenate(
            [model.get_component_features(component, machine) for model in self.models],
            axis=1,
        )


class Sum(Combination[C]):
    """
    Sum the costs that submodels predict for each component. The coefficients
    of linear submodels concatenate like their features.
    """

    def predict_component_cost(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> float:
        return sum(
            (model.predict_component_cost(component, machine) for model in self.models),
            0.0,
        )


class Concat(Combination[C]):
    """Predict the cost of each component from the features of all submodels."""

    def __init__(
        self,
        *models: CostModel[C],
        predictor: Callable[[np.ndarray], float],
    ):
        super().__init__(*models)
        self.predictor = predictor


class ByKind(Combination[C], Generic[C, K]):
    """
    Predict the cost of each component with the submodel of its kind, which
    key returns, by default the kind attribute of the component. The features
    of a component fill the columns of its submodel, and the other columns are
    zero, so the columns of a linear fit line up with the coefficients of each
    kind.
    """

    def __init__(
        self,
        models: Mapping[K, CostModel[C]],
        key: Callable[[C], K] = attrgetter("kind"),
    ):
        super().__init__(*models.values())
        self.by_kind = dict(models)
        self.key = key

    def get_component_features(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        kind = self.key(component)
        start = 0
        for other, model in self.by_kind.items():
            if other == kind:
                rows = model.get_component_features(component, machine)
                features = np.zeros((len(rows), self.n_features))
                features[:, start : start + model.n_features] = rows
                return features
            start += model.n_features
        raise KeyError(kind)

    def predict_component_cost(
        self,
        component: C,
        machine: StatsMachine[NS],
    ) -> float:
        model = self.by_kind[self.key(component)]
        return model.predict_component_cost(component, machine)
