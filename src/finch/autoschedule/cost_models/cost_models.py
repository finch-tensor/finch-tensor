from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import TypeVar

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.finch_logic import Alias, LogicStatement, StatsFactory, TensorStats
from finch.symbolic import Form

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
