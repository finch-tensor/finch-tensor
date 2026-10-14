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




