from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Generic, TypeVar

import numpy as np

from finch.algebra import FinchOperator
from finch.finch_logic import Aggregate, Field, LogicExpression, MapJoin
from finch.finch_logic.tensor_stats import StatsFactory

from .exact_stats import ExactStats, ExactStatsFactory
from .numeric_stats import NumericStats

NS = TypeVar("NS", bound=NumericStats)


class ErrorLoggerStatsFactory(StatsFactory["ErrorLoggerStats[NS]"], Generic[NS]):
    def __init__(
        self,
        estimate_factory: StatsFactory[NS],
        exact_factory: ExactStatsFactory | None = None,
        error_log: ErrorLog | None = None,
    ):
        self.estimate_factory = estimate_factory
        if exact_factory is None:
            self.exact_factory = ExactStatsFactory()
        else:
            self.exact_factory = exact_factory

        if error_log is None:
            self.error_log = ErrorLog()
        else:
            self.error_log = error_log

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> ErrorLoggerStats[NS]:
        return ErrorLoggerStats(
            exact=self.exact_factory(tensor, fields),
            estimate=self.estimate_factory(tensor, fields),
        )

    def copy(self, stat: ErrorLoggerStats[NS]) -> ErrorLoggerStats[NS]:
        return ErrorLoggerStats(
            exact=self.exact_factory.copy(stat.exact),
            estimate=self.estimate_factory.copy(stat.estimate),
        )

    def support(self, stat: ErrorLoggerStats[NS]) -> ErrorLoggerStats[NS]:
        return ErrorLoggerStats(
            exact=self.exact_factory.support(stat.exact),
            estimate=self.estimate_factory.support(stat.estimate),
        )

    def mapjoin(
        self, op: FinchOperator, *stats: ErrorLoggerStats[NS]
    ) -> ErrorLoggerStats[NS]:
        exact = self.exact_factory.mapjoin(op, *(stat.exact for stat in stats))
        estimate = self.estimate_factory.mapjoin(op, *(stat.estimate for stat in stats))
        self.error_log.record(MapJoin, op, exact, estimate)
        return ErrorLoggerStats(exact=exact, estimate=estimate)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ErrorLoggerStats[NS],
    ) -> ErrorLoggerStats[NS]:
        exact = self.exact_factory.aggregate(op, init, reduce_indices, stats.exact)
        estimate = self.estimate_factory.aggregate(
            op, init, reduce_indices, stats.estimate
        )
        self.error_log.record(Aggregate, op, exact, estimate)
        return ErrorLoggerStats(exact=exact, estimate=estimate)

    def relabel(
        self, stats: ErrorLoggerStats[NS], relabel_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats[NS]:
        estimate = self.estimate_factory.relabel(stats.estimate, relabel_indices)
        exact = self.exact_factory.relabel(stats.exact, relabel_indices)
        return ErrorLoggerStats(exact=exact, estimate=estimate)

    def reorder(
        self, stats: ErrorLoggerStats[NS], reorder_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats[NS]:
        estimate = self.estimate_factory.reorder(stats.estimate, reorder_indices)
        exact = self.exact_factory.reorder(stats.exact, reorder_indices)
        return ErrorLoggerStats(exact=exact, estimate=estimate)


class ErrorLoggerStats(NumericStats, Generic[NS]):
    def __init__(
        self,
        exact: ExactStats,
        estimate: NS,
    ):
        super().__init__(estimate)
        self.exact = exact
        self.estimate = estimate

    def estimate_non_fill_values(self, over: Iterable[Field] = ()) -> float:
        return self.estimate.estimate_non_fill_values(over=over)

    def get_embedding(self) -> np.ndarray:
        return self.estimate.get_embedding()


class ErrorLog:
    """Record difference between estimated and exact stats."""

    def __init__(self):
        self.samples = {}

    def record(
        self,
        operation: type[LogicExpression],
        operator: FinchOperator,
        exact: NumericStats,
        estimate: NumericStats,
    ) -> None:
        label = f"{operation.__name__}({operator})"

        if label not in self.samples:
            self.samples[label] = []

        self.samples[label].append(
            (exact.estimate_non_fill_values(), estimate.estimate_non_fill_values())
        )


class ErrorReport:
    """Create box and whisker plots from errors."""

    def __init__(self, error_log: ErrorLog):
        self.error_log = error_log

    def plot(self, filename: str) -> None:
        import matplotlib.pyplot as plt

        labels = []
        errors = []
        for operation, samples in self.error_log.samples.items():
            operation_errors = [abs(estimate - exact) for exact, estimate in samples]
            print(f"{operation}: {operation_errors}")
            labels.append(operation)
            errors.append(operation_errors)

        figure, axis = plt.subplots()
        axis.boxplot(errors, tick_labels=labels, orientation="horizontal")
        axis.set_xlabel("Absolute NNZ error")
        axis.set_ylabel("Operation")
        axis.set_title("NNZ Estimation Error")
        figure.savefig(filename, bbox_inches="tight")
        plt.close(figure)
