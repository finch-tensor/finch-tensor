from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from finch.algebra import FinchOperator
from finch.finch_logic import Field
from finch.finch_logic.tensor_stats import StatsFactory

from .blocked_stats import BlockedStatsFactory
from .exact_stats import ExactStats, ExactStatsFactory
from .numeric_stats import NumericStats
from .sampling_stats import SamplingStatsFactory
from .tensor_stats import BaseTensorStatsFactory


class ErrorLoggerStatsFactory(
    BaseTensorStatsFactory["ErrorLoggerStats"],
    StatsFactory["ErrorLoggerStats"],
):
    def __init__(
        self,
        *estimate_factories: StatsFactory[Any],
        error_log: ErrorLog | None = None,
    ):
        if not estimate_factories:
            raise ValueError("At least one estimator is required")

        super().__init__(ErrorLoggerStats)
        self.estimate_factories = estimate_factories

        self.exact_factory = ExactStatsFactory()
        self.error_log = ErrorLog() if error_log is None else error_log

    def record(self, stat: ErrorLoggerStats) -> None:
        for factory, estimate in stat.estimates.items():
            self.error_log.record(factory, stat.exact, estimate)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> ErrorLoggerStats:
        return ErrorLoggerStats(
            exact=self.exact_factory(tensor, fields),
            estimates={
                factory: factory(tensor, fields) for factory in self.estimate_factories
            },
        )

    def copy(self, stat: ErrorLoggerStats) -> ErrorLoggerStats:
        return ErrorLoggerStats(
            exact=self.exact_factory.copy(stat.exact),
            estimates={
                factory: factory.copy(stat.estimates[factory])
                for factory in self.estimate_factories
            },
        )

    def mapjoin(self, op: FinchOperator, *stats: ErrorLoggerStats) -> ErrorLoggerStats:
        exact = self.exact_factory.mapjoin(op, *(stat.exact for stat in stats))
        estimates = {
            factory: factory.mapjoin(op, *(stat.estimates[factory] for stat in stats))
            for factory in self.estimate_factories
        }
        return ErrorLoggerStats(exact=exact, estimates=estimates)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ErrorLoggerStats,
    ) -> ErrorLoggerStats:
        exact = self.exact_factory.aggregate(op, init, reduce_indices, stats.exact)
        estimates = {
            factory: factory.aggregate(
                op, init, reduce_indices, stats.estimates[factory]
            )
            for factory in self.estimate_factories
        }
        return ErrorLoggerStats(exact=exact, estimates=estimates)

    def relabel(
        self, stats: ErrorLoggerStats, relabel_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats:
        estimates = {
            factory: factory.relabel(stats.estimates[factory], relabel_indices)
            for factory in self.estimate_factories
        }
        exact = self.exact_factory.relabel(stats.exact, relabel_indices)
        return ErrorLoggerStats(exact=exact, estimates=estimates)

    def reorder(
        self, stats: ErrorLoggerStats, reorder_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats:
        estimates = {
            factory: factory.reorder(stats.estimates[factory], reorder_indices)
            for factory in self.estimate_factories
        }
        exact = self.exact_factory.reorder(stats.exact, reorder_indices)
        return ErrorLoggerStats(exact=exact, estimates=estimates)


class ErrorLoggerStats(NumericStats):
    def __init__(
        self, exact: ExactStats, estimates: dict[StatsFactory[Any], NumericStats]
    ):
        self.estimate = next(iter(estimates.values()))
        super().__init__(self.estimate)
        self.exact = exact
        self.estimates = estimates

    def estimate_non_fill_values(self, over: Iterable[Field] = ()) -> float:
        return self.estimate.estimate_non_fill_values(over=over)

    def get_embedding(self) -> np.ndarray:
        return self.estimate.get_embedding()


class ErrorLog:
    """Record one exact and estimated non-fill count pair per estimator."""

    def __init__(self):
        self.samples: dict[StatsFactory[Any], tuple[float, float]] = {}

    def record(
        self,
        estimator: StatsFactory[Any],
        exact: NumericStats,
        estimate: NumericStats,
    ) -> None:
        if estimator in self.samples:
            raise ValueError("An estimate has already been recorded for this estimator")

        self.samples[estimator] = (
            exact.estimate_non_fill_values(),
            estimate.estimate_non_fill_values(),
        )


class ErrorReport:
    """Create bar charts of signed relative errors."""

    def __init__(self, error_log: ErrorLog):
        self.error_log = error_log

    def plot(self, filename: str) -> None:
        import matplotlib.pyplot as plt

        relative_errors = []
        labels = []
        for factory, (exact, estimate) in self.error_log.samples.items():
            if exact == 0:
                error = 0.0 if estimate == 0 else float("inf")
            else:
                error = (estimate - exact) / exact
            relative_errors.append(error)

            inner_factory = factory
            blocked_names = []
            while True:
                match inner_factory:
                    case BlockedStatsFactory(inner_factory=inner):
                        blocked_names.append(
                            type(inner_factory)
                            .__name__.removesuffix("Factory")
                            .removesuffix("Stats")
                        )
                        inner_factory = inner
                    case _:
                        break
            name = (
                type(inner_factory)
                .__name__.removesuffix("Factory")
                .removesuffix("Stats")
            )
            match inner_factory:
                case SamplingStatsFactory(estimator=estimator):
                    name = f"{name} ({estimator})"
            name = " ".join([*blocked_names, name])
            labels.append(name)
            print(f"{name}: {error}")

        figure, axis = plt.subplots(figsize=(10, 5))
        bars = axis.barh(
            range(len(labels)),
            [error if np.isfinite(error) else 0.0 for error in relative_errors],
            tick_label=labels,
        )
        for bar, error in zip(bars, relative_errors, strict=True):
            if np.isinf(error):
                axis.annotate(
                    "∞",
                    (0, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Relative Error: (estimate - exact) / estimate")
        axis.set_ylabel("Estimator")
        axis.set_title("Relative Error Across Stats")
        figure.savefig(filename, bbox_inches="tight")
        plt.close(figure)
