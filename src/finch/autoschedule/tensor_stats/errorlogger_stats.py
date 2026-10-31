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
    def __init__(self, *factories: StatsFactory[Any]):
        if not factories:
            raise ValueError("At least one estimator is required")

        super().__init__(ErrorLoggerStats)
        self.factories = factories
        self.exact_factory = ExactStatsFactory()
        self.log = ErrorLog()

        # Sampling factories with the same budget share one seed dict
        seeds: dict[int, dict[tuple[Field, int], int]] = {}
        for factory in factories:
            match factory:
                case SamplingStatsFactory(sample_nnz=sample_nnz):
                    factory._seeds = seeds.setdefault(sample_nnz, factory._seeds)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> ErrorLoggerStats:
        return ErrorLoggerStats(
            log=self.log,
            exact=self.exact_factory(tensor, fields),
            estimates={factory: factory(tensor, fields) for factory in self.factories},
        )

    def copy(self, stat: ErrorLoggerStats) -> ErrorLoggerStats:
        return ErrorLoggerStats(
            log=self.log,
            exact=self.exact_factory.copy(stat.exact),
            estimates={
                factory: factory.copy(stat.estimates[factory])
                for factory in self.factories
            },
        )

    def mapjoin(self, op: FinchOperator, *stats: ErrorLoggerStats) -> ErrorLoggerStats:
        exact = self.exact_factory.mapjoin(op, *(stat.exact for stat in stats))
        estimates = {
            factory: factory.mapjoin(op, *(stat.estimates[factory] for stat in stats))
            for factory in self.factories
        }
        return ErrorLoggerStats(log=self.log, exact=exact, estimates=estimates)

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
            for factory in self.factories
        }
        return ErrorLoggerStats(log=self.log, exact=exact, estimates=estimates)

    def relabel(
        self, stats: ErrorLoggerStats, relabel_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats:
        estimates = {
            factory: factory.relabel(stats.estimates[factory], relabel_indices)
            for factory in self.factories
        }
        exact = self.exact_factory.relabel(stats.exact, relabel_indices)
        return ErrorLoggerStats(log=self.log, exact=exact, estimates=estimates)

    def reorder(
        self, stats: ErrorLoggerStats, reorder_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats:
        estimates = {
            factory: factory.reorder(stats.estimates[factory], reorder_indices)
            for factory in self.factories
        }
        exact = self.exact_factory.reorder(stats.exact, reorder_indices)
        return ErrorLoggerStats(log=self.log, exact=exact, estimates=estimates)


class ErrorLoggerStats(NumericStats):
    def __init__(
        self,
        log: ErrorLog,
        exact: ExactStats,
        estimates: dict[StatsFactory[Any], NumericStats],
    ):
        self.estimate = next(iter(estimates.values()))
        super().__init__(self.estimate)
        self.log = log
        self.exact = exact
        self.estimates = estimates

    def estimate_non_fill_values(
        self, over: Iterable[Field] = (), *, exact_count: float | None = None
    ) -> float:
        fields = tuple(over)

        if exact_count is None:
            actual = self.exact.estimate_non_fill_values(over=fields)
        else:
            actual = exact_count

        for factory, estimate in self.estimates.items():
            count = estimate.estimate_non_fill_values(over=fields)
            self.log.record(factory, actual, count)
        return self.log.samples[next(iter(self.estimates))][1]

    def get_embedding(self) -> np.ndarray:
        return self.estimate.get_embedding()


class ErrorLog:
    """Record the latest exact and estimated non-fill count pair per estimator."""

    def __init__(self):
        self.samples: dict[StatsFactory[Any], tuple[float, float]] = {}

    def record(
        self,
        estimator: StatsFactory[Any],
        exact: float,
        estimate: float,
    ) -> None:
        self.samples[estimator] = (exact, estimate)


class ErrorReport:
    """Create bar charts of signed relative errors on a symmetric log scale."""

    def __init__(self, log: ErrorLog):
        self.log = log

    def plot(self, filename: str) -> None:
        import matplotlib.pyplot as plt

        relative_errors = []
        labels = []
        for factory, (exact, estimate) in self.log.samples.items():
            if exact == 0:
                error = 0.0 if estimate == 0 else float("inf")
            else:
                error = (estimate - exact) / exact
            relative_errors.append(error)

            names = []
            factories = [factory]
            for inner_factory in factories:
                name = (
                    type(inner_factory)
                    .__name__.removesuffix("Factory")
                    .removesuffix("Stats")
                )
                match inner_factory:
                    case BlockedStatsFactory(inner_factory=inner):
                        factories.append(inner)
                    case SamplingStatsFactory(estimator=estimator):
                        name = f"{name} ({estimator})"
                names.append(name)
            name = " ".join(names)
            labels.append(name)
            print(f"{name}: {error}")

        limit = max(
            (abs(error) for error in relative_errors if np.isfinite(error)),
            default=1.0,
        )
        limit = max(limit, 0.01)
        figure, axis = plt.subplots(figsize=(10, max(3, 0.4 * len(labels) + 1.5)))
        bars = axis.barh(
            range(len(labels)),
            np.nan_to_num(relative_errors, nan=0.0, posinf=limit, neginf=-limit),
            tick_label=labels,
            color=[
                "tab:blue" if error < 0 else "tab:orange" for error in relative_errors
            ],
        )
        for bar, error in zip(bars, relative_errors, strict=True):
            center = bar.get_y() + bar.get_height() / 2
            if np.isinf(error):
                axis.annotate(
                    "−∞" if error < 0 else "∞",
                    (bar.get_width(), center),
                    xytext=(-4 if error < 0 else 4, 0),
                    textcoords="offset points",
                    ha="right" if error < 0 else "left",
                    va="center",
                )
            elif error == 0:
                axis.plot(0, center, marker="o", color="tab:gray", markersize=3)
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlim(-1.2 * limit, 1.2 * limit)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="both", alpha=0.25)
        axis.set_xlabel(
            "Relative error: (estimate - exact) / exact\n"
            "Underestimation < 0 < Overestimation (symmetric log scale)"
        )
        axis.set_ylabel("Estimator")
        axis.set_title("Relative Error Across Stats")
        figure.savefig(filename, bbox_inches="tight")
        plt.close(figure)
