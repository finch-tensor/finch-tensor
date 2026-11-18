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

    def support(self, stat: ErrorLoggerStats) -> ErrorLoggerStats:
        return ErrorLoggerStats(
            log=self.log,
            exact=self.exact_factory.support(stat.exact),
            estimates={
                factory: factory.support(stat.estimates[factory])
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
        return self.log.samples[next(iter(self.estimates))][-1][1]

    def get_embedding(self) -> np.ndarray:
        return self.estimate.get_embedding()


class ErrorLog:
    """Record every exact and estimated non-fill count pair per estimator."""

    def __init__(self):
        self.samples: dict[StatsFactory[Any], list[tuple[float, float]]] = {}

    def record(
        self,
        estimator: StatsFactory[Any],
        exact: float,
        estimate: float,
    ) -> None:
        self.samples.setdefault(estimator, []).append((exact, estimate))


class ErrorReport:
    """Create box plots of estimate/exact ratios across prediction requests."""

    def __init__(self, log: ErrorLog):
        self.log = log

    def plot(self, filename: str, *, max_ratio: float | None = 10.0) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        if max_ratio is not None and (not np.isfinite(max_ratio) or max_ratio <= 1):
            raise ValueError("max_ratio must be finite and greater than 1, or None")

        estimate_ratios = []
        labels = []
        for factory, samples in self.log.samples.items():
            ratios = []
            for exact, estimate in samples:
                if np.isnan(exact) or np.isnan(estimate) or exact < 0 or estimate < 0:
                    ratio = float("nan")
                elif exact == 0:
                    ratio = 1.0 if estimate == 0 else float("inf")
                else:
                    ratio = estimate / exact
                ratios.append(ratio)
            ratios = np.array(ratios)
            estimate_ratios.append(ratios[np.isfinite(ratios) & (ratios > 0)])

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
            counts = [f"n={len(samples)}"]
            for mask, label in (
                (ratios == 0, "zero"),
                (np.isposinf(ratios), "∞"),
                (np.isnan(ratios), "invalid"),
            ):
                count = np.count_nonzero(mask)
                if count:
                    counts.append(f"{label}={count}")
            labels.append(f"{name} ({', '.join(counts)})")

        minimum = min(
            (ratio for ratios in estimate_ratios for ratio in ratios),
            default=1.0,
        )
        maximum = max(
            (ratio for ratios in estimate_ratios for ratio in ratios),
            default=1.0,
        )
        right_limit = max(2.0, maximum * 1.2)
        if max_ratio is not None:
            right_limit = min(right_limit, max_ratio)
        figure, axis = plt.subplots(figsize=(10, max(3, 0.4 * len(labels) + 2.5)))
        if estimate_ratios:
            artists = axis.boxplot(
                estimate_ratios,
                orientation="horizontal",
                tick_labels=labels,
                patch_artist=True,
                whis=(0, 100),
                showfliers=False,
                boxprops={"facecolor": "none"},
                medianprops={"color": "black"},
            )
            for box in artists["boxes"]:
                bounds = box.get_path().vertices
                if not np.isfinite(bounds).all():
                    continue
                left, bottom = bounds.min(axis=0)
                right, top = bounds.max(axis=0)
                if left < 1:
                    axis.fill_betweenx(
                        [bottom, top],
                        left,
                        min(right, 1),
                        color="tab:blue",
                        alpha=0.35,
                        zorder=1,
                    )
                if right > 1:
                    axis.fill_betweenx(
                        [bottom, top],
                        max(left, 1),
                        right,
                        color="tab:orange",
                        alpha=0.35,
                        zorder=1,
                    )
                box.set_edgecolor(
                    "tab:blue"
                    if right < 1
                    else "tab:orange"
                    if left > 1
                    else "tab:gray"
                )
            for line in artists["whiskers"] + artists["caps"] + artists["medians"]:
                values = line.get_xdata()
                line.set_color(
                    "tab:blue"
                    if max(values) < 1
                    else "tab:orange"
                    if min(values) > 1
                    else "tab:gray"
                )
            for position, ratios in enumerate(estimate_ratios, start=1):
                if not len(ratios):
                    axis.text(
                        1,
                        position,
                        "No positive finite ratios",
                        ha="center",
                        va="center",
                    )
        else:
            axis.set_yticks([])
            axis.text(
                0.5,
                0.5,
                "No predictions recorded",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_xscale("log")
        axis.set_xlim(min(0.5, minimum / 1.2), right_limit)
        for position, ratios in enumerate(estimate_ratios, start=1):
            if not len(ratios) or ratios.max() <= right_limit:
                continue
            axis.plot(
                right_limit,
                position,
                marker=">",
                color="tab:orange",
                markersize=6,
                clip_on=False,
            )
            axis.annotate(
                f"{ratios.max():,.6g}×",
                (right_limit, position),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="tab:orange",
                annotation_clip=False,
            )
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.axvline(1, color="black", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="both", alpha=0.25)
        axis.set_xlabel(
            "Estimate / exact (log scale; 1 = exact)\n"
            "Blue < 1: underestimation; orange > 1: overestimation"
        )
        axis.set_ylabel("Estimator\n(n = number of prediction requests)")
        axis.set_title("Estimate / Exact Distributions")
        figure.savefig(filename, bbox_inches="tight")
        plt.close(figure)
