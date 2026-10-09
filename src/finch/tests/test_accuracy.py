"""Compare full-size estimator accuracy against JSON and PNG references.

Regenerate with::

    pixi run -e test-julia pytest src/finch/tests/test_accuracy.py --regen-all

Bundled Matrix Market inputs retain their full dimensions. Input values are
normalized to their nonzero pattern to avoid signed cancellation. The uniform
random matrix and sampling factories are seeded; JSON is rounded to six decimals.
Floating-point comparisons allow rounding noise across platforms.
Synthetic inputs exceed the sampling budget. Sampling estimators share sketches.
"""

from __future__ import annotations

import json
import logging
import math
from io import BytesIO
from pathlib import Path

import pytest

import numpy as np
import scipy.sparse as sps

from matplotlib import colormaps
from matplotlib.figure import Figure

import finch as ft
from finch.autoschedule.tensor_stats.sampling_stats import (
    SamplingStats,
    SamplingStatsFactory,
)
from finch.compile_jl.julia import julia_available
from finch.tests.stats_cases import (
    BLOCK_COUNT,
    DATASETS,
    DENSE_MATRIX_SIZE,
    KERNELS,
    RANDOM_DENSITY,
    RANDOM_MATRIX_SIZE,
    SAMPLE_NNZ,
    SEED,
    SPARSE_MATRIX_SIZE,
    make_kernel_estimator,
    make_kernel_stats,
    make_models,
)

GEOMEAN_DATASETS = ("ct20stif", "roadNet-PA", "soc-sign-epinions")


def actual_nonzeros(matrix, batch_rows=128):
    # Boolean products count reachability without large path counts. Row batches
    # bound intermediate storage even when powers of a sparse graph become dense.
    pattern = matrix.astype(bool)
    actual = dict.fromkeys(KERNELS, 0)
    actual["Hadamard"] = int(pattern.count_nonzero())
    for start in range(0, pattern.shape[0], batch_rows):
        rows = pattern[start : start + batch_rows]
        squared = rows @ pattern
        actual["SpGEMM"] += int(squared.count_nonzero())
        actual["SpGEMM2"] += int((squared @ pattern).count_nonzero())
        actual["Triangle Counting"] += int(squared.multiply(rows).count_nonzero())
    return actual


@pytest.fixture(scope="module")
def accuracy_results():
    results = {kernel: {} for kernel in KERNELS}
    q_errors = {model: [] for model in make_models()}
    datasets = {}
    for dataset, make_matrix in DATASETS.items():
        a = sps.csr_array(make_matrix())
        datasets[dataset] = {"shape": list(a.shape), "nnz": int(a.nnz)}
        tensor = ft.asarray(a)
        logging.getLogger(__name__).info("Computing reference counts for %s", dataset)
        actual = actual_nonzeros(a)
        sampling_stats: dict[str, SamplingStats] = {}
        # Fresh factories isolate sampling masks between datasets.
        for model, factory in make_models().items():
            logging.getLogger(__name__).info("Estimating %s with %s", dataset, model)
            match factory:
                case SamplingStatsFactory():
                    # Sampling variants use identical budgets and seeds; only
                    # the estimator applied to the materialized sketch differs.
                    if not sampling_stats:
                        kernel_stats = make_kernel_stats(factory, tensor)
                        sampling_stats = {
                            kernel: kernel_stats(kernel) for kernel in KERNELS
                        }
                        for kernel, stats in sampling_stats.items():
                            _, nnz, _, _ = stats.scan(needs_freq=True)
                            assert nnz <= SAMPLE_NNZ
                            assert (
                                math.prod(stats.sample_probs) * stats.remainder_prob < 1
                            )
                            logging.getLogger(__name__).info(
                                "%s / %s: %s retained nonzeros", dataset, kernel, nnz
                            )
                    estimates = {}
                    for kernel, cached_stats in sampling_stats.items():
                        stats = factory.copy(cached_stats)
                        stats.estimator = factory.estimator
                        estimates[kernel] = stats.estimate_non_fill_values()
                case _:
                    estimate_kernel = make_kernel_estimator(factory, tensor)
                    estimates = {kernel: estimate_kernel(kernel) for kernel in KERNELS}
            for kernel, estimate in estimates.items():
                estimate = float(estimate)
                assert np.isfinite(estimate) and estimate >= 0, (
                    kernel,
                    dataset,
                    model,
                    estimate,
                )
                # Clamp only for ratios, retaining the raw counts in the reference.
                ratio = max(estimate, 1) / max(actual[kernel], 1)
                q_error = max(ratio, 1 / ratio)
                if dataset in GEOMEAN_DATASETS:
                    q_errors[model].append(q_error)
                results[kernel].setdefault(dataset, {})[model] = {
                    "actual_nnz": actual[kernel],
                    "estimated_nnz": round(estimate, 6),
                    "ratio": round(ratio, 6),
                    "q_error": round(q_error, 6),
                }
    return {
        "dense_matrix_size": DENSE_MATRIX_SIZE,
        "sparse_matrix_size": SPARSE_MATRIX_SIZE,
        "random_matrix_size": RANDOM_MATRIX_SIZE,
        "random_density": RANDOM_DENSITY,
        "datasets": datasets,
        "block_count": BLOCK_COUNT,
        "seed": SEED,
        "sample_nnz": SAMPLE_NNZ,
        "geomean_datasets": list(GEOMEAN_DATASETS),
        "results": results,
        "geomean_q_error": {
            model: round(float(np.exp(np.mean(np.log(errors)))), 6)
            for model, errors in q_errors.items()
        },
    }


def plot_accuracy(results):
    fig = Figure(figsize=(18, 16), layout="constrained")
    axes = fig.subplots(len(KERNELS), 1)
    models = list(make_models())
    x = np.arange(len(DATASETS))
    width = 0.8 / len(models)
    for ax, kernel in zip(axes, KERNELS, strict=True):
        for index, model in enumerate(models):
            values = np.log2(
                [results[kernel][dataset][model]["ratio"] for dataset in DATASETS]
            )
            positions = x + (index - (len(models) - 1) / 2) * width
            color = colormaps["tab20"](index / 19)
            ax.bar(
                positions,
                np.clip(values, -10, 10),
                width=width,
                color=color,
                label=model,
                zorder=3,
            )
            for position, value in zip(positions, values, strict=True):
                if abs(value) < 0.1:
                    ax.plot(position, 0, marker="x", color=color, markersize=5)
                elif abs(value) > 10:
                    ax.text(
                        position,
                        np.sign(value) * 9.5,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=kernel, ylabel="log2(estimated / actual nnz)", ylim=(-10, 10))
        ax.set_xticks(x, list(DATASETS), rotation=20, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle(f"Sparsity estimator accuracy — sampling budget {SAMPLE_NNZ:,} nnz")
    return fig


def plot_geomean(geomeans):
    fig = Figure(figsize=(14, 6), layout="constrained")
    ax = fig.subplots()
    models = list(make_models())
    x = np.arange(len(models))
    ax.bar(
        x,
        [geomeans[model] for model in models],
        color=[colormaps["tab20"](i / 19) for i in x],
    )
    ax.axhline(1, color="black", linestyle="--", label="Perfect estimate")
    ax.set_yscale("log", base=2)
    ax.set_xticks(x, models, rotation=35, ha="right", fontsize=9)
    ax.set(
        ylabel="Geometric mean q-error",
        title="Estimator accuracy — real matrices only\n"
        + ", ".join(GEOMEAN_DATASETS)
        + f" ({len(KERNELS)} kernels each)",
    )
    ax.legend()
    return fig


def check_accuracy_results(obtained_filename: Path, expected_filename: Path):
    obtained = json.loads(obtained_filename.read_text())
    # Large estimates can differ beyond six decimal places through roundoff.
    # Keep integer counts and metadata exact while comparing floats numerically.
    expected = json.loads(
        expected_filename.read_text(),
        parse_float=lambda value: pytest.approx(float(value), rel=1e-12, abs=1e-6),
    )
    assert obtained == expected


@pytest.mark.parametrize(
    "result,matches",
    [
        ({"estimated_nnz": 133092290.0, "actual_nnz": 3099760}, True),
        ({"estimated_nnz": 133092289.999999, "actual_nnz": 3099760}, True),
        ({"estimated_nnz": 133092390.0, "actual_nnz": 3099760}, False),
        ({"estimated_nnz": 133092290.0, "actual_nnz": 3099761}, False),
        ({"estimated_nnz": 133092290.0}, False),
    ],
)
def test_accuracy_results_comparison(tmp_path, result, matches):
    expected = tmp_path / "expected.json"
    obtained = tmp_path / "obtained.json"
    expected.write_text(
        json.dumps({"LP": {"estimated_nnz": 133092290.0, "actual_nnz": 3099760}})
    )
    obtained.write_text(json.dumps({"LP": result}))
    if matches:
        check_accuracy_results(obtained, expected)
    else:
        with pytest.raises(AssertionError):
            check_accuracy_results(obtained, expected)


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(), reason="accuracy regression requires the Julia backend"
)
def test_statistics_accuracy(accuracy_results, file_regression, image_regression):
    file_regression.check(
        json.dumps(accuracy_results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        extension=".json",
        check_fn=check_accuracy_results,
    )
    for view, fig in (
        ("accuracy", plot_accuracy(accuracy_results["results"])),
        ("geomean", plot_geomean(accuracy_results["geomean_q_error"])),
    ):
        with BytesIO() as output:
            fig.savefig(output, format="png", dpi=100)
            image_regression.check(
                output.getvalue(), basename=f"test_statistics_accuracy_plot_{view}_"
            )
