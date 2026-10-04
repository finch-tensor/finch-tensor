"""Compare full-size estimator accuracy against JSON and PNG references.

Regenerate with::

    pixi run -e test-julia pytest src/finch/tests/test_accuracy.py --regen-all

Bundled Matrix Market inputs retain their full dimensions. Input values are
normalized to their nonzero pattern to avoid signed cancellation. The uniform
random matrix and sampling factories are seeded; JSON is rounded to six decimals.
"""

from __future__ import annotations

import json
import logging
from io import BytesIO

import pytest

import numpy as np
import scipy.sparse as sps

from matplotlib.figure import Figure

import finch as ft
from finch.compile_jl.julia import julia_available
from finch.tests.stats_cases import (
    BLOCK_COUNT,
    DATASETS,
    KERNELS,
    RANDOM_DENSITY,
    RANDOM_MATRIX_SIZE,
    SEED,
    N,
    make_kernel_estimator,
    make_models,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not julia_available(), reason="accuracy regression requires the Julia backend"
    ),
]


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
        # Fresh factories isolate sampling masks between datasets.
        for model, factory in make_models().items():
            logging.getLogger(__name__).info("Estimating %s with %s", dataset, model)
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
                q_errors[model].append(q_error)
                results[kernel].setdefault(dataset, {})[model] = {
                    "actual_nnz": actual[kernel],
                    "estimated_nnz": round(estimate, 6),
                    "ratio": round(ratio, 6),
                    "q_error": round(q_error, 6),
                }
    return {
        "matrix_size": N,
        "random_matrix_size": RANDOM_MATRIX_SIZE,
        "random_density": RANDOM_DENSITY,
        "datasets": datasets,
        "block_count": BLOCK_COUNT,
        "seed": SEED,
        "results": results,
        "geomean_q_error": {
            model: round(float(np.exp(np.mean(np.log(errors)))), 6)
            for model, errors in q_errors.items()
        },
    }


def plot_accuracy(results):
    fig = Figure(figsize=(14, 14), layout="constrained")
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
            color = f"C{index}"
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
    fig.suptitle("Sparsity estimator accuracy")
    return fig


def plot_geomean(geomeans):
    fig = Figure(figsize=(12, 5), layout="constrained")
    ax = fig.subplots()
    models = list(geomeans)
    x = np.arange(len(models))
    ax.bar(x, list(geomeans.values()), color=[f"C{i}" for i in x])
    ax.axhline(1, color="black", linestyle="--", label="Perfect estimate")
    ax.set_yscale("log", base=2)
    ax.set_xticks(x, models, rotation=30, ha="right", fontsize=9)
    ax.set(ylabel="Geometric mean q-error", title="Overall estimator accuracy")
    ax.legend()
    return fig


def test_statistics_accuracy(accuracy_results, file_regression, image_regression):
    file_regression.check(
        json.dumps(accuracy_results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        extension=".json",
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
