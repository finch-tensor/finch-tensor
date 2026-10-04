"""Compare estimator accuracy against JSON and PNG files in ``reference/``.

Regenerate with::

    pixi run -e test-julia pytest src/finch/tests/test_accuracy.py --regen-all

The synthetic datasets need no downloads. Sampling uses fresh, seeded factories;
JSON values are rounded to six decimals to suppress floating-point noise.
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
from finch.algebra import ffuncs
from finch.autoschedule.tensor_stats import (
    BlockedUniformStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
)
from finch.compile_jl.julia import julia_available
from finch.finch_logic import Field

pytestmark = pytest.mark.skipif(
    not julia_available(), reason="accuracy regression requires the Julia backend"
)

i, j, k, ell = (Field(name) for name in "ijkl")
N = 100
BLOCK_COUNT = 5
SEED = 42
KERNELS = ("Hadamard", "SpGEMM", "SpGEMM2", "Triangle Counting")


def make_models():
    sampling = SamplingStatsFactory(sample_prob=0.5)
    sampling._rng = np.random.default_rng(SEED)
    return {
        "Dense": DenseStatsFactory(),
        "Uniform": UniformStatsFactory(),
        "DC": DCStatsFactory(),
        "LP": LPStatsFactory(),
        "Sampling_0.5": sampling,
        "Blocked-Uniform": BlockedUniformStatsFactory(block_count=BLOCK_COUNT),
    }


def make_diagonal(n):
    return np.eye(n, dtype=np.float64)


def make_tridiagonal(n):
    A = np.eye(n, k=0) + np.eye(n, k=1) + np.eye(n, k=-1)
    return (A > 0).astype(np.float64)


def make_banded(n, bw=5):
    r, c = np.indices((n, n))
    return (np.abs(r - c) <= bw).astype(np.float64)


def make_triangular(n):
    return np.triu(np.ones((n, n), dtype=np.float64))


def make_striped(n):
    A = np.zeros((n, n), dtype=np.float64)
    A[:, ::5] = 1
    return A


# Keep the reference self-contained: the original SNAP .mat files are not bundled.
DATASETS = {
    "Diagonal": make_diagonal,
    "Tridiagonal": make_tridiagonal,
    "Banded": make_banded,
    "Triangular": make_triangular,
    "Striped": make_striped,
}


def est_hadamard(factory, tns_a, tns_b):
    s_a = factory(tns_a, (i, j))
    s_b = factory(tns_b, (i, j))
    return factory.mapjoin(ffuncs.mul, s_a, s_b).estimate_non_fill_values()


def est_spgemm(factory, tns_a, tns_b):
    s_a = factory(tns_a, (i, k))
    s_b = factory(tns_b, (k, j))
    mm = factory.mapjoin(ffuncs.mul, s_a, s_b)
    return factory.aggregate(ffuncs.add, 0.0, (k,), mm).estimate_non_fill_values()


def est_spgemm2(factory, tns_a, tns_b):
    s_a = factory(tns_a, (i, ell))
    s_b1 = factory(tns_b, (ell, k))
    mm1 = factory.aggregate(
        ffuncs.add, 0.0, (ell,), factory.mapjoin(ffuncs.mul, s_a, s_b1)
    )
    s_b2 = factory(tns_b, (k, j))
    mm2 = factory.mapjoin(ffuncs.mul, mm1, s_b2)
    return factory.aggregate(ffuncs.add, 0.0, (k,), mm2).estimate_non_fill_values()


def est_triangle(factory, tns_a):
    s_a1 = factory(tns_a, (i, k))
    s_a2 = factory(tns_a, (k, j))
    mm = factory.aggregate(
        ffuncs.add, 0.0, (k,), factory.mapjoin(ffuncs.mul, s_a1, s_a2)
    )
    s_a3 = factory(tns_a, (i, j))
    return factory.mapjoin(ffuncs.mul, mm, s_a3).estimate_non_fill_values()


@pytest.fixture(scope="module")
def accuracy_results():
    results = {kernel: {} for kernel in KERNELS}
    q_errors = {model: [] for model in make_models()}
    for dataset, make_matrix in DATASETS.items():
        a = sps.csr_array(make_matrix(N))
        b = a.copy()
        tns_a, tns_b = ft.asarray(a), ft.asarray(b)
        ab = a @ b
        actual = {
            "Hadamard": int(a.multiply(b).count_nonzero()),
            "SpGEMM": int(ab.count_nonzero()),
            "SpGEMM2": int((ab @ b).count_nonzero()),
            "Triangle Counting": int((a @ a).multiply(a).count_nonzero()),
        }
        # Fresh factories isolate sampling masks between datasets.
        for model, factory in make_models().items():
            logging.getLogger(__name__).info("Estimating %s with %s", dataset, model)
            estimates = {
                "Hadamard": est_hadamard(factory, tns_a, tns_b),
                "SpGEMM": est_spgemm(factory, tns_a, tns_b),
                "SpGEMM2": est_spgemm2(factory, tns_a, tns_b),
                "Triangle Counting": est_triangle(factory, tns_a),
            }
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
        "block_count": BLOCK_COUNT,
        "seed": SEED,
        "results": results,
        "geomean_q_error": {
            model: round(float(np.exp(np.mean(np.log(errors)))), 6)
            for model, errors in q_errors.items()
        },
    }


def test_statistics_accuracy(accuracy_results, file_regression):
    file_regression.check(
        json.dumps(accuracy_results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        extension=".json",
    )


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
        ax.set_xticks(x, list(DATASETS))
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


@pytest.mark.parametrize("view", ["accuracy", "geomean"])
def test_statistics_accuracy_plot(accuracy_results, image_regression, view):
    if view == "accuracy":
        fig = plot_accuracy(accuracy_results["results"])
    else:
        fig = plot_geomean(accuracy_results["geomean_q_error"])
    with BytesIO() as output:
        fig.savefig(output, format="png", dpi=100)
        image_regression.check(output.getvalue())
