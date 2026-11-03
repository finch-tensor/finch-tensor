"""
Fit a LinearCostModel to the features and timings which condense_protocol_data
collected into DATA, of the basis described there, and write it to model.json
if given. To show how the fit improves with data, a held-out tenth of the
programs scores models fitted to growing subsets of the rest, and the R^2 of
each is printed and plotted next to DATA, to DATA.learning_curve.png.

    pixi run -e fitting python -m scripts.fit_protocol_data \
        DATA [--seed S] [model.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from finch.autoschedule.cost_models import Basis, LinearCostModel


def r_squared(predicted, actual) -> float:
    actual = np.asarray(actual)
    residual = ((actual - predicted) ** 2).sum()
    return float(1 - residual / ((actual - actual.mean()) ** 2).sum())


def learning_curve(basis, features, runtimes, rng, n_points=12) -> tuple[list, list]:
    """
    The R^2 on a held-out tenth of the programs of models fitted to nested,
    geometrically growing subsets of the others.
    """
    order = rng.permutation(len(features))
    n_test = max(2, len(features) // 10)
    test, train = order[:n_test], order[n_test:]
    untrained = LinearCostModel(basis, np.zeros(basis.n_features))
    sizes = np.unique(np.geomspace(1, len(train), n_points).round().astype(int))
    scores = []
    for size in sizes:
        subset = train[:size]
        model = untrained.fit(
            [features[i] for i in subset], [runtimes[i] for i in subset]
        )
        predicted = [model.predict(features[i]) for i in test]
        scores.append(r_squared(predicted, [runtimes[i] for i in test]))
    return list(sizes), scores


def draw_learning_curve(ax, sizes, scores, n_features, ylabel) -> None:
    # A model fitted to a few programs can score far below zero, so scores
    # below -1 are drawn at -1 and marked.
    sizes, scores = np.asarray(sizes), np.asarray(scores)
    clipped = np.maximum(scores, -1)
    ax.plot(sizes, clipped, marker="o")
    low = scores < -1
    if low.any():
        ax.plot(sizes[low], clipped[low], "v", color="C3", label="R^2 < -1")
    ax.axvline(n_features, color="gray", linestyle="--", label="# features")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("# programs trained on")
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_learning_curve(sizes, scores, n_test, n_features, path: Path) -> None:
    fig, ax = plt.subplots()
    draw_learning_curve(
        ax, sizes, scores, n_features, f"R^2 on {n_test} held-out programs"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit(data: Path, output: Path | None, seed) -> None:
    record = json.loads(data.read_text())
    basis = Basis.from_json(record["basis"])
    features = basis.features_from_json(record)
    runtimes = record["runtimes"]
    print(f"fitting to {len(features)} programs", flush=True)
    model = LinearCostModel(basis, np.zeros(basis.n_features)).fit(features, runtimes)
    width = max(map(len, basis.feature_names))
    for name, coefficient in zip(basis.feature_names, model.coefficients, strict=True):
        print(f"{name:{width}} {coefficient:.3e}")
    predicted = np.array([model.predict(rows) for rows in features])
    error = np.abs(predicted - runtimes) / np.asarray(runtimes)
    print(f"median relative error {np.median(error):.2f}")
    if len(features) < 3:
        print("too few programs for a learning curve")
    else:
        rng = np.random.default_rng(seed)
        sizes, scores = learning_curve(basis, features, runtimes, rng)
        n_test = len(features) - sizes[-1]
        print(f"R^2 on {n_test} held-out programs:")
        for size, score in zip(sizes, scores, strict=True):
            print(f"{size:8d} {score:8.3f}")
        plot = data.with_suffix(".learning_curve.png")
        plot_learning_curve(sizes, scores, n_test, basis.n_features, plot)
    if output is not None:
        output.write_text(json.dumps(model.to_json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed which holds programs out"
    )
    parser.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    fit(args.data, args.model, args.seed)


if __name__ == "__main__":
    main()
