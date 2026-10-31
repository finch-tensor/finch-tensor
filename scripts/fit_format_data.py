"""
Fit a LinearCostModel of the formatter's basis to the features which
featurize_format_data and the timings which time_format_data wrote to DIR, or
to those of one chunk, and write it to model.json if given. To show how the
fit improves with data, a held-out tenth of the programs scores models fitted
to growing subsets of the rest, and the R^2 of each is printed and plotted to
DIR/learning_curve.png.

    pixi run -e fitting python -m scripts.fit_format_data \
        DIR [--chunk K] [--seed S] [model.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from finch.autoschedule.cost_models import LinearCostModel
from scripts.featurize_format_data import BASIS, KINDS
from scripts.generate_format_data import examples, path


def r_squared(predicted, actual) -> float:
    actual = np.asarray(actual)
    residual = ((actual - predicted) ** 2).sum()
    return float(1 - residual / ((actual - actual.mean()) ** 2).sum())


def learning_curve(features, runtimes, rng, n_points=12) -> tuple[list, list]:
    """
    The R^2 on a held-out tenth of the programs of models fitted to nested,
    geometrically growing subsets of the others.
    """
    order = rng.permutation(len(features))
    n_test = max(2, len(features) // 10)
    test, train = order[:n_test], order[n_test:]
    untrained = LinearCostModel(BASIS, np.zeros(BASIS.n_features))
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


def plot_learning_curve(sizes, scores, n_test, path: Path) -> None:
    # A model fitted to a few programs can score far below zero, so scores
    # below -1 are drawn at -1 and marked.
    sizes, scores = np.asarray(sizes), np.asarray(scores)
    clipped = np.maximum(scores, -1)
    fig, ax = plt.subplots()
    ax.plot(sizes, clipped, marker="o")
    low = scores < -1
    if low.any():
        ax.plot(sizes[low], clipped[low], "v", color="C3", label="R^2 < -1")
    ax.axvline(BASIS.n_features, color="gray", linestyle="--", label="# features")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("# programs trained on")
    ax.set_ylabel(f"R^2 on {n_test} held-out programs")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit(directory: Path, chunk, output: Path | None, seed) -> None:
    features: list[np.ndarray] = []
    runtimes: list[float] = []
    for example in examples(directory, chunk):
        rows = json.loads(path(directory, example, "features.json").read_text())
        features.append(np.array(rows["features"]).reshape(-1, BASIS.n_features))
        timing = json.loads(path(directory, example, "timing.json").read_text())
        runtimes.append(timing["runtime"])
    print(f"fitting to {len(features)} programs", flush=True)
    model = LinearCostModel(BASIS, np.zeros(BASIS.n_features)).fit(features, runtimes)
    start = 0
    for kind, basis in KINDS.by_kind.items():
        coefficients = model.coefficients[start : start + basis.n_features]
        print(kind, np.array2string(coefficients, precision=3))
        start += basis.n_features
    predicted = np.array([model.predict(rows) for rows in features])
    error = np.abs(predicted - runtimes) / np.asarray(runtimes)
    print(f"median relative error {np.median(error):.2f}")
    if len(features) < 3:
        print("too few programs for a learning curve")
    else:
        sizes, scores = learning_curve(features, runtimes, np.random.default_rng(seed))
        n_test = len(features) - sizes[-1]
        print(f"R^2 on {n_test} held-out programs:")
        for size, score in zip(sizes, scores, strict=True):
            print(f"{size:8d} {score:8.3f}")
        plot_learning_curve(sizes, scores, n_test, directory / "learning_curve.png")
    if output is not None:
        output.write_text(json.dumps(model.to_json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk", type=int, help="the only chunk to fit to")
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed which holds programs out"
    )
    parser.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    fit(args.directory, args.chunk, args.model, args.seed)


if __name__ == "__main__":
    main()
