"""
Fit a LinearCostModel to the features and timings which condense_physical_data
collected into DATA, of the basis of physical_bases named there, and write it
to model.json if given. A model of the same basis reads model.json back with
from_json. The variants of each example are every option of one decision, so
the model is also scored by its regret: the runtime of the variant it predicts
to be fastest over that of the fastest, for each example. To show how the fit
improves with data, a held-out tenth of the examples scores models fitted to
growing subsets of the rest, and the R^2 and the median regret of each are
printed and plotted next to DATA, to DATA.learning_curve.png.

    pixi run -e fitting python -m scripts.fitting.fit_physical_data \\
        DATA [--seed S] [model.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from finch.autoschedule.cost_models import LinearCostModel
from scripts.fitting.physical_bases import BASES


def r_squared(predicted, actual) -> float:
    actual = np.asarray(actual)
    residual = ((actual - predicted) ** 2).sum()
    return float(1 - residual / ((actual - actual.mean()) ** 2).sum())


def by_example(examples) -> dict[str, list[int]]:
    """The indices of the variants of each example, by its name."""
    groups: dict[str, list[int]] = {}
    for i, (example, _) in enumerate(examples):
        groups.setdefault(example, []).append(i)
    return groups


def regrets(model, features, runtimes, groups) -> np.ndarray:
    """
    For each example of several variants, the runtime of the variant model
    predicts to be fastest over that of the fastest.
    """
    result = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        predicted = [model.predict(features[i]) for i in indices]
        chosen = indices[int(np.argmin(predicted))]
        result.append(runtimes[chosen] / min(runtimes[i] for i in indices))
    return np.array(result)


def learning_curve(basis, features, runtimes, examples, rng, n_points=12):
    """
    The R^2 and median regret on a held-out tenth of the examples of models
    fitted to nested, geometrically growing subsets of the others. Examples
    are held out whole, with all of their variants.
    """
    groups = by_example(examples)
    names = rng.permutation(list(groups))
    n_test = max(2, len(names) // 10)
    test = [i for name in names[:n_test] for i in groups[name]]
    train_names = names[n_test:]
    untrained = LinearCostModel(basis, np.zeros(basis.n_features))
    sizes = np.unique(np.geomspace(1, len(train_names), n_points).round().astype(int))
    scores, medians = [], []
    for size in sizes:
        train = [i for name in train_names[:size] for i in groups[name]]
        model = untrained.fit(
            [features[i] for i in train], [runtimes[i] for i in train]
        )
        predicted = [model.predict(features[i]) for i in test]
        scores.append(r_squared(predicted, [runtimes[i] for i in test]))
        held = {name: groups[name] for name in names[:n_test]}
        medians.append(float(np.median(regrets(model, features, runtimes, held))))
    return list(sizes), scores, medians, n_test


def draw_learning_curve(ax, sizes, scores, n_features, ylabel, medians=None) -> None:
    # A model fitted to a few programs can score far below zero, so scores
    # below -1 are drawn at -1 and marked.
    sizes, scores = np.asarray(sizes), np.asarray(scores)
    clipped = np.maximum(scores, -1)
    ax.plot(sizes, clipped, marker="o", label="R^2")
    low = scores < -1
    if low.any():
        ax.plot(sizes[low], clipped[low], "v", color="C3", label="R^2 < -1")
    ax.axvline(n_features, color="gray", linestyle="--", label="# features")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("# examples trained on")
    ax.set_ylabel(ylabel)
    if medians is not None:
        regret_ax = ax.twinx()
        regret_ax.plot(sizes, medians, marker="s", color="C2", label="median regret")
        regret_ax.set_yscale("log")
        regret_ax.set_ylabel("median regret of the held-out examples")
        regret_ax.legend(loc="lower right")
    ax.legend(loc="lower left")


def plot_learning_curve(sizes, scores, medians, n_test, n_features, path) -> None:
    fig, ax = plt.subplots()
    draw_learning_curve(
        ax, sizes, scores, n_features, f"R^2 on {n_test} held-out examples", medians
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit(data: Path, output: Path | None, seed) -> None:
    record = json.loads(data.read_text())
    basis = BASES[record["name"]]
    features = basis.features_from_json(record)
    runtimes = record["runtimes"]
    examples = record["examples"]
    groups = by_example(examples)
    print(f"fitting to {len(features)} variants of {len(groups)} examples", flush=True)
    model = LinearCostModel(basis, np.zeros(basis.n_features)).fit(features, runtimes)
    width = max(map(len, basis.feature_names))
    for name, coefficient in zip(basis.feature_names, model.coefficients, strict=True):
        print(f"{name:{width}} {coefficient:.3e}")
    if features:
        predicted = np.array([model.predict(rows) for rows in features])
        error = np.abs(predicted - runtimes) / np.asarray(runtimes)
        print(f"median relative error {np.median(error):.2f}")
        regret = regrets(model, features, runtimes, groups)
        if len(regret):
            print(
                f"regret over {len(regret)} examples: median {np.median(regret):.2f}, "
                f"90th percentile {np.percentile(regret, 90):.2f}, "
                f"best chosen in {np.mean(regret == 1):.0%}"
            )
    if len(groups) < 3:
        print("too few examples for a learning curve")
    else:
        rng = np.random.default_rng(seed)
        sizes, scores, medians, n_test = learning_curve(
            basis, features, runtimes, examples, rng
        )
        print(f"R^2 and median regret on {n_test} held-out examples:")
        for size, score, median in zip(sizes, scores, medians, strict=True):
            print(f"{size:8d} {score:8.3f} {median:8.2f}")
        plot = data.with_suffix(".learning_curve.png")
        plot_learning_curve(sizes, scores, medians, n_test, basis.n_features, plot)
    if output is not None:
        output.write_text(json.dumps(model.to_json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed which holds examples out"
    )
    parser.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    fit(args.data, args.model, args.seed)


if __name__ == "__main__":
    main()
