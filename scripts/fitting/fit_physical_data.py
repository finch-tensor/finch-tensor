"""
Fit a CostModel to the examples which coalesce_physical_data collected
into EXAMPLES, with the basis of physical_bases named by --basis, or that of
the stage of the examples, and write it to model.json if given. A model of
the same basis reads model.json back with from_json. Each example is
featurized as it was timed: the stages below make their choices with the
models of --model, as time_physical_data made them, so no example is compiled
or run, and the stats are counted exactly from the Kronecker blocks of its
inputs, so none is materialized. The formats of a variant enter only through
the bases, as in the optimizers, which see the supports of their inputs. The
loop stage orders loops before the workspacer, the formatter and the protocol
optimizer run, so its features come from the stats of its inputs alone,
whatever their formats, as the loop orderer prices them, and it chooses
nothing. The variants of each example are every option of one decision, so
the model is also scored by its regret: the runtime of the variant it
predicts to be fastest over that of the fastest, for each example.
To show how the fit improves with data, a held-out tenth of the examples
scores models fitted to growing subsets of the rest, and the relative R^2 and
the median regret of each are printed and plotted next to EXAMPLES, to
NAME.learning_curve.png. With --tune-cutoff, the region cutoff of the classes
of the writes of the loop basis is tuned first, as tune_cutoff tunes it, and
written with the model, which parse_models reads it back from. The model of
the protocol stage is the protocol cost model the physical optimizers use,
read from src/finch/autoschedule/cost_models/protocol_cost_model.json, so a
fit which should ship is written there.

    pixi run -e fitting python -m scripts.fitting.fit_physical_data \\
        EXAMPLES [--basis NAME] [--model NAME=PATH]... [--seed S] \\
        [--tune-cutoff] [model.json]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from finch.autoschedule.cost_models import CostModel, WriteClasses
from scripts.fitting.generation import record_inputs
from scripts.fitting.physical_bases import (
    BASES,
    TUNING_BASIS,
    featurize_record,
    loop_basis,
    parse_models,
    stratified,
)


def r_squared(predicted, actual) -> float:
    """
    The R^2 of predicted under relative error, which models are fitted to: one
    minus its squared relative error over that of the best constant, so that
    the longest runtimes don't outweigh the rest.
    """
    actual = np.asarray(actual)
    inverse = 1 / actual
    constant = inverse.sum() / (inverse**2).sum()
    residual = ((predicted * inverse - 1) ** 2).sum()
    return float(1 - residual / ((constant * inverse - 1) ** 2).sum())


def by_example(examples) -> dict[str, list[int]]:
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
    The relative R^2 and median regret on a held-out tenth of the examples of
    models fitted to nested, geometrically growing subsets of the others.
    Examples are held out whole, with all of their variants.
    """
    groups = by_example(examples)
    names = rng.permutation(list(groups))
    n_test = max(2, len(names) // 10)
    test = [i for name in names[:n_test] for i in groups[name]]
    train_names = names[n_test:]
    untrained = CostModel(basis, np.zeros(basis.n_features))
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


def plot_learning_curve(sizes, scores, medians, n_test, n_features, path) -> None:
    # Only the fitting environment has matplotlib, and only plots need it.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatter

    # R^2 and regret share the training sizes but not a scale, so each gets a
    # panel, and the legend sits above both rather than over the lines.
    fig, (ax, regret_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6), layout="constrained"
    )
    # A model fitted to a few programs can score far below zero, so scores
    # below -1 are drawn at -1 and marked.
    sizes, scores = np.asarray(sizes), np.asarray(scores)
    clipped = np.maximum(scores, -1)
    ax.plot(sizes, clipped, marker="o", label="relative $R^2$")
    low = scores < -1
    if low.any():
        ax.plot(sizes[low], clipped[low], "v", color="C3", label="relative $R^2 < -1$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f"relative $R^2$ on\n{n_test} held-out examples")
    regret_ax.plot(sizes, medians, marker="s", color="C2", label="median regret")
    regret_ax.set_yscale("log")
    # Regrets sit near 1, where scientific notation hides the digits that differ.
    regret_ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    regret_ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    regret_ax.set_ylabel("median regret of\nthe held-out examples")
    regret_ax.set_xscale("log")
    regret_ax.set_xlabel("# examples trained on")
    ax.axvline(n_features, color="gray", linestyle="--", label="# features")
    regret_ax.axvline(n_features, color="gray", linestyle="--")
    for panel in (ax, regret_ax):
        panel.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    regret_handles, regret_labels = regret_ax.get_legend_handles_labels()
    fig.legend(
        handles + regret_handles,
        labels + regret_labels,
        loc="outside upper center",
        ncols=len(labels + regret_labels),
        frameon=False,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def featurize(records, name, models, basis=None):
    features, runtimes, examples = [], [], []
    for i, record in enumerate(records, 1):
        rows = featurize_record(record, record_inputs(record), name, models, basis)
        timed = record["runtimes"]
        for index, variant_rows in enumerate(rows):
            if str(index) in timed:
                features.append(variant_rows)
                runtimes.append(timed[str(index)])
                examples.append((record["name"], index))
        print(f"[{i}/{len(records)}] {record['name']}: {len(rows)} rows", flush=True)
    return features, runtimes, examples


def cross_validated_regret(basis, sums, runtimes, examples, seed, folds=5) -> float:
    """
    The mean log regret of models of basis over folds of the examples, each
    held out in turn and priced by a model fitted to the rest, where sums are
    the summed rows of each variant and runtimes its runtime, as numpy arrays.
    The folds are drawn with seed, so models compared on them see the same.
    """
    groups = by_example(examples)
    names = list(np.random.default_rng(seed).permutation(list(groups)))
    logs = []
    for k in range(folds):
        test = set(names[k::folds])
        train = [i for name in names if name not in test for i in groups[name]]
        model = CostModel(basis, np.zeros(basis.n_features)).fit(
            sums[train][:, None, :], runtimes[train]
        )
        predicted = sums @ model.coefficients
        for name in test:
            indices = groups[name]
            if len(indices) > 1:
                chosen = indices[int(np.argmin(predicted[indices]))]
                logs.append(np.log(runtimes[chosen] / runtimes[indices].min()))
    return float(np.mean(logs)) if logs else 0.0


def tune_cutoff(raw, runtimes, examples, seed) -> WriteClasses:
    """
    The region cutoff of WriteClasses under which models of the loop basis,
    fitted to raw, the rows of TUNING_BASIS of each variant, have the least
    cross_validated_regret. The cutoff is tried on a grid of half powers of
    two, between the power of two sizes of the regions, up to the largest
    region, and past it, where no row is above it.
    """
    runtimes = np.asarray(runtimes)
    stacked = np.concatenate(raw)
    starts = np.cumsum([0, *(len(rows) for rows in raw[:-1])])
    regions = stacked[:, TUNING_BASIS.feature_names.index("scattered_region")]
    top = math.ceil(math.log2(max(regions.max(), 1.0)))
    best, best_score = WriteClasses(math.inf), math.inf
    for cutoff in [*2.0 ** (np.arange(top) + 0.5), math.inf]:
        classes = WriteClasses(float(cutoff))
        sums = np.add.reduceat(stratified(stacked, classes), starts, axis=0)
        score = cross_validated_regret(
            loop_basis(classes), sums, runtimes, examples, seed
        )
        print(f"region cutoff {cutoff:.4g}: mean log regret {score:.4f}", flush=True)
        if score < best_score:
            best, best_score = classes, score
    return best


def importances(basis, model, features, runtimes, examples, seed):
    """
    For each feature of basis which some variant has, its mean share of the
    runtime model predicts for each variant, and how much its
    cross_validated_regret grows when the feature is left out and the models
    are fitted again, most first.
    """
    sums = np.array([rows.sum(axis=0) for rows in features])
    runtimes = np.asarray(runtimes)
    predicted = sums @ model.coefficients
    shares = np.mean(
        sums * model.coefficients / np.maximum(predicted, 1e-300)[:, None], axis=0
    )
    base = cross_validated_regret(basis, sums, runtimes, examples, seed)
    table = []
    for feature, name in enumerate(basis.feature_names):
        if not sums[:, feature].any():
            continue
        dropped = sums.copy()
        dropped[:, feature] = 0.0
        loss = cross_validated_regret(basis, dropped, runtimes, examples, seed)
        table.append((name, shares[feature], loss - base))
    table.sort(key=lambda row: -row[2])
    return base, table


def fit(data: Path, name, models, output: Path | None, seed, tune=False) -> None:
    records = json.loads(data.read_text())["examples"]
    if name is None:
        stages = {record["stage"] for record in records}
        if len(stages) != 1:
            raise ValueError(
                f"Expected examples of one stage, got {sorted(stages)}; pass --basis"
            )
        (name,) = stages
    classes = None
    if tune:
        if name != "loop":
            raise ValueError(f"Only the loop basis has a cutoff to tune, not {name}")
        raw, runtimes, examples = featurize(records, name, models, TUNING_BASIS)
        classes = tune_cutoff(raw, runtimes, examples, seed)
        print(f"tuned region cutoff {classes.region:.4g}", flush=True)
        basis = loop_basis(classes)
        features = [stratified(rows, classes) for rows in raw]
    else:
        basis = BASES[name]
        features, runtimes, examples = featurize(records, name, models)
    groups = by_example(examples)
    print(f"fitting to {len(features)} variants of {len(groups)} examples", flush=True)
    model = CostModel(basis, np.zeros(basis.n_features)).fit(features, runtimes)
    width = max(map(len, basis.feature_names))
    for feature, coefficient in zip(
        basis.feature_names, model.coefficients, strict=True
    ):
        print(f"{feature:{width}} {coefficient:.3e}")
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
        base, table = importances(basis, model, features, runtimes, examples, seed)
        print(
            f"feature importance: the mean share of the predicted runtime, and the "
            f"growth of the cross-validated mean log regret {base:.4f} without it"
        )
        for feature, share, loss in table:
            print(f"{feature:{width}} {share:8.1%} {loss:+9.4f}")
    if len(groups) < 3:
        print("too few examples for a learning curve")
    else:
        rng = np.random.default_rng(seed)
        sizes, scores, medians, n_test = learning_curve(
            basis, features, runtimes, examples, rng
        )
        print(f"relative R^2 and median regret on {n_test} held-out examples:")
        for size, score, median in zip(sizes, scores, medians, strict=True):
            print(f"{size:8d} {score:8.3f} {median:8.2f}")
        plot = data.parent / f"{name}.learning_curve.png"
        plot_learning_curve(sizes, scores, medians, n_test, basis.n_features, plot)
    if output is not None:
        saved = model.to_json()
        if classes is not None:
            # parse_models classes the writes of the model by this cutoff.
            saved["region_cutoff"] = classes.region
        output.write_text(json.dumps(saved, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path, metavar="EXAMPLES")
    parser.add_argument(
        "--basis", choices=BASES, help="the basis to fit; the stage's by default"
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=PATH",
        help="a fitted model of a stage below, which may be repeated",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed which holds examples out"
    )
    parser.add_argument(
        "--tune-cutoff",
        action="store_true",
        help="tune the region cutoff of the classes of writes of the loop basis",
    )
    parser.add_argument("output", type=Path, nargs="?", metavar="model.json")
    args = parser.parse_args()
    fit(
        args.data,
        args.basis,
        parse_models(args.model),
        args.output,
        args.seed,
        args.tune_cutoff,
    )


if __name__ == "__main__":
    main()
