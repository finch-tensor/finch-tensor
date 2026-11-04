"""
Grow a local collection of examples for fitting the cost model of a stage of
physical_bases in DIR, one at a time, and watch the fit improve. Programs are
drawn as generate_physical_data draws them, and the variants of each example
are timed as time_physical_data times them and featurized with the basis of
the stage as featurize_physical_data does, so DIR has the layout of the
cluster's steps, and condense_physical_data and fit_physical_data can read it
too. A fraction of the examples, which a hash of their names picks so that it
survives restarts, is held out. After each example, a LinearCostModel is fitted
to the others, and its predictions of the held-out runtimes and its held-out
R^2 as the collection grows are drawn in a window, which only responds between
examples, and to DIR/NAME.live.png. The examples already in DIR start the
collection, and those not yet timed or featurized are finished first. Stop
with Ctrl-C, or by closing the window.

    pixi run -e fitting python -m scripts.fitting.live_fit_physical_data \\
        DIR [--stage S] [--model NAME=PATH]... [--holdout F] [--programs N] \\
        [--samples N] [--keep N] [--repeat N] [--variants N] \\
        [--lg-max-space N] [--lg-max-dim N] [--density-bias D] \\
        [--size-bias D] [--seed S] [--no-window]
"""

import argparse
import contextlib
import itertools
import json
import zlib
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from finch.autoschedule.cost_models import LinearCostModel
from scripts.fitting.featurize_physical_data import featurize_example
from scripts.fitting.fit_physical_data import draw_learning_curve, r_squared
from scripts.fitting.generate_physical_data import (
    examples,
    path,
    program_inputs,
    write_example,
)
from scripts.fitting.physical_bases import BASES, STAGES, features_kind, parse_models
from scripts.fitting.time_physical_data import Timer


def held_out(example, fraction) -> bool:
    return zlib.crc32(example.encode()) < fraction * 2**32


def is_collected(directory: Path, name, example) -> bool:
    return all(
        path(directory, example, kind).exists()
        for kind in ("timing.json", features_kind(name))
    )


def collect(
    directory: Path, name, example, timer: Timer, models
) -> list[tuple[np.ndarray, float]]:
    """
    The features of the basis name and the runtime of each variant of example,
    which are timed and featurized if they haven't been, or if the features
    are of an earlier definition of the basis.
    """
    basis = BASES[name]
    runtimes = timer.time_example(directory, example, resume=True)
    features_path = path(directory, example, features_kind(name))
    rows = None
    # Features of an earlier definition of the basis are an error to read.
    if features_path.exists():
        with contextlib.suppress(ValueError):
            rows = basis.features_from_json(json.loads(features_path.read_text()))
    if rows is None:
        rows = featurize_example(directory, example, [name], models)[name]
    return [(rows[i], runtimes[str(i)]) for i in range(len(rows)) if str(i) in runtimes]


def new_examples(directory: Path, stage, programs, generate_args, rng):
    """Draw programs, and write their examples to directory after those in it."""
    (directory / "data").mkdir(parents=True, exist_ok=True)
    taken = [int(example.split("-")[1]) for example in examples(directory, 0)]
    index = itertools.count(max(taken, default=-1) + 1)
    for _ in range(programs) if programs is not None else itertools.count():
        for record in program_inputs(rng, stage, *generate_args):
            example = f"0000-{next(index):05d}"
            write_example(directory, example, record)
            yield example


class LiveFit:
    """
    The features and runtimes of the variants of the examples collected so
    far, by whether they are held out, and the R^2 on those held out of the
    model fitted to the others each time it's fitted, drawn to plot and, if
    window, to a window.
    """

    def __init__(self, basis, holdout, plot: Path, window):
        self.basis = basis
        self.holdout = holdout
        self.plot = plot
        self.window = window
        self.examples: dict[bool, list[tuple[np.ndarray, float]]] = {
            False: [],
            True: [],
        }
        self.n_examples = 0
        self.history: list[tuple[int, float]] = []
        self.latest: list[tuple[np.ndarray, float]] = []
        if window:
            plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        if window:
            plt.show(block=False)

    def add(self, example, variants) -> None:
        test = held_out(example, self.holdout)
        self.examples[test].extend(variants)
        self.latest = variants if test else []
        self.n_examples += 1

    @property
    def closed(self) -> bool:
        return self.window and not plt.fignum_exists(self.fig.number)

    def update(self) -> str:
        """Fit the model, draw it, and describe it."""
        train, test = self.examples[False], self.examples[True]
        if not train:
            return f"no variants to fit to, {len(test)} held out"
        model = LinearCostModel(self.basis, np.zeros(self.basis.n_features)).fit(
            [features for features, _ in train], [runtime for _, runtime in train]
        )
        predicted = np.array([model.predict(features) for features, _ in test])
        measured = np.array([runtime for _, runtime in test])
        score = r_squared(predicted, measured) if len(test) > 1 else np.nan
        if not np.isnan(score):
            self.history.append((len(train), score))
        self.draw(model, measured, predicted, score)
        return f"fit to {len(train)} variants, R^2 {score:.3f} on {len(test)} held out"

    def draw(self, model, measured, predicted, score) -> None:
        fit_ax, curve_ax = self.axes
        fit_ax.clear()
        curve_ax.clear()
        if len(measured):
            fit_ax.scatter(measured, predicted, s=12, alpha=0.6)
            for features, runtime in self.latest:
                fit_ax.plot(runtime, model.predict(features), "o", color="C3")
            # Nonnegative coefficients can predict zero, which log axes can't
            # show, so the diagonal spans the positive runtimes.
            positive = np.concatenate([measured, predicted[predicted > 0]])
            bounds = [positive.min(), positive.max()]
            fit_ax.plot(bounds, bounds, color="gray", linestyle="--")
            fit_ax.set_xscale("log")
            fit_ax.set_yscale("log")
        fit_ax.set_xlabel("measured runtime (s)")
        fit_ax.set_ylabel("predicted runtime (s)")
        n_train = len(self.examples[False])
        fit_ax.set_title(f"fit to {n_train} variants, R^2 {score:.3f}")
        if self.history:
            sizes, scores = zip(*self.history, strict=True)
            draw_learning_curve(
                curve_ax,
                sizes,
                scores,
                self.basis.n_features,
                "R^2 on the held-out variants",
            )
            curve_ax.set_xlabel("# variants trained on")
        self.fig.savefig(self.plot, dpi=150, bbox_inches="tight")
        if self.window:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


def live_fit(
    directory: Path,
    stage,
    models,
    holdout,
    programs,
    repeat,
    window,
    generate_args,
    rng,
):
    live = LiveFit(BASES[stage], holdout, directory / f"{stage}.live.png", window)
    timer = Timer(models, repeat)
    pending = []
    for example in examples(directory):
        if is_collected(directory, stage, example):
            live.add(example, collect(directory, stage, example, timer, models))
        else:
            pending.append(example)
    print(f"{live.n_examples} examples collected, {len(pending)} pending")
    print(live.update(), flush=True)
    for example in itertools.chain(
        pending, new_examples(directory, stage, programs, generate_args, rng)
    ):
        if live.closed:
            break
        variants = collect(directory, stage, example, timer, models)
        live.add(example, variants)
        best = min(runtime for _, runtime in variants)
        kind = "held out" if held_out(example, holdout) else "trained on"
        print(
            f"{example}: {len(variants)} variants, best {best:.2e}s, {kind}; "
            f"{live.update()}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="protocol")
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=PATH",
        help="a fitted model of a stage below, which may be repeated",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="the fraction of the examples which are held out",
    )
    parser.add_argument(
        "--programs", type=int, help="the programs to draw; unlimited by default"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--variants", type=int, help="the most variants of a program; all by default"
    )
    parser.add_argument("--lg-max-space", type=int, default=20)
    parser.add_argument("--lg-max-dim", type=int, default=10)
    parser.add_argument("--density-bias", type=float, default=3.0)
    parser.add_argument("--size-bias", type=float, default=3.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="only draw to DIR/NAME.live.png",
    )
    args = parser.parse_args()
    if args.keep > args.samples:
        parser.error("--keep may be at most --samples")
    # A seed drawn again after a restart still draws new programs.
    start = len(examples(args.directory))
    rng = np.random.default_rng(None if args.seed is None else (args.seed, start))
    generate_args = (
        args.lg_max_space,
        args.lg_max_dim,
        args.samples,
        args.keep,
        args.variants,
        args.density_bias,
        args.size_bias,
    )
    try:
        live_fit(
            args.directory,
            args.stage,
            parse_models(args.model),
            args.holdout,
            args.programs,
            args.repeat,
            args.window,
            generate_args,
            rng,
        )
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
