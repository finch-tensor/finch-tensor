"""
Extract the features of bases of physical_bases, by default that of the stage
of DIR, of the variants of the programs which generate_physical_data wrote to
DIR, or of those of one chunk, and write them to DIR/data as JSON under the
name of each basis, as Basis.features_to_json does, with one entry for each
variant. The stages below make their choices with the models of --model, as
time_physical_data makes them, so each variant is featurized as it was timed,
without compiling or running it, and a new basis needs no new timings. The
inputs are copied into their formats by Finch.jl, once for each format of
each, and each value they store is set to 1, so that their stats count the
values each format stores. The loop stage orders loops before its inputs are
formatted, so its features are those of their supports. The stats are exact,
and memoized across the variants of each example.

    pixi run -e fitting python -m scripts.fitting.featurize_physical_data \\
        DIR [--model NAME=PATH]... [--basis NAME]... [--chunk K] [--resume]
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finch.algebra import Tensor
from finch.autoschedule import protocolize
from finch.autoschedule.cost_models import Basis, ElementLevelOption, LevelOption
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.memo_stats import MemoStatsFactory
from finch.compile_jl.interop import jl_tensor_to_python
from finch.finch_logic import LogicLoader
from finch.symbolic import UnvalidatedForm
from finch.tensor import ElementLevel
from scripts.fitting.generate_physical_data import examples, load, path
from scripts.fitting.physical_bases import (
    BASES,
    features_kind,
    input_perms,
    parse_models,
    resolve,
    variant_plan,
)
from scripts.fitting.time_physical_data import Reformatter


class FeatureLoader(UnvalidatedForm, LogicLoader):
    """
    A loader which records the features of each of bases of the last program
    it loads, with the level options of formats, and loads a kernel which does
    nothing. Unless protocols is false, each read without a protocol takes the
    default of its level, as Finch.jl reads it.
    """

    def __init__(self, bases: dict[str, Basis], formats, protocols=True):
        self.bases = bases
        self.formats = {
            alias: [*options, ElementLevelOption()]
            for alias, options in formats.items()
        }
        self.protocols = protocols
        self.features: dict[str, np.ndarray] = {}

    def lower(self, prgm, bindings, stats, stats_factory):
        if self.protocols:
            prgm = protocolize(prgm, bindings)
        self.features = {
            name: basis.get_features(prgm, self.formats, stats, stats_factory)
            for name, basis in self.bases.items()
        }
        return SimpleNamespace(main=lambda *args: ()), bindings


def stored(tns: Tensor, options: list[LevelOption], perm=()) -> Tensor:
    """
    A copy of tns with the level options, storing its fields in the order of
    perm, each value of which it stores is 1.
    """
    from scripts.fitting.time_physical_data import reformat

    # A scalar has no levels to format, and Finch.jl returns it as a Scalar.
    copy = (
        jl_tensor_to_python(reformat(tns, options, perm).raw_julia_obj)
        if options
        else deepcopy(tns)
    )
    lvl = copy.lvl
    while True:
        match lvl:
            case ElementLevel(_, val):
                val.arr[:] = 1
                return copy
            case _:
                lvl = lvl.lvl


def features(
    bases, plan, formats, bindings, stats_factory, protocols=True
) -> dict[str, np.ndarray]:
    """The features of each of bases of plan, as the executor would format it."""
    loader = FeatureLoader(bases, formats, protocols)
    LogicExecutor(loader, stats_factory=stats_factory)(plan, bindings)
    return loader.features


def featurize_example(directory: Path, example, names, models) -> dict[str, list]:
    """
    The features of each basis of names of each variant of example, which are
    written to its features files.
    """
    record, inputs = load(directory, example)
    stage = record["stage"]
    bases = {name: BASES[name] for name in names}
    # The choices of the stages below are made from the supports of the
    # inputs, as they were timed, and the features from what the formats
    # store, except in the loop stage, which sees only the supports.
    supports = Reformatter()
    stores = Reformatter(stored)
    choices_factory = DCStatsFactory()
    # The bases build the same projections many times, so the exact stats of
    # each expression are memoized, across the variants of the example.
    stats_factory = MemoStatsFactory(ExactStatsFactory())
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for variant in record["variants"]:
        plan, formats, bindings = resolve(
            record, variant, inputs, choices_factory, models, supports
        )
        if stage == "loop":
            # The loop orderer prices the concordant program before the
            # formatter and the protocol optimizer run, without protocols,
            # from the supports of its inputs, which resolve stored in loop
            # order.
            plan = variant_plan(record, variant)
        else:
            perms = input_perms(record, plan)
            bindings = {
                tns: stores(arg, formats[tns], perms[tns])
                for tns, arg in inputs.items()
                if tns in perms
            }
        found = features(
            bases, plan, formats, bindings, stats_factory, protocols=stage != "loop"
        )
        for name in names:
            rows[name].append(found[name])
    for name, basis in bases.items():
        path(directory, example, features_kind(name)).write_text(
            json.dumps(basis.features_to_json(rows[name]))
        )
    return rows


def featurize(directory: Path, names, models, chunk, resume) -> None:
    # Each example is featurized with the bases it lacks, or all if not resuming.
    todo = []
    for example in examples(directory, chunk):
        record = json.loads(path(directory, example, "program.json").read_text())
        wanted = names or [record["stage"]]
        missing = [
            name
            for name in wanted
            if not (resume and path(directory, example, features_kind(name)).exists())
        ]
        if missing:
            todo.append((example, missing))
    for i, (example, missing) in enumerate(todo, 1):
        rows = featurize_example(directory, example, missing, models)
        counts = ", ".join(
            f"{name} {sum(map(len, rows[name]))} rows" for name in missing
        )
        print(f"[{i}/{len(todo)}] {example}: {counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=PATH",
        help="a fitted model of a stage below, which may be repeated",
    )
    parser.add_argument(
        "--basis",
        action="append",
        choices=BASES,
        help="a basis to extract, which may be repeated; the stage's by default",
    )
    parser.add_argument("--chunk", type=int, help="the only chunk to featurize")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="extract only the bases which each program lacks features of",
    )
    args = parser.parse_args()
    names = list(dict.fromkeys(args.basis or []))
    featurize(args.directory, names, parse_models(args.model), args.chunk, args.resume)


if __name__ == "__main__":
    main()
