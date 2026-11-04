"""
Extract the features of bases of physical_bases, by default that of the stage
of DIR, of the variants of the programs which generate_physical_data wrote to
DIR, or of those of one chunk, and write them to DIR/data as JSON under the
name of each basis, as Basis.features_to_json does, with one entry for each
variant. The stages below make their choices with the models of --model, as
time_physical_data makes them, so each variant is featurized as it was timed,
without compiling or running it, and a new basis needs no new timings. The
stats are counted exactly from the Kronecker blocks of the inputs, so no input
is materialized. The formats of a variant enter only through the bases, as in
the optimizers, which see the supports of their inputs. The loop stage orders
loops before its inputs are formatted, so it reads them without protocols.

    pixi run -e fitting python -m scripts.fitting.featurize_physical_data \\
        DIR [--model NAME=PATH]... [--basis NAME]... [--chunk K] [--resume]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finch.autoschedule import protocolize
from finch.autoschedule.cost_models import Basis, ElementLevelOption
from scripts.fitting.generate_physical_data import examples, load, path, write_json
from scripts.fitting.kronecker_stats import KroneckerStatsFactory
from scripts.fitting.physical_bases import (
    BASES,
    features_kind,
    input_fields,
    input_ftype,
    parse_models,
    read_inputs,
    resolve,
    variant_plan,
)


def features(
    bases: dict[str, Basis], plan, formats, inputs, stats_factory, protocols=True
) -> dict[str, np.ndarray]:
    """
    The features of each of bases of plan, with the level options of formats,
    from the stats of inputs, which store their fields in the order plan reads
    them. Unless protocols is false, each read without a protocol takes the
    default of its level, as Finch.jl reads it.
    """
    fields = input_fields(plan)
    stats = {tns: stats_factory(tensor, fields[tns]) for tns, tensor in inputs.items()}
    if protocols:
        ftypes = {
            tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()
        }
        plan = protocolize(plan, ftypes)
    formats = {
        alias: [*options, ElementLevelOption()] for alias, options in formats.items()
    }
    return {
        name: basis.get_features(plan, formats, stats, stats_factory)
        for name, basis in bases.items()
    }


def featurize_example(directory: Path, example, names, models) -> dict[str, list]:
    """
    The features of each basis of names of each variant of example, which are
    written to its features files.
    """
    record, inputs = load(directory, example)
    stage = record["stage"]
    bases = {name: BASES[name] for name in names}
    stats_factory = KroneckerStatsFactory(record["block_bits"])
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for variant in record["variants"]:
        plan, formats = resolve(record, variant, inputs, models)
        if stage == "loop":
            # The loop orderer prices the concordant program before the
            # formatter and the protocol optimizer run, without protocols.
            plan = variant_plan(record, variant)
        found = features(
            bases,
            plan,
            formats,
            read_inputs(record, plan, inputs),
            stats_factory,
            protocols=stage != "loop",
        )
        for name in names:
            rows[name].append(found[name])
    for name, basis in bases.items():
        write_json(
            path(directory, example, features_kind(name)),
            basis.features_to_json(rows[name]),
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
