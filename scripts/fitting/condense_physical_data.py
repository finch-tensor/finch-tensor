"""
Collect the features of a basis of physical_bases which featurize_physical_data
and the timings which time_physical_data wrote to DIR/data, or those of one
chunk, into one JSON file, DIR/NAME.json by default, as Basis.features_to_json
does, with the name of the basis and, for each variant, the name of its
example, its index, and its runtime, so fit_physical_data needs no separate
files. Variants which lack either are left out, and features of a basis whose
definition has since changed are an error.

    pixi run -e fitting python -m scripts.fitting.condense_physical_data \\
        DIR [--basis NAME] [--chunk K] [data.json]
"""

import argparse
import json
from pathlib import Path

from scripts.fitting.generate_physical_data import examples, path
from scripts.fitting.physical_bases import BASES, features_kind


def condense(directory: Path, name, chunk, output: Path | None) -> None:
    basis, kind = BASES[name], features_kind(name)
    programs = examples(directory, chunk)
    # Few examples are timed, so only theirs are looked up, which is much
    # faster on a parallel filesystem than looking up each example's files.
    timed_examples = examples(directory, chunk, "timing.json")
    names, features, runtimes = [], [], []
    for example in timed_examples:
        features_path = path(directory, example, kind)
        timing_path = path(directory, example, "timing.json")
        if not features_path.exists():
            continue
        try:
            rows = basis.features_from_json(json.loads(features_path.read_text()))
        except ValueError as error:
            raise ValueError(f"{features_path}: featurize it again") from error
        timed = json.loads(timing_path.read_text())["runtimes"]
        for index, variant_rows in enumerate(rows):
            if str(index) not in timed:
                continue
            names.append([example, index])
            features.append(variant_rows)
            runtimes.append(timed[str(index)])
    data = {
        **basis.features_to_json(features),
        "name": name,
        "examples": names,
        "runtimes": runtimes,
    }
    output = directory / f"{name}.json" if output is None else output
    output.write_text(json.dumps(data))
    print(
        f"wrote {len(names)} variants of {len({e for e, _ in names})} of "
        f"{len(programs)} programs to {output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--basis", choices=BASES, default="protocol")
    parser.add_argument("--chunk", type=int, help="the only chunk to condense")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    condense(args.directory, args.basis, args.chunk, args.output)


if __name__ == "__main__":
    main()
