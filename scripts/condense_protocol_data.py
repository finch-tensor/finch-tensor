"""
Collect the features of a basis of protocol_bases which featurize_protocol_data
and the timings which time_protocol_data wrote to DIR/data, or those of one
chunk, into one JSON file, DIR/NAME.json by default, as Basis.features_to_json
does, with the names of the programs and their runtimes, so fit_protocol_data
needs neither the separate files nor the definition of the basis. Programs
which lack either are left out, and features of a basis whose definition has
since changed are an error.

    pixi run -e fitting python -m scripts.condense_protocol_data \
        DIR [--basis NAME] [--chunk K] [data.json]
"""

import argparse
import json
from pathlib import Path

from scripts.generate_protocol_data import examples, path
from scripts.protocol_bases import BASES, features_kind


def condense(directory: Path, name, chunk, output: Path | None) -> None:
    basis, kind = BASES[name], features_kind(name)
    programs = examples(directory, chunk)
    names, features, runtimes = [], [], []
    for example in programs:
        features_path = path(directory, example, kind)
        timing_path = path(directory, example, "timing.json")
        if not (features_path.exists() and timing_path.exists()):
            continue
        try:
            features += basis.features_from_json(json.loads(features_path.read_text()))
        except ValueError as error:
            raise ValueError(f"{features_path}: featurize it again") from error
        names.append(example)
        runtimes.append(json.loads(timing_path.read_text())["runtime"])
    data = {**basis.features_to_json(features), "examples": names, "runtimes": runtimes}
    output = directory / f"{name}.json" if output is None else output
    output.write_text(json.dumps(data))
    print(f"wrote {len(names)} of {len(programs)} programs to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--basis", choices=BASES, default="level")
    parser.add_argument("--chunk", type=int, help="the only chunk to condense")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    condense(args.directory, args.basis, args.chunk, args.output)


if __name__ == "__main__":
    main()
