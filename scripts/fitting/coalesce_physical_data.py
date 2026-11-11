"""
Collect the programs which a generator wrote to DIR/data and the
timings which time_physical_data wrote there, or those of one chunk, into one
JSON file, DIR/examples.json by default: the record of each timed example,
under its name, with the runtime of each of its variants which was timed, and
whatever else its timing saved of them, such as the formats they ran with, by
its index. The inputs
of an example are drawn from the seeds in its record, so the file is small,
and fit_physical_data featurizes and fits the examples from it alone,
anywhere, without DIR/data.

    pixi run -e fitting python -m scripts.fitting.coalesce_physical_data DIR
"""

import argparse
import json
from pathlib import Path

from scripts.fitting.generation import examples, path


def coalesce(directory: Path, chunk, output: Path | None) -> None:
    # Globbing a missing directory finds nothing, which would coalesce nothing.
    if not (directory / "data").is_dir():
        raise SystemExit(f"{directory}/data is not a directory; check DIR")
    programs = examples(directory, chunk)
    # Few examples are timed, so only theirs are looked up, which is much
    # faster on a parallel filesystem than looking up each example's files.
    records = []
    variants = 0
    for example in examples(directory, chunk, "timing.json"):
        record = json.loads(path(directory, example, "program.json").read_text())
        timing = json.loads(path(directory, example, "timing.json").read_text())
        if not timing["runtimes"]:
            continue
        # Timings written before formats were saved have none.
        records.append({"name": example, **record, "formats": {}, **timing})
        variants += len(timing["runtimes"])
    output = directory / "examples.json" if output is None else output
    output.write_text(json.dumps({"examples": records}) + "\n")
    print(
        f"wrote {variants} timed variants of {len(records)} of {len(programs)} "
        f"programs to {output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk", type=int, help="the only chunk to coalesce")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    coalesce(args.directory, args.chunk, args.output)


if __name__ == "__main__":
    main()
