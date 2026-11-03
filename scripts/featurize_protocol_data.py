"""
Extract the features of bases of protocol_bases, by default level, of the
programs which generate_protocol_data wrote to DIR, or of those of one chunk, and
write them to DIR/data as JSON under the name of each basis, as
Basis.features_to_json does. The features come from each program, its formats
of A, B, and C, and the stats of its inputs, so they are extracted without
compiling or running it, and a new basis needs no new timings. The inputs are
copied into their formats by Finch.jl once for every basis, and each value they
store is set to 1, so that their stats count the values each format stores. The
stats are exact, and memoized.

    pixi run -e fitting python -m scripts.featurize_protocol_data \
        DIR [--basis NAME]... [--chunk K] [--resume]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finch.algebra import Tensor
from finch.autoschedule import protocolize
from finch.autoschedule.cost_models import Basis, ElementLevelOption, LevelOption
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.memo_stats import MemoStatsFactory
from finch.compile_jl.interop import jl_tensor_to_python
from finch.finch_logic import LogicLoader
from finch.symbolic import UnvalidatedForm
from finch.tensor import ElementLevel
from scripts.generate_protocol_data import examples, load, path
from scripts.protocol_bases import BASES, features_kind
from scripts.time_protocol_data import reformat


class FeatureLoader(UnvalidatedForm, LogicLoader):
    """
    A loader which records the features of each of bases of the last program
    it loads, with the level options of formats, and loads a kernel which does
    nothing.
    """

    def __init__(self, bases: dict[str, Basis], formats):
        self.bases = bases
        self.formats = {
            alias: [*options, ElementLevelOption()]
            for alias, options in formats.items()
        }
        self.features: dict[str, np.ndarray] = {}

    def lower(self, prgm, bindings, stats, stats_factory):
        # Finch.jl read each level with its default protocol.
        prgm = protocolize(prgm, bindings)
        self.features = {
            name: basis.get_features(prgm, self.formats, stats, stats_factory)
            for name, basis in self.bases.items()
        }
        return SimpleNamespace(main=lambda *args: ()), bindings


def stored(tns: Tensor, options: list[LevelOption]) -> Tensor:
    """A copy of tns with the level options, each value of which it stores is 1."""
    # A scalar has no levels to format, and Finch.jl returns it as a Scalar.
    copy = jl_tensor_to_python(reformat(tns, options).raw_julia_obj) if options else tns
    lvl = copy.lvl
    while True:
        match lvl:
            case ElementLevel(_, val):
                val.arr[:] = 1
                return copy
            case _:
                lvl = lvl.lvl


def features(bases, plan, formats, bindings) -> dict[str, np.ndarray]:
    """The features of each of bases of plan, as the executor would format it."""
    loader = FeatureLoader(bases, formats)
    # The bases build the same projections many times, so the exact stats of
    # each expression are memoized.
    stats_factory = MemoStatsFactory(ExactStatsFactory())
    LogicExecutor(loader, stats_factory=stats_factory)(plan, bindings)
    return loader.features


def featurize_example(directory: Path, example, names) -> dict[str, np.ndarray]:
    """
    The features of each basis of names of example, which are written to its
    features files.
    """
    plan, formats, bindings = load(directory, example)
    bindings = {tns: stored(arg, formats[tns]) for tns, arg in bindings.items()}
    bases = {name: BASES[name] for name in names}
    rows = features(bases, plan, formats, bindings)
    for name, basis in bases.items():
        path(directory, example, features_kind(name)).write_text(
            json.dumps(basis.features_to_json([rows[name]]))
        )
    return rows


def featurize(directory: Path, names, chunk, resume) -> None:
    # Each example is featurized with the bases it lacks, or all if not resuming.
    todo = []
    for example in examples(directory, chunk):
        missing = [
            name
            for name in names
            if not (resume and path(directory, example, features_kind(name)).exists())
        ]
        if missing:
            todo.append((example, missing))
    for i, (example, missing) in enumerate(todo, 1):
        rows = featurize_example(directory, example, missing)
        counts = ", ".join(f"{name} {len(rows[name])} rows" for name in missing)
        print(f"[{i}/{len(todo)}] {example}: {counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--basis",
        action="append",
        choices=BASES,
        help="a basis to extract, which may be repeated; level by default",
    )
    parser.add_argument("--chunk", type=int, help="the only chunk to featurize")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="extract only the bases which each program lacks features of",
    )
    args = parser.parse_args()
    names = list(dict.fromkeys(args.basis or ["level"]))
    featurize(args.directory, names, args.chunk, args.resume)


if __name__ == "__main__":
    main()
