"""
Extract the basis features of the programs which generate_format_data wrote
to DIR, or of those of one chunk, and write them to DIR/data as JSON. The features
come from each program, its formats of A, B, and C, and the stats of its inputs,
so they are extracted without compiling or running it, and a new basis needs
no new timings. The inputs are copied into their formats by Finch.jl, and each
value they store is set to 1, so that their stats count the values each format
stores.

    pixi run -e fitting python -m scripts.featurize_format_data \
        DIR [--chunk K]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finch.algebra import Tensor
from finch.autoschedule.cost_models import (
    ByKind,
    ElementLevelOption,
    LevelBasis,
    LevelOption,
)
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.compile_jl.interop import jl_tensor_to_python
from finch.finch_logic import LogicLoader
from finch.symbolic import UnvalidatedForm
from finch.tensor import ElementLevel
from scripts.generate_format_data import examples, load, path
from scripts.time_format_data import reformat

BASIS = ByKind(
    {
        option.kind: LevelBasis(type(option))
        for option in (*LEVEL_OPTIONS, ElementLevelOption())
    }
)


class FeatureLoader(UnvalidatedForm, LogicLoader):
    """
    A loader which records the basis features of the last program it loads,
    with the level options of formats, and loads a kernel which does nothing.
    """

    def __init__(self, formats):
        self.formats = {
            alias: [*options, ElementLevelOption()]
            for alias, options in formats.items()
        }
        self.features = np.empty((0, BASIS.n_features))

    def lower(self, prgm, bindings, stats, stats_factory):
        self.features = BASIS.get_features(prgm, self.formats, stats, stats_factory)
        return SimpleNamespace(main=lambda *args: ()), bindings


def stored(tns: Tensor, options: list[LevelOption]) -> Tensor:
    """A copy of tns with the level options, each value of which it stores is 1."""
    copy = jl_tensor_to_python(reformat(tns, options).raw_julia_obj)
    lvl = copy.lvl
    while True:
        match lvl:
            case ElementLevel(_, val):
                val.arr[:] = 1
                return copy
            case _:
                lvl = lvl.lvl


def features(plan, formats, bindings) -> np.ndarray:
    """The basis features of plan, as the executor would format it."""
    loader = FeatureLoader(formats)
    LogicExecutor(loader, stats_factory=DCStatsFactory())(plan, bindings)
    return loader.features


def featurize(directory: Path, chunk) -> None:
    names = examples(directory, chunk)
    for i, example in enumerate(names, 1):
        plan, formats, bindings = load(directory, example)
        bindings = {tns: stored(arg, formats[tns]) for tns, arg in bindings.items()}
        rows = features(plan, formats, bindings)
        path(directory, example, "features.json").write_text(
            json.dumps({"features": rows.tolist()})
        )
        print(f"[{i}/{len(names)}] {example}: {len(rows)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk", type=int, help="the only chunk to featurize")
    args = parser.parse_args()
    featurize(args.directory, args.chunk)


if __name__ == "__main__":
    main()
