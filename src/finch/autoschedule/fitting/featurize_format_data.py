"""
Extract the basis features of the programs which generate_format wrote to DIR,
or of those of one chunk, and write them to DIR as JSON. The features come
from each program, its formats of w and C, and the stats of its inputs, so
they are extracted without compiling or running it, and a new basis needs no
new timings.

    pixi run -e test-julia python -m finch.autoschedule.fitting.featurize_format \
        DIR [--chunk K]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finch.autoschedule.cost_models import ByKind, ElementLevelOption, LevelBasis
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.fitting.generate_format import examples
from finch.autoschedule.fitting.time_format_data import load
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.finch_logic import LogicLoader
from finch.symbolic import UnvalidatedForm

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


def features(plan, formats, bindings) -> np.ndarray:
    """The basis features of plan, as the executor would format it."""
    loader = FeatureLoader(formats)
    LogicExecutor(loader, stats_factory=DCStatsFactory())(plan, bindings)
    return loader.features


def featurize(directory: Path, chunk) -> None:
    for example in examples(directory, chunk):
        rows = features(*load(directory, example))
        (directory / f"{example}.features.json").write_text(
            json.dumps({"features": rows.tolist()})
        )
        print(example, flush=True)


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
