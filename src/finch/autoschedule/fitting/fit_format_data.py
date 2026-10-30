"""
Fit a LinearCostModel of the formatter's basis to the features which
featurize_format and the timings which time_format wrote to DIR, or to those
of one chunk, and write it to model.json if given.

    pixi run -e test-julia python -m finch.autoschedule.fitting.fit_format \
        DIR [--chunk K] [model.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finch.autoschedule.cost_models import LinearCostModel
from finch.autoschedule.fitting.featurize_format_data import BASIS
from finch.autoschedule.fitting.generate_format import examples


def fit(directory: Path, chunk, output: Path | None) -> None:
    features: list[np.ndarray] = []
    runtimes: list[float] = []
    for example in examples(directory, chunk):
        rows = json.loads((directory / f"{example}.features.json").read_text())
        features.append(np.array(rows["features"]).reshape(-1, BASIS.n_features))
        timing = json.loads((directory / f"{example}.timing.json").read_text())
        runtimes.append(timing["runtime"])
    model = LinearCostModel(BASIS, np.zeros(BASIS.n_features)).fit(features, runtimes)
    start = 0
    for kind, basis in BASIS.by_kind.items():
        coefficients = model.coefficients[start : start + basis.n_features]
        print(kind, np.array2string(coefficients, precision=3))
        start += basis.n_features
    predicted = np.array([model.predict(rows) for rows in features])
    error = np.abs(predicted - runtimes) / np.asarray(runtimes)
    print(f"median relative error {np.median(error):.2f}")
    if output is not None:
        output.write_text(json.dumps(model.to_json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk", type=int, help="the only chunk to fit to")
    parser.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    fit(args.directory, args.chunk, args.model)


if __name__ == "__main__":
    main()
