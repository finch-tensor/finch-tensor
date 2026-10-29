"""
Fit a LinearCostModel of the formatter's basis to the timings which
time_format wrote to DIR, and write it to model.json if given.

    pixi run -e test-julia python -m finch.autoschedule.fitting.fit_format \
        DIR [model.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from finch.autoschedule.cost_models import LinearCostModel
from finch.autoschedule.fitting.time_format import BASIS


def fit(directory: Path, output: Path | None) -> None:
    timings = [
        json.loads(path.read_text()) for path in sorted(directory.glob("*.timing.json"))
    ]
    features = [
        np.array(timing["features"]).reshape(-1, BASIS.n_features) for timing in timings
    ]
    runtimes = [timing["runtime"] for timing in timings]
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
    parser.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    fit(args.directory, args.model)


if __name__ == "__main__":
    main()
