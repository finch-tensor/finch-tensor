"""
Time the variants of the programs which a generator wrote to DIR,
or those of one chunk. The stages below that of DIR, the workspacer, the
formatter and the protocol optimizer, make their choices with the models of
--model, or the built-in ones, from the exact stats of the Kronecker blocks of
the inputs, as physical_bases resolves them, and then each variant is compiled
by Finch.jl with its formats of C and of any workspace, and reads its inputs
after they are materialized as COO from their blocks and copied into their
formats. The variants of the examples of a program compile to the same
kernels, which are compiled once and kept for as long as this runs. The
runtime of each variant, and the formats of every tensor it ran with, are
written to DIR/data as JSON after it is measured, by its index, so that
fit_physical_data needn't choose the formats again, and the variants already
timed are skipped. With --hours H, no variant is timed after H hours, so a
job measures for about as long as it is given, whatever its programs cost.

    pixi run -e fitting python -m scripts.fitting.time_physical_data DIR [options]
"""

import argparse
import json
import math
import time
import timeit
from collections.abc import Mapping
from pathlib import Path

from finch.algebra import Tensor
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.cost_models import CostModel, LevelOption
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.smart_formatter import CostFormatter, format_ftype
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.util import fuse_loop_order
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Alias, Field, HardAlias, Plan, Produces, Query, Table
from scripts.fitting.generation import (
    examples,
    materialize,
    path,
    record_inputs,
)
from scripts.fitting.physical_bases import (
    format_kinds,
    input_perms,
    parse_models,
    resolve,
)


class FixedFormatter(CostFormatter):
    def __init__(self, loader, formats: Mapping[Alias, list[LevelOption]]):
        super().__init__(loader)
        self.formats = formats

    def choose_formats(self, prgm, bindings, stats, stats_factory, stored, fill_types):
        return {
            alias: format_ftype(self.formats[alias], fill_types[alias], shape_type)
            for alias, shape_type in stored.items()
        }


def julia_loader():
    return CompilerFormLowerer(LogicCompiler(FinchJLCompiler()))


def reformat(tns: Tensor, options: list[LevelOption], perm=()) -> Tensor:
    src, dst = HardAlias("src"), HardAlias("dst")
    idxs = tuple(Field(f"i{n}") for n in range(tns.ndim))
    stored = tuple(idxs[p] for p in perm) if perm else idxs
    query = fuse_loop_order(Query(Table(dst, stored), Table(src, idxs)), stored)
    plan = Plan((query, Produces((dst,))))
    executor = LogicExecutor(FixedFormatter(julia_loader(), {dst: options}))
    (result,) = executor(plan, {src: tns})
    return result


class Reformatter:
    """
    Copies tensors into formats and field orders with reformat, keeping the
    last copy of each tensor: the variants of an example mostly keep the
    format of each input from one to the next, and every copy of a large
    input takes its memory.
    """

    def __init__(self):
        self.copies: dict[int, tuple[tuple, Tensor]] = {}

    def __call__(self, tns: Tensor, options: list[LevelOption], perm=()) -> Tensor:
        key = (tuple(option.kind for option in options), tuple(perm))
        cached = self.copies.get(id(tns))
        if cached is None or cached[0] != key:
            cached = (key, reformat(tns, options, perm))
            self.copies[id(tns)] = cached
        return cached[1]


def write_json(file: Path, data) -> None:
    # A job killed while writing leaves the file as it was, rather than cut off.
    partial = file.with_name(f"{file.name}.partial")
    partial.write_text(json.dumps(data))
    partial.replace(file)


class Timer:
    """
    Times programs with their formats, keeping the executor of each program
    and its formats, which keeps the kernel it compiled for every example
    which runs it again.
    """

    def __init__(self, models: Mapping[str, CostModel], repeat=5):
        self.models = models
        self.repeat = repeat
        self.executors: dict[str, LogicExecutor] = {}

    def measure(self, plan, formats, bindings) -> float:
        key = json.dumps([str(plan), format_kinds(formats)], sort_keys=True)
        if key not in self.executors:
            formatter = FixedFormatter(julia_loader(), formats)
            self.executors[key] = LogicExecutor(
                formatter, stats_factory=DCStatsFactory(), cache=True
            )
        executor = self.executors[key]
        # The first run compiles the kernel, which the others reuse.
        executor(plan, bindings)
        runs = timeit.repeat(
            lambda: executor(plan, bindings), number=1, repeat=self.repeat
        )
        return min(runs)

    def time_example(self, directory: Path, example, deadline) -> dict[str, float]:
        """
        Time each variant of example which hasn't been timed, and write their
        runtimes and the formats they ran with to its timing.json after each,
        until the time of time.monotonic passes deadline.
        """
        record = json.loads(path(directory, example, "program.json").read_text())
        timing_path = path(directory, example, "timing.json")
        runtimes: dict[str, float] = {}
        kinds: dict[str, dict[str, list[str]]] = {}
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            runtimes = timing["runtimes"]
            # Timings written before formats were saved have none.
            kinds = timing.get("formats", {})
        if len(runtimes) == len(record["variants"]):
            return runtimes
        inputs = record_inputs(record)
        materialized = {tns: materialize(tensor) for tns, tensor in inputs.items()}
        reformatter = Reformatter()
        for index, variant in enumerate(record["variants"]):
            if str(index) in runtimes:
                continue
            if time.monotonic() > deadline:
                break
            plan, formats = resolve(record, variant, inputs, self.models)
            # The inputs the program reads are stored in the order it reads them.
            perms = input_perms(record, plan)
            bindings = {
                tns: reformatter(tensor, formats[tns], perms[tns])
                for tns, tensor in materialized.items()
                if tns in perms
            }
            runtimes[str(index)] = self.measure(plan, formats, bindings)
            kinds[str(index)] = format_kinds(formats)
            write_json(timing_path, {"runtimes": runtimes, "formats": kinds})
        return runtimes


def time_programs(directory: Path, models, chunk, repeat, hours) -> None:
    timer = Timer(models, repeat)
    names = examples(directory, chunk)
    deadline = time.monotonic() + 3600 * hours
    for i, example in enumerate(names, 1):
        runtimes = timer.time_example(directory, example, deadline)
        if runtimes:
            print(
                f"[{i}/{len(names)}] {example}: {len(runtimes)} variants, "
                f"best {min(runtimes.values()):.2e}s, {len(timer.executors)} kernels",
                flush=True,
            )
        if time.monotonic() > deadline:
            print(f"stopped after {hours} hours, at {i} of {len(names)}", flush=True)
            return


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
    parser.add_argument("--chunk", type=int, help="the only chunk to time")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--hours",
        type=float,
        default=math.inf,
        help="the hours after which no more variants are timed",
    )
    args = parser.parse_args()
    time_programs(
        args.directory, parse_models(args.model), args.chunk, args.repeat, args.hours
    )


if __name__ == "__main__":
    main()
