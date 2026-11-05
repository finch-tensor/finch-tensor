"""
Time the variants of the programs which generate_physical_data wrote to DIR,
or those of one chunk. The stages below that of DIR make their choices with
the models of --model, or the built-in ones, from the exact stats of the
Kronecker blocks of the inputs, as physical_bases resolves them, and then each
variant is compiled by Finch.jl with its format of C, and reads its inputs
after they are materialized as COO from their blocks and copied into their
formats. The variants of the examples of a program compile to the same
kernels, which are compiled once and kept for as long as this runs. The
runtime of each variant is written to DIR/data as JSON after it is measured,
by its index. With --hours H, no variant is timed after H hours, so a job
measures for about as long as it is given, whatever its programs cost.

    pixi run -e fitting python -m scripts.fitting.time_physical_data \\
        DIR [--model NAME=PATH]... [--chunk K] [--repeat N] [--resume] \\
        [--hours H]
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
from scripts.fitting.generate_physical_data import (
    examples,
    load,
    materialize,
    path,
    write_json,
)
from scripts.fitting.physical_bases import input_perms, parse_models, resolve


class FixedFormatter(CostFormatter):
    """A formatter which gives each alias the level options of formats."""

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
    """
    A copy of tns with the level options, made by Finch.jl, storing the fields
    of tns in the order of perm, if given.
    """
    src, dst = HardAlias("src"), HardAlias("dst")
    idxs = tuple(Field(f"i{n}") for n in range(tns.ndim))
    stored = tuple(idxs[p] for p in perm) if perm else idxs
    query = fuse_loop_order(Query(Table(dst, stored), Table(src, idxs)), stored)
    plan = Plan((query, Produces((dst,))))
    executor = LogicExecutor(FixedFormatter(julia_loader(), {dst: options}))
    (result,) = executor(plan, {src: tns})
    return result


class Reformatter:
    """Copies tensors into formats and field orders with copy, each once."""

    def __init__(self, copy=reformat):
        self.copy = copy
        self.copies: dict[tuple, Tensor] = {}

    def __call__(self, tns: Tensor, options: list[LevelOption], perm=()) -> Tensor:
        key = (id(tns), tuple(option.kind for option in options), tuple(perm))
        if key not in self.copies:
            self.copies[key] = self.copy(tns, options, perm)
        return self.copies[key]


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
        """The fastest of the runs of plan."""
        kinds = {tns.name: [opt.kind for opt in opts] for tns, opts in formats.items()}
        key = json.dumps([str(plan), kinds], sort_keys=True)
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

    def time_example(
        self, directory: Path, example, resume, deadline=math.inf
    ) -> dict[str, float]:
        """
        Time each variant of example which hasn't been timed, if resuming, and
        write their runtimes to its timing.json after each, until the time of
        time.monotonic passes deadline.
        """
        record, inputs = load(directory, example)
        timing_path = path(directory, example, "timing.json")
        runtimes: dict[str, float] = {}
        if resume and timing_path.exists():
            runtimes = json.loads(timing_path.read_text())["runtimes"]
        if len(runtimes) == len(record["variants"]):
            return runtimes
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
            write_json(timing_path, {"runtimes": runtimes})
        return runtimes


def time_programs(directory: Path, models, chunk, repeat, resume, hours) -> None:
    timer = Timer(models, repeat)
    names = examples(directory, chunk)
    deadline = time.monotonic() + 3600 * hours
    for i, example in enumerate(names, 1):
        runtimes = timer.time_example(directory, example, resume, deadline)
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
        "--resume", action="store_true", help="skip the variants already timed"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=math.inf,
        help="the hours after which no more variants are timed",
    )
    args = parser.parse_args()
    time_programs(
        args.directory,
        parse_models(args.model),
        args.chunk,
        args.repeat,
        args.resume,
        args.hours,
    )


if __name__ == "__main__":
    main()
