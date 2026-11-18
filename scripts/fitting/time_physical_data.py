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
In the factorize stage, each query of a variant which computes something is
timed on its own, as physical_bases splits it, with the tensors it reads
copied into its loop order and formats outside the timer, and the runtime of
the variant is the sum of theirs. The runtime, loop order, and formats of
each query are written to the timing too, under queries.

    pixi run -e fitting python -m scripts.fitting.time_physical_data DIR [options]
"""

import argparse
import json
import math
import time
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
from finch.compile_jl.runtime import DefaultFinchJLRuntime
from finch.finch_logic import Alias, Field, HardAlias, Plan, Produces, Query, Table
from scripts.fitting.generation import (
    examples,
    materialize,
    path,
    record_inputs,
)
from scripts.fitting.physical_bases import (
    factorized_queries,
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


def julia_loader(runtime: DefaultFinchJLRuntime | None = None):
    return CompilerFormLowerer(LogicCompiler(FinchJLCompiler(runtime)))


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
    which runs it again, and the Julia runtime the kernel runs in.
    """

    def __init__(self, models: Mapping[str, CostModel], repeat=10, seconds=5.0):
        self.models = models
        self.repeat = repeat
        self.seconds = seconds
        self.executors: dict[str, tuple[LogicExecutor, DefaultFinchJLRuntime]] = {}

    def measure(self, plan, formats, bindings) -> float:
        key = json.dumps([str(plan), format_kinds(formats)], sort_keys=True)
        if key not in self.executors:
            runtime = DefaultFinchJLRuntime()
            formatter = FixedFormatter(julia_loader(runtime), formats)
            executor = LogicExecutor(
                formatter, stats_factory=DCStatsFactory(), cache=True
            )
            self.executors[key] = (executor, runtime)
        executor, runtime = self.executors[key]

        def run() -> float:
            # The runtime would lend each run the storage of the outputs and
            # workspaces the last run grew, so each run allocates them anew
            # instead, as a program run once does.
            runtime.free_pool.clear()
            start = time.perf_counter()
            executor(plan, bindings)
            return time.perf_counter() - start

        # The first run compiles the kernel, which the others reuse. The others
        # stop after repeat runs or seconds of runs, so a slow variant runs once.
        run()
        runs: list[float] = []
        while len(runs) < self.repeat and sum(runs) < self.seconds:
            runs.append(run())
        return min(runs)

    def time_variant(self, record, variant, inputs, copy):
        """
        The runtime of a variant of the example of record, whose inputs are
        inputs, each of which copy(tns, options, perm) materializes in the
        formats and the order the variant reads it in, and the formats it ran
        with.
        """
        plan, formats = resolve(record, variant, inputs, self.models)
        # The inputs the program reads are stored in the order it reads them.
        perms = input_perms(record, plan)
        bindings = {
            tns: copy(tns, formats[tns], perms[tns]) for tns in inputs if tns in perms
        }
        return self.measure(plan, formats, bindings), format_kinds(formats)

    def time_factorization(self, record, variant, inputs, materialized, reformatter):
        """
        The runtime of a variant of the factorize stage, the sum of those of
        its queries, and the runtime, loop order, and formats of each. The
        results of earlier queries a query reads are materialized for it alone.
        """
        queries = []
        for query, tensors in factorized_queries(record, variant, inputs, self.models):

            def copy(tns, options, perm, tensors=tensors):
                if tns in materialized:
                    return reformatter(materialized[tns], options, perm)
                return reformat(materialize(tensors[tns]), options, perm)

            runtime, kinds = self.time_variant(
                query, {"loop_order": query["loop_order"]}, tensors, copy
            )
            queries.append(
                {
                    "runtime": runtime,
                    "loop_order": query["loop_order"],
                    "formats": kinds,
                }
            )
        return sum(query["runtime"] for query in queries), queries

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
        queries: dict[str, list[dict]] = {}
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            runtimes = timing["runtimes"]
            # Timings written before formats were saved have none.
            kinds = timing.get("formats", {})
            queries = timing.get("queries", {})
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
            if record["stage"] == "factorize":
                runtimes[str(index)], queries[str(index)] = self.time_factorization(
                    record, variant, inputs, materialized, reformatter
                )
                timing = {"runtimes": runtimes, "queries": queries}
            else:
                runtimes[str(index)], kinds[str(index)] = self.time_variant(
                    record,
                    variant,
                    inputs,
                    lambda tns, options, perm: reformatter(
                        materialized[tns], options, perm
                    ),
                )
                timing = {"runtimes": runtimes, "formats": kinds}
            write_json(timing_path, timing)
        return runtimes


def time_programs(directory: Path, models, chunk, repeat, seconds, hours) -> None:
    timer = Timer(models, repeat, seconds)
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
    parser.add_argument(
        "--repeat", type=int, default=10, help="the most timed runs of a variant"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="the seconds of timed runs after which a variant runs no more",
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
        args.seconds,
        args.hours,
    )


if __name__ == "__main__":
    main()
