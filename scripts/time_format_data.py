"""
Time the programs which generate_format_data wrote to DIR, or those of one
chunk. Each program is compiled by Finch.jl with its format of C, and reads
its inputs after they are copied from binsparse COO into their formats. Its
runtime is measured and written to DIR/data as JSON.

    pixi run -e fitting python -m scripts.time_format_data \
        DIR [--chunk K] [--repeat N]
"""

import argparse
import json
import timeit
from pathlib import Path

from finch.algebra import Tensor, ftypes
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.cost_models import LevelOption
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.smart_formatter import LEVEL_BUILDERS, CostFormatter
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.util import fuse_loop_order
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Alias, Field, HardAlias, Plan, Produces, Query, Table
from finch.tensor import element, fiber_tensor
from scripts.generate_format_data import examples, load, path


class FixedFormatter(CostFormatter):
    """A formatter which gives each alias the level options of formats."""

    def __init__(self, loader, formats: dict[Alias, list[LevelOption]]):
        super().__init__(loader)
        self.formats = formats

    def get_tensor_ftype(self, alias, term, machine, fill_type, shape_type):
        lvl = element(fill_type, fill_type.element_type, ftypes.intp)
        options = zip(self.formats[alias], shape_type, strict=True)
        for option, dim in reversed(list(options)):
            lvl = LEVEL_BUILDERS[option.kind](lvl, dim)
        return fiber_tensor(lvl)


def julia_loader():
    return CompilerFormLowerer(LogicCompiler(FinchJLCompiler()))


def reformat(tns: Tensor, options: list[LevelOption]) -> Tensor:
    """A copy of tns with the level options, made by Finch.jl."""
    src, dst = HardAlias("src"), HardAlias("dst")
    idxs = tuple(Field(f"i{n}") for n in range(tns.ndim))
    query = fuse_loop_order(Query(Table(dst, idxs), Table(src, idxs)), idxs)
    plan = Plan((query, Produces((dst,))))
    executor = LogicExecutor(FixedFormatter(julia_loader(), {dst: options}))
    (result,) = executor(plan, {src: tns})
    return result


def measure(plan, formats, bindings, repeat=5) -> float:
    """The fastest of the runs of plan."""
    formatter = FixedFormatter(julia_loader(), formats)
    executor = LogicExecutor(formatter, stats_factory=DCStatsFactory(), cache=True)
    # The first run compiles the kernel, which the others reuse.
    executor(plan, bindings)
    return min(timeit.repeat(lambda: executor(plan, bindings), number=1, repeat=repeat))


def time_programs(directory: Path, chunk, repeat) -> None:
    names = examples(directory, chunk)
    for i, example in enumerate(names, 1):
        plan, formats, bindings = load(directory, example)
        bindings = {tns: reformat(arg, formats[tns]) for tns, arg in bindings.items()}
        runtime = measure(plan, formats, bindings, repeat)
        timing = {"runtime": runtime}
        path(directory, example, "timing.json").write_text(json.dumps(timing))
        print(f"[{i}/{len(names)}] {example}: {runtime:.2e}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk", type=int, help="the only chunk to time")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    time_programs(args.directory, args.chunk, args.repeat)


if __name__ == "__main__":
    main()
