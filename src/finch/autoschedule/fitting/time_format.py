"""
Time the programs which generate_format wrote to DIR. Each program is compiled
by Finch.jl with its formats of w and C, and its inputs are copied out of
binsparse into dense arrays. Its format features are recorded and its runtime
is measured, and both are written to DIR as JSON.

    pixi run -e test-julia python -m finch.autoschedule.fitting.time_format \
        DIR [--repeat N]
"""

import argparse
import json
import timeit
from pathlib import Path

import numpy as np

import binsparse as bsp

import finch as fl
from finch import ffuncs
from finch.algebra import ftypes
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.cost_models import (
    ByKind,
    ElementLevelOption,
    LevelBasis,
    LevelOption,
)
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.fitting.generate_format import OPS, OPTIONS, with_subsequence
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_BUILDERS,
    LEVEL_OPTIONS,
    CostFormatter,
)
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.util import flatten_plans, fuse_loop_order
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    FusedAlias,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.tensor import element, fiber_tensor

A, B, C, w = HardAlias("A"), HardAlias("B"), HardAlias("C"), HardAlias("w")
BASIS = ByKind(
    {
        option.kind: LevelBasis(type(option))
        for option in (*LEVEL_OPTIONS, ElementLevelOption())
    }
)


class FixedFormatter(CostFormatter):
    """
    A formatter which gives each alias the level options of formats, and
    records the features of the last program it lowers.
    """

    def __init__(self, loader, formats: dict[Alias, list[LevelOption]]):
        super().__init__(loader)
        self.formats = formats
        self.features = np.empty((0, BASIS.n_features))

    def get_tensor_ftype(self, alias, term, machine, fill_type, shape_type):
        lvl = element(fill_type, fill_type.element_type, ftypes.intp)
        options = zip(self.formats[alias], shape_type, strict=True)
        for option, dim in reversed(list(options)):
            lvl = LEVEL_BUILDERS[option.kind](lvl, dim)
        return fiber_tensor(lvl)

    def lower(self, prgm, bindings, stats, stats_factory):
        formats = {
            alias: [*options, ElementLevelOption()]
            for alias, options in self.formats.items()
        }
        self.features = BASIS.get_features(prgm, formats, stats, stats_factory)
        return super().lower(prgm, bindings, stats, stats_factory)


def julia_loader():
    return CompilerFormLowerer(LogicCompiler(FinchJLCompiler()))


def build(prgm) -> tuple[Plan, dict[Alias, list[LevelOption]]]:
    """The plan of a program, and the formats of its w and C."""
    A_idxs, B_idxs, C_idxs, w_idxs, reduced = (
        tuple(Field(name) for name in with_subsequence(prgm["order"], names))
        for names in ("ikmn", "jlmn", "kln", "kln", "ijm")
    )
    # w is fused along its leading fields, which become the outer loops of
    # both queries, so it stores a slice of its other fields for each of them.
    nfused = prgm["nfused"]
    fused = w_idxs[:nfused]
    view = FusedAlias(w, nfused) if nfused else w

    product = MapJoin(Literal(OPS[prgm["op"]]), (Table(A, A_idxs), Table(B, B_idxs)))
    reduce = Query(
        Table(view, w_idxs),
        Aggregate(Literal(ffuncs.add), Literal(0.0), product, reduced),
    )
    copy = Query(Table(C, C_idxs), Table(view, w_idxs))
    reduce_order = tuple(Field(name) for name in prgm["reduce_order"])
    copy_order = tuple(Field(name) for name in prgm["copy_order"])
    body = Plan(
        (fuse_loop_order(reduce, reduce_order), fuse_loop_order(copy, copy_order))
    )
    plan = flatten_plans(Plan((fuse_loop_order(body, fused), Produces((C,)))))
    formats: dict[Alias, list[LevelOption]] = {
        tns: [OPTIONS[kind] for kind in prgm["formats"][tns.name]] for tns in (w, C)
    }
    return plan, formats


def dense(tensor: bsp.BinsparseTensor) -> np.ndarray:
    """Allocates an array of the shape of a coo tensor and copies it in."""
    match tensor.level:
        case bsp.ElementLevel(values=values):
            return values.reshape(())
        case bsp.SparseLevel(level=bsp.ElementLevel(values=values), indices=indices):
            out = np.zeros(tensor.shape, dtype=values.dtype)
            out[indices] = values
            return out
    raise ValueError(f"expected a binsparse COO tensor, got {tensor}")


def measure(plan, formats, bindings, repeat=5) -> tuple[np.ndarray, float]:
    """The format features of plan, and the fastest of its runs."""
    formatter = FixedFormatter(julia_loader(), formats)
    executor = LogicExecutor(formatter, stats_factory=DCStatsFactory(), cache=True)
    # The first run compiles the kernel, which the others reuse.
    executor(plan, bindings)
    runs = timeit.repeat(lambda: executor(plan, bindings), number=1, repeat=repeat)
    return formatter.features, min(runs)


def time_programs(directory: Path, repeat) -> None:
    for path in sorted(directory.glob("*.program.json")):
        example = path.name.removesuffix(".program.json")
        plan, formats = build(json.loads(path.read_text()))
        bindings = {
            tns: fl.asarray(
                dense(bsp.load_binsparse(directory / f"{example}.{tns.name}.bsp.npz"))
            )
            for tns in (A, B)
        }
        features, runtime = measure(plan, formats, bindings, repeat)
        timing = {"features": features.tolist(), "runtime": runtime}
        (directory / f"{example}.timing.json").write_text(json.dumps(timing))
        print(f"{example}: {runtime:.2e}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    time_programs(args.directory, args.repeat)


if __name__ == "__main__":
    main()
