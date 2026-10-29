"""
Fit the formatter's cost model from measured programs. Hypothesis draws
programs which write a tensor B from an input A and then copy B into C, each
in a drawn loop order and layout, along with the formats of B and C and the
shape, density, and dtype of A. Each program is compiled by Finch.jl with
those formats, its format features are recorded, and its runtime is measured.
A LinearCostModel of the formatter's basis is then fitted to all of them.

    pixi run -e test-julia python -m finch.autoschedule.fitting.collect_format \
        [examples] [model.json]
"""

import json
import sys
import timeit
from math import prod

import numpy as np
from hypothesis import HealthCheck, Phase, assume, given, settings
from hypothesis import strategies as st

import finch as fl
from finch import ffuncs
from finch.algebra import ftypes
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.cost_models import (
    ByKind,
    ElementLevelOption,
    LevelBasis,
    LevelOption,
    LinearCostModel,
)
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_BUILDERS,
    LEVEL_OPTIONS,
    CostFormatter,
)
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.util import fuse_loop_order
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Alias, Field, HardAlias, Plan, Produces, Query, Table
from finch.tensor import element, fiber_tensor

A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")
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


@st.composite
def programs(draw):
    ndim = draw(st.integers(1, 3))
    op = draw(st.sampled_from([ffuncs.add, ffuncs.mul]))
    # Each region of the Venn diagram of the indices of A, B, and C has one
    # field or none: i is only in C, j only in A, k only in B, l in A and C,
    # m in B and C, n in A and B, and p in all three.
    i, j, k, l, m, n, p = (
        draw(st.sampled_from([(Field(name),), ()])) for name in "ijklmnp"
    )
    idxs = (*i, *j, *k, *l, *m, *n, *p)
    lg_dims = {idx: draw(st.integers(1, 9)) for idx in idxs}
    assume(sum(lg_dims.values()) < 18)
    idx_perm = tuple(draw(st.permutations(idxs)))
    C_idxs = with_subsequence(idx_perm, (*i, *l, *m, *p))
    A_idxs = with_subsequence(idx_perm, (*j, *l, *n, *p))
    B_idxs = with_subsequence(idx_perm, (*k, *m, *n, *p))
    w_idxs = with_subsequence(idx_perm, (*l, *m, *p))

    shape = tuple(
        draw(st.lists(st.sampled_from([8, 64, 512]), min_size=ndim, max_size=ndim))
    )
    assume(prod(shape) <= 2**18)
    density = draw(st.sampled_from([0.01, 0.1, 1.0]))
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    seed = draw(st.integers(0, 2**32 - 1))

    fields = tuple(Field(f"i{n}") for n in range(ndim))
    b_idxs, c_idxs, b_order, c_order = (
        tuple(draw(st.permutations(fields))) for _ in range(4)
    )
    level = st.sampled_from(LEVEL_OPTIONS)
    formats: dict[Alias, list[LevelOption]] = {
        tns: draw(st.lists(level, min_size=ndim, max_size=ndim)) for tns in (B, C)
    }



    rng = np.random.default_rng(seed)
    data = rng.random(shape) * (rng.random(shape) < density)
    plan = Plan(
        (
            Query(Table(w, w_idxs), Aggregate(ffuncs.add, Literal(0), MapJoin(op, Table(A, A_idxs), Table(B, B_idxs)))),
            Query(Table(C, C_idxs), Table(w, w_idxs)),
            Produces((C,)),
        )
    )
    return plan, formats, fl.asarray(data.astype(dtype))


def measure(plan, formats, tensor, repeat=5) -> tuple[np.ndarray, float]:
    """The format features of plan, and the fastest of its runs."""
    formatter = FixedFormatter(julia_loader(), formats)
    executor = LogicExecutor(formatter, stats_factory=DCStatsFactory(), cache=True)
    # The first run compiles the kernel, which the others reuse.
    executor(plan, {A: tensor})
    runs = timeit.repeat(lambda: executor(plan, {A: tensor}), number=1, repeat=repeat)
    return formatter.features, min(runs)


def collect(examples: int) -> tuple[list[np.ndarray], list[float]]:
    features: list[np.ndarray] = []
    runtimes: list[float] = []

    @settings(
        max_examples=examples,
        deadline=None,
        database=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(programs())
    def run(sample):
        rows, runtime = measure(*sample)
        features.append(rows)
        runtimes.append(runtime)
        print(f"{len(runtimes)}: {runtime:.2e}s", flush=True)

    run()
    return features, runtimes


def main() -> None:
    examples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    features, runtimes = collect(examples)
    model = LinearCostModel(BASIS, np.zeros(BASIS.n_features)).fit(features, runtimes)
    start = 0
    for kind, basis in BASIS.by_kind.items():
        coefficients = model.coefficients[start : start + basis.n_features]
        print(kind, np.array2string(coefficients, precision=3))
        start += basis.n_features
    predicted = np.array([model.predict(rows) for rows in features])
    error = np.abs(predicted - runtimes) / np.asarray(runtimes)
    print(f"median relative error {np.median(error):.2f}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as file:
            json.dump(model.to_json(), file, indent=2)


if __name__ == "__main__":
    main()
