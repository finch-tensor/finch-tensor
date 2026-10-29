"""
Fit the formatter's cost model from measured programs. We draw random
programs which reduce a map of inputs A and B into a workspace w and copy w
into C. Their fields cover each region of the Venn diagram of the fields of A,
B, and C which lies in A or B, since nothing would give a field of C alone an
extent. The layouts of all tensors follow one drawn order of the fields, and
each query has its own drawn loop order. The formats of w and C, and the
extents, densities, and dtype of the inputs are drawn too.

The script runs in three steps, which share a directory. generate draws the
programs, writing each as JSON and its inputs as binsparse. time compiles each
program with Finch.jl and its formats, records its format features, measures
its runtime, and writes both as JSON. fit fits a LinearCostModel of the
formatter's basis to all the timings.

    pixi run -e test-julia python -m finch.autoschedule.fitting.collect_format \
        generate DIR [--examples N] [--lg-max-space N] [--lg-max-dim N] [--seed S]
    pixi run -e test-julia python -m finch.autoschedule.fitting.collect_format \
        time DIR [--repeat N]
    pixi run -e test-julia python -m finch.autoschedule.fitting.collect_format \
        fit DIR [model.json]
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
    LinearCostModel,
)
from finch.autoschedule.executor import LogicExecutor
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
OPS = {"add": ffuncs.add, "mul": ffuncs.mul}
OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}
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


def with_subsequence(order, fields):
    """The fields of order which are in fields, in the order of order."""
    return tuple(idx for idx in order if idx in fields)


def seed(s_x, s_y, r_xy):
    S = [
        [1 + s_x + s_y + r_xy, 1 + s_x - s_y - r_xy],
        [1 - s_x + s_y - r_xy, 1 - s_x - s_y + r_xy],
    ]
    # Rounding can leave an empty corner slightly below zero.
    return np.maximum(S, 0) / 4


def interaction_range(s_x, s_y):
    return abs(s_x + s_y) - 1, 1 - abs(s_x - s_y)


def choose(rng, options):
    return options[rng.integers(len(options))]


def shuffled(rng, items):
    return tuple(items[p] for p in rng.permutation(len(items)))


def interaction(rng, base, scale, align, r_range):
    """Draws a skew (base + scale * r) / (1 + align) of C and solves for r."""
    s = rng.uniform(*sorted((base + scale * r) / (1 + align) for r in r_range))
    # Clipping only absorbs rounding, which dividing by a small scale amplifies.
    return np.clip(((1 + align) * s - base) / scale, *r_range)


def seeds(rng, lg_A, lg_B, lg_C, lg_red, lg_k):
    # R-MAT seeds and densities for A[i, k] and B[k, j], whose product sums
    # over k into C[i, j]. A, B, and C may have other fields, which the seeds
    # leave unskewed, so lg_A, lg_B, and lg_C are the bits of their whole
    # shapes, lg_red the bits of every field reduced, and lg_k those of k.
    # Each level of a seed is a 2x2 matrix S over fields x and y, with skews
    # s_x and s_y in [-1, 1] and interaction r_xy, where
    #     S = [[1 + s_x + s_y + r_xy, 1 + s_x - s_y - r_xy],
    #          [1 - s_x + s_y - r_xy, 1 - s_x - s_y + r_xy]] / 4.
    # rmat marginalizes S over a field once it runs out of bits, so only the
    # levels of k see both A and B. The expected number of products which
    # land in each entry of C is
    #     p_C = p_A p_B 2 ** lg_red (1 + align) ** lg_k,
    # where align = s_Ak s_Bk, and on the levels of k,
    #     s_Ci = (s_Ai + r_Aik s_Bk) / (1 + align),
    #     s_Cj = (s_Bj + s_Ak r_Bkj) / (1 + align),
    #     r_Cij = (s_Ai s_Bj + r_Aik r_Bkj) / (1 + align),
    # while C copies s_Ai and s_Bj on the other levels of i and j. This is also
    # the seed and density of C while its products rarely collide. Drawing log
    # densities p_A, p_B, and p_C, from one expected nonzero up to dense with
    # p_C < p_A p_B 2 ** (lg_red + lg_k), fixes align, and align = 0 would pin
    # a skew of C. Then draw s_Ai and s_Bj in [-1, 1], s_Ak with the sign of
    # align and |align| <= |s_Ak| <= 1, and set s_Bk = align / s_Ak. An
    # interaction keeps S nonnegative when it lies in
    # [|s_x + s_y| - 1, 1 - |s_x - s_y|], so draw s_Ci and s_Cj from the images
    # of those intervals and solve for r_Aik and r_Bkj.
    # A batch field can leave too few products for one nonzero of C, so
    # redraw p_A and p_B until there are enough.
    while True:
        lg_pA, lg_pB = rng.uniform(-lg_A, 0), rng.uniform(-lg_B, 0)
        lg_max = min(0, lg_pA + lg_pB + lg_red + lg_k)
        if not lg_k or -lg_C <= lg_max:
            break
    if lg_k:
        lg_pC = rng.uniform(-lg_C, lg_max)
        align = 2 ** ((lg_pC - lg_pA - lg_pB - lg_red) / lg_k) - 1
    else:
        # Without levels of k, p_C = p_A p_B 2 ** lg_red whatever align is.
        align = rng.uniform(-1, 1)
    s_Ai, s_Bj = rng.uniform(-1, 1, 2)
    s_Ak = np.copysign(rng.uniform(abs(align), 1), align)
    s_Bk = align / s_Ak
    r_Aik = interaction(rng, s_Ai, s_Bk, align, interaction_range(s_Ai, s_Ak))
    r_Bkj = interaction(rng, s_Bj, s_Ak, align, interaction_range(s_Bk, s_Bj))
    return (
        (2**lg_pA, seed(s_Ai, s_Ak, r_Aik)),
        (2**lg_pB, seed(s_Bk, s_Bj, r_Bkj)),
    )


def rmat(S, lg_shape, density, rng):
    """
    Coordinates of density * 2 ** sum(lg_shape) balls dropped into a tensor of
    shape 2 ** lg_shape. At each level a ball picks an entry of the seed S, and
    each axis with bits left takes that entry's bit along it, so an axis which
    has run out of bits marginalizes S over it.
    """
    if not lg_shape:
        return ()
    S = np.asarray(S)
    nnz = round(density * 2 ** sum(lg_shape))
    picks = rng.choice(S.size, size=(nnz, max(lg_shape)), p=S.ravel())
    bits = np.unravel_index(picks, S.shape)
    return tuple(
        (b[:, :lg] << np.arange(lg - 1, -1, -1)).sum(axis=1)
        for b, lg in zip(bits, lg_shape, strict=True)
    )


def program(rng, lg_max_space, lg_max_dim):
    """A random program, as the JSON which build turns into a plan."""
    # Each region of the Venn diagram of the indices of A, B, and C has one
    # field or none: i is only in A, j only in B, k in A and C, l in B and C,
    # m in A and B, and n in all three. Every field has 2 ** lg_dims values,
    # and all of them together address at most 2 ** lg_max_space entries.
    while True:
        names = [name for name in "ijklmn" if rng.random() < 0.5]
        lg_dims = {name: int(rng.integers(1, lg_max_dim + 1)) for name in names}
        if sum(lg_dims.values()) <= lg_max_space:
            break
    order = shuffled(rng, names)
    w_names = with_subsequence(order, "kln")
    nfused = int(rng.integers(0, len(w_names) + 1))
    return {
        "op": choose(rng, list(OPS)),
        "order": order,
        "lg_dims": lg_dims,
        "nfused": nfused,
        "reduce_order": shuffled(
            rng, (*w_names[nfused:], *with_subsequence(order, "ijm"))
        ),
        "copy_order": shuffled(rng, w_names[nfused:]),
        "formats": {
            name: [choose(rng, list(OPTIONS)) for _ in range(ndim)]
            for name, ndim in (("w", len(w_names) - nfused), ("C", len(w_names)))
        },
        "dtype": choose(rng, ["float32", "float64"]),
    }


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


def inputs(rng, prgm) -> dict[str, bsp.CustomTensor]:
    """Binsparse COO tensors A and B for a program, drawn from R-MAT seeds."""
    order, lg_dims = prgm["order"], prgm["lg_dims"]
    lg_A, lg_B, lg_C, lg_red, lg_m = (
        sum(lg_dims.get(name, 0) for name in names)
        for names in ("ikmn", "jlmn", "kln", "ijm", "m")
    )
    (p_A, S_A), (p_B, S_B) = seeds(rng, lg_A, lg_B, lg_C, lg_red, lg_m)
    dtype = np.dtype(prgm["dtype"])
    tensors = {}
    # The seeds skew A[k, m] B[m, l], and leave i, j, and n unskewed.
    for tns, names, p, S, xy in (
        ("A", "ikmn", p_A, S_A, "km"),
        ("B", "jlmn", p_B, S_B, "ml"),
    ):
        idxs = with_subsequence(order, names)
        axes = [idx for idx in xy if idx in lg_dims]
        S = S.sum(axis=tuple(ax for ax, idx in enumerate(xy) if idx not in lg_dims))
        for idx in idxs:
            if idx not in axes:
                S = np.multiply.outer(S, [0.5, 0.5])
                axes.append(idx)
        S = np.transpose(S, [axes.index(idx) for idx in idxs])
        lg_shape = tuple(lg_dims[idx] for idx in idxs)
        coords = rmat(S, lg_shape, p, rng)
        tensors[tns] = coo(rng, coords, tuple(2**lg for lg in lg_shape), dtype)
    return tensors


def coo(rng, coords, shape, dtype) -> bsp.CustomTensor:
    """A binsparse COO tensor of random values at coords, merging duplicates."""
    if shape:
        flat = np.unique(np.ravel_multi_index(coords, shape))
        values = rng.random(flat.size).astype(dtype)
        indices = np.unravel_index(flat, shape)
        level = bsp.SparseLevel(len(shape), bsp.ElementLevel(values), indices)
    else:
        values = rng.random(1).astype(dtype)
        level = bsp.ElementLevel(values)
    return bsp.CustomTensor(shape, values.size, True, dtype.type(0), level)


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


def generate(directory: Path, examples, lg_max_space, lg_max_dim, rng) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for example in range(examples):
        prgm = program(rng, lg_max_space, lg_max_dim)
        (directory / f"{example:05d}.program.json").write_text(json.dumps(prgm))
        for tns, tensor in inputs(rng, prgm).items():
            bsp.save_binsparse(tensor, directory / f"{example:05d}.{tns}.bsp.npz")


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
    steps = parser.add_subparsers(dest="step", required=True)
    step = steps.add_parser("generate", help="draw programs and their inputs")
    step.add_argument("directory", type=Path)
    step.add_argument("--examples", type=int, default=20)
    step.add_argument(
        "--lg-max-space",
        type=int,
        default=17,
        help="log2 of the most entries all the fields of a program may address",
    )
    step.add_argument(
        "--lg-max-dim",
        type=int,
        default=9,
        help="log2 of the most values of any one field",
    )
    step.add_argument("--seed", type=int)
    step = steps.add_parser("time", help="time the programs in a directory")
    step.add_argument("directory", type=Path)
    step.add_argument("--repeat", type=int, default=5)
    step = steps.add_parser("fit", help="fit the cost model to the timings")
    step.add_argument("directory", type=Path)
    step.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    match args.step:
        case "generate":
            rng = np.random.default_rng(args.seed)
            generate(
                args.directory, args.examples, args.lg_max_space, args.lg_max_dim, rng
            )
        case "time":
            time_programs(args.directory, args.repeat)
        case "fit":
            fit(args.directory, args.model)


if __name__ == "__main__":
    main()
