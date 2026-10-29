"""
Fit the formatter's cost model from measured programs. We draw random
programs which reduce a map of inputs A and B into a workspace w and copy w
into C. Their fields cover each region of the Venn diagram of the fields of A,
B, and C which lies in A or B, since nothing would give a field of C alone an
extent. The layouts of all tensors follow one drawn order of the fields, and
each query has its own drawn loop order. The formats of w and C, and the
extents, densities, and dtype of the inputs are drawn too. Each program is
compiled by Finch.jl with those formats, its format features are recorded,
and its runtime is measured. A LinearCostModel of the formatter's basis is
then fitted to all of them.

    pixi run -e test-julia python -m finch.autoschedule.fitting.collect_format \
        [examples] [model.json]
"""

import json
import sys
import timeit

import numpy as np

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


def program(rng):
    op = choose(rng, [ffuncs.add, ffuncs.mul])
    # Each region of the Venn diagram of the indices of A, B, and C has one
    # field or none: i is only in A, j only in B, k in A and C, l in B and C,
    # m in A and B, and n in all three.
    i, j, k, l, m, n = (  # noqa: E741
        choose(rng, [(Field(name),), ()]) for name in "ijklmn"
    )
    idxs = (*i, *j, *k, *l, *m, *n)
    while True:
        lg_dims = {idx: int(rng.integers(1, 10)) for idx in idxs}
        if sum(lg_dims.values()) < 18:
            break
    idx_perm = shuffled(rng, idxs)

    C_idxs = with_subsequence(idx_perm, (*k, *l, *n))
    A_idxs = with_subsequence(idx_perm, (*i, *k, *m, *n))
    B_idxs = with_subsequence(idx_perm, (*j, *l, *m, *n))
    w_idxs = with_subsequence(idx_perm, (*k, *l, *n))

    reduced = with_subsequence(idx_perm, (*i, *j, *m))
    # w is fused along its leading fields, which become the outer loops of
    # both queries, so it stores a slice of its other fields for each of them.
    nfused = int(rng.integers(0, len(w_idxs) + 1))
    fused = w_idxs[:nfused]
    view = FusedAlias(w, nfused) if nfused else w

    product = MapJoin(Literal(op), (Table(A, A_idxs), Table(B, B_idxs)))
    reduce = Query(
        Table(view, w_idxs),
        Aggregate(Literal(ffuncs.add), Literal(0.0), product, reduced),
    )
    copy = Query(Table(C, C_idxs), Table(view, w_idxs))
    reduce_order = shuffled(rng, (*w_idxs[nfused:], *reduced))
    copy_order = shuffled(rng, with_subsequence(C_idxs, set(C_idxs) - set(fused)))
    body = Plan(
        (fuse_loop_order(reduce, reduce_order), fuse_loop_order(copy, copy_order))
    )
    plan = flatten_plans(Plan((fuse_loop_order(body, fused), Produces((C,)))))

    stored = ((w, len(w_idxs) - nfused), (C, len(C_idxs)))
    formats: dict[Alias, list[LevelOption]] = {
        tns: [choose(rng, LEVEL_OPTIONS) for _ in range(ndim)] for tns, ndim in stored
    }

    lg_A, lg_B, lg_C, lg_red, lg_m = (
        sum(lg_dims[idx] for idx in f) for f in (A_idxs, B_idxs, C_idxs, reduced, m)
    )
    (p_A, S_A), (p_B, S_B) = seeds(rng, lg_A, lg_B, lg_C, lg_red, lg_m)
    dtype = choose(rng, [np.float32, np.float64])
    bindings = {}
    # The seeds skew A[k, m] B[m, l], and leave i, j, and n unskewed.
    for tns, tns_idxs, p, S, (x, y) in (
        (A, A_idxs, p_A, S_A, (k, m)),
        (B, B_idxs, p_B, S_B, (m, l)),
    ):
        axes = [*x, *y]
        S = S.sum(axis=tuple(ax for ax, f in enumerate((x, y)) if not f))
        for idx in tns_idxs:
            if idx not in axes:
                S = np.multiply.outer(S, [0.5, 0.5])
                axes.append(idx)
        S = np.transpose(S, [axes.index(idx) for idx in tns_idxs])
        lg_shape = tuple(lg_dims[idx] for idx in tns_idxs)
        data = np.zeros(tuple(2**lg for lg in lg_shape), dtype=dtype)
        coords = rmat(S, lg_shape, p, rng)
        data[coords] = rng.random(data[coords].shape)
        bindings[tns] = fl.asarray(data)
    return plan, formats, bindings


def measure(plan, formats, bindings, repeat=5) -> tuple[np.ndarray, float]:
    """The format features of plan, and the fastest of its runs."""
    formatter = FixedFormatter(julia_loader(), formats)
    executor = LogicExecutor(formatter, stats_factory=DCStatsFactory(), cache=True)
    # The first run compiles the kernel, which the others reuse.
    executor(plan, bindings)
    runs = timeit.repeat(lambda: executor(plan, bindings), number=1, repeat=repeat)
    return formatter.features, min(runs)


def collect(examples: int) -> tuple[list[np.ndarray], list[float]]:
    rng = np.random.default_rng()
    features: list[np.ndarray] = []
    runtimes: list[float] = []
    for _ in range(examples):
        rows, runtime = measure(*program(rng))
        features.append(rows)
        runtimes.append(runtime)
        print(f"{len(runtimes)}: {runtime:.2e}s", flush=True)
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
