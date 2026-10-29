"""
Draw programs for fitting the formatter's cost model, and their inputs. Each
program reduces a map of inputs A and B into a workspace w and copies w into
C. Their fields cover each region of the Venn diagram of the fields of A, B,
and C which lies in A or B, since nothing would give a field of C alone an
extent. The layouts of all tensors follow one drawn order of the fields, and
each query has its own drawn loop order. The formats of w and C, and the
extents, densities, and dtype of the inputs are drawn too. Each program is
written to DIR as JSON, and its inputs as binsparse.

    pixi run -e test-julia python -m finch.autoschedule.fitting.generate_format \
        DIR [--examples N] [--lg-max-space N] [--lg-max-dim N] [--seed S]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import binsparse as bsp

from finch import ffuncs
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS

OPS = {"add": ffuncs.add, "mul": ffuncs.mul}
OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}


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
    """A random program, as the JSON which time_format.build makes a plan of."""
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


def generate(directory: Path, examples, lg_max_space, lg_max_dim, rng) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for example in range(examples):
        prgm = program(rng, lg_max_space, lg_max_dim)
        (directory / f"{example:05d}.program.json").write_text(json.dumps(prgm))
        for tns, tensor in inputs(rng, prgm).items():
            bsp.save_binsparse(tensor, directory / f"{example:05d}.{tns}.bsp.npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--lg-max-space",
        type=int,
        default=17,
        help="log2 of the most entries all the fields of a program may address",
    )
    parser.add_argument(
        "--lg-max-dim",
        type=int,
        default=9,
        help="log2 of the most values of any one field",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    generate(args.directory, args.examples, args.lg_max_space, args.lg_max_dim, rng)


if __name__ == "__main__":
    main()
