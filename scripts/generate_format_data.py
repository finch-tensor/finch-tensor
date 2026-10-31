"""
Draw programs for fitting the formatter's cost model, and their inputs. Each
program is a random subprogram of C_lmq += A_jlpq * B_kmpq, with a drawn loop
order and formats of A, B, and C, and its inputs are R-MAT tensors of drawn
densities. Of the samples draws of the inputs of a program, keep are kept,
spread evenly over the densities of A, B, and C, and each makes an example with
the program. Each example is written to DIR/data as JSON, with the text of its
logic plan, the formats of A, B, and C, and the paths of their binsparse files,
under the name of its chunk and its index in the chunk, so jobs of different
chunks can share DIR. The inputs are stored as COO, and C as an empty COO tensor
of its rank.

    pixi run -e fitting python -m scripts.generate_format_data \
        DIR [--programs N] [--samples N] [--keep N] [--chunk K] \
        [--lg-max-space N] [--lg-max-dim N] [--seed S]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import binsparse as bsp

import finch as fl
from finch import ffuncs
from finch.algebra import Tensor, ftype
from finch.autoschedule.cost_models import LevelOption
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS
from finch.autoschedule.util import fuse_loop_order
from finch.codegen import NumpyBuffer
from finch.fileio import from_binsparse, to_binsparse
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    HardAlias,
    Literal,
    LogicInterpreter,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
    parse_logic,
)
from finch.tensor import ElementLevel, FiberTensor, SparseCOOLevel, element

OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}
A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")


def with_subsequence(order, fields):
    """The fields of order which are in fields, in the order of order."""
    return tuple(idx for idx in order if idx in fields)


def choose(rng, options):
    return options[rng.integers(len(options))]


def shuffled(rng, items):
    return tuple(items[p] for p in rng.permutation(len(items)))


def seed(skew, sparsity):
    S = np.ones(())
    for s in skew:
        S = np.multiply.outer(S, [0.5 - s / 2, 0.5 + s / 2])
    return S * sparsity


def levels(S, lg_shape):
    """
    The axes with bits left at each level of R-MAT with the seed S on a tensor
    of shape 2 ** lg_shape, and the seed over them. At each level a ball picks
    an entry of S, and each axis with bits left takes that entry's bit along it,
    so an axis which has run out of bits marginalizes S over it.
    """
    S = np.asarray(S) / np.sum(S)
    for level in range(max(lg_shape, default=0)):
        done = tuple(d for d, lg in enumerate(lg_shape) if lg <= level)
        live = tuple(d for d, lg in enumerate(lg_shape) if level < lg)
        yield live, S.sum(axis=done)


def cell_probabilities(S, lg_shape):
    """
    The probability that a ball dropped by R-MAT with the seed S lands in each
    cell of a tensor of shape 2 ** lg_shape.
    """
    P = np.ones(())
    bits = []
    for level, (live, S_level) in enumerate(levels(S, lg_shape)):
        P = np.multiply.outer(P, S_level)
        bits += [(d, level) for d in live]
    # The bits of an axis, from the top, make its coordinate.
    order = sorted(range(len(bits)), key=bits.__getitem__)
    return P.transpose(order).reshape(tuple(2**lg for lg in lg_shape))


def drop_balls(S, lg_shape, nnz, rng):
    """
    The flat indices of the first nnz distinct cells which balls dropped by
    R-MAT with the seed S land in, in a tensor of shape 2 ** lg_shape.
    """
    shape = tuple(2**lg for lg in lg_shape)
    cells = np.empty(0, np.int64)
    distinct, first = cells, cells
    while distinct.size < nnz:
        need = nnz - distinct.size
        n = need + need // 4 + 16
        coords = [np.zeros(n, np.int64) for _ in lg_shape]
        for live, S_level in levels(S, lg_shape):
            picks = rng.choice(S_level.size, size=n, p=S_level.ravel())
            for d, bit in zip(
                live, np.unravel_index(picks, S_level.shape), strict=True
            ):
                coords[d] = coords[d] << 1 | bit
        cells = np.concatenate([cells, np.ravel_multi_index(coords, shape)])
        distinct, first = np.unique(cells, return_index=True)
    return distinct[np.argsort(first)[:nnz]]


def rmat(S, lg_shape, dtype, rng):
    """
    A dense array of shape 2 ** lg_shape whose S.sum() * 2 ** sum(lg_shape)
    nonzeros are the first distinct cells which balls dropped by R-MAT with the
    seed S land in.
    """
    # Values are in (0, 1], so neither a sum nor a product of them is zero.
    if not lg_shape:
        return np.array(1 - rng.random(), dtype)
    size = 2 ** sum(lg_shape)
    nnz = round(np.sum(S) * size)
    # A ball lands in one of about 2 ** H cells, for the entropy H of the cell
    # it lands in. When nnz is well under that and under the size, balls are
    # dropped until nnz cells are hit, and otherwise every cell is ordered.
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -sum(np.nansum(P * np.log2(P)) for _, P in levels(S, lg_shape))
    if np.log2(nnz) <= min(H - 2, sum(lg_shape) - 5):
        cells = drop_balls(S, lg_shape, nnz, rng)
    else:
        # The order in which balls first land in cells is the order of
        # exponential draws over the probabilities of the cells.
        with np.errstate(divide="ignore"):
            keys = rng.exponential(size=size) / cell_probabilities(S, lg_shape).ravel()
        cells = np.argpartition(keys, nnz - 1)[:nnz]
    tns = np.zeros(size, dtype)
    tns[cells] = 1 - rng.random(nnz)
    return tns.reshape(tuple(2**lg for lg in lg_shape))


def program(rng, lg_max_space, lg_max_dim):
    """
    A random subprogram of C_lmq += A_jlpq * B_kmpq. C has no fields of its
    own, since nothing would give them extents.
    """
    while True:
        idxs = [idx for idx in "jklmpq" if rng.random() < 0.5]
        lg_dims = {idx: int(rng.integers(1, lg_max_dim + 1)) for idx in idxs}
        if sum(lg_dims.values()) <= lg_max_space:
            break
    order = shuffled(rng, idxs)
    A_idxs = tuple(Field(idx) for idx in with_subsequence(order, "jlpq"))
    # if rng.random() < 0.2:
    #    A_idxs = shuffled(rng, A_idxs)
    B_idxs = tuple(Field(idx) for idx in with_subsequence(order, "kmpq"))
    # if rng.random() < 0.2:
    #    B_idxs = shuffled(rng, B_idxs)
    C_idxs = tuple(Field(idx) for idx in with_subsequence(order, "lmq"))
    if rng.random() < 0.3:
        C_idxs = shuffled(rng, C_idxs)
    r_idxs = tuple(Field(idx) for idx in with_subsequence(order, "jkp"))
    prgm = Plan.from_children(
        Query(
            Table(HardAlias("C"), C_idxs),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0.0),
                MapJoin.from_children(
                    Literal(ffuncs.mul),
                    Table(HardAlias("A"), A_idxs),
                    Table(HardAlias("B"), B_idxs),
                ),
                r_idxs,
            ),
        ),
        Produces((C,)),
    )
    input_fields = {
        HardAlias("A"): A_idxs,
        HardAlias("B"): B_idxs,
    }
    output_fields = {
        HardAlias("C"): C_idxs,
    }
    return prgm, input_fields, output_fields, lg_dims


def choose_seed(rng, idxs, lg_dims):
    skews = [rng.choice([-1, 1]) * rng.random() for _ in idxs]
    sparsity = 2 ** -rng.uniform(0, sum(lg_dims[idx.name] for idx in idxs))
    return seed(skews, sparsity)


def inputs(rng, input_fields, lg_dims, seeds, dtype):
    return {
        name: rmat(seeds[name], [lg_dims[idx.name] for idx in idxs], dtype, rng)
        for name, idxs in input_fields.items()
    }


def seed_lg_densities(prgm, input_fields, output_fields, lg_dims, seeds):
    """
    The log nonzeros of the inputs which rmat draws from seeds, and the log
    expected products of the output of prgm, each as a fraction of its log
    size. By the mixed product property, the expected products are the product
    over levels of the sum of prgm run on the seeds of the level, which are the
    same at levels where the same fields have bits left.
    """
    fields = {**input_fields, **output_fields}
    lg_size = {
        tns: sum(lg_dims[idx.name] for idx in idxs) for tns, idxs in fields.items()
    }
    nnzs = {tns: round(np.sum(seeds[tns]) * 2 ** lg_size[tns]) for tns in input_fields}
    products = nnzs[A] * nnzs[B]
    done = 0
    for lg in sorted(set(lg_dims.values())):
        level = {}
        for tns, idxs in input_fields.items():
            S = np.asarray(seeds[tns]) / np.sum(seeds[tns])
            out = tuple(d for d, idx in enumerate(idxs) if lg_dims[idx.name] < lg)
            level[tns] = fl.asarray(S.sum(axis=out, keepdims=True))
        (prod,) = LogicInterpreter()(prgm, level)
        products *= np.sum(prod.to_numpy()) ** (lg - done)
        done = lg

    def fraction(nnz, lg):
        return min(np.log2(max(nnz, 1)) / lg, 1.0) if lg else 1.0

    return [
        fraction(nnzs[A], lg_size[A]),
        fraction(nnzs[B], lg_size[B]),
        fraction(products, lg_size[C]),
    ]


def spread(rng, points, n):
    """
    The indices of n of points, each the nearest to a uniform draw from the
    unit cube which no earlier draw took.
    """
    points = np.asarray(points)
    chosen: list[int] = []
    for target in rng.random((n, points.shape[1])):
        distance = np.linalg.norm(points - target, axis=1)
        distance[chosen] = np.inf
        chosen.append(int(np.argmin(distance)))
    return chosen


def program_inputs(rng, lg_max_space, lg_max_dim, samples, keep):
    """
    A drawn program, as its plan with a drawn loop order and the drawn formats
    of A, B, and C, paired with the inputs of each of keep of samples draws of
    their seeds, spread evenly over the log densities of A and B and the log
    expected products of C.
    """
    prgm, input_fields, output_fields, lg_dims = program(rng, lg_max_space, lg_max_dim)
    dtype = choose(rng, [np.float32, np.float64])
    seeds = [
        {tns: choose_seed(rng, idxs, lg_dims) for tns, idxs in input_fields.items()}
        for _ in range(samples)
    ]
    points = [
        seed_lg_densities(prgm, input_fields, output_fields, lg_dims, S) for S in seeds
    ]
    query, produces = prgm.bodies
    loop_order = shuffled(rng, tuple(Field(idx) for idx in lg_dims))
    plan = Plan((fuse_loop_order(query, loop_order), produces))
    formats = {
        tns.name: [choose(rng, list(OPTIONS)) for _ in idxs]
        for tns, idxs in {**input_fields, **output_fields}.items()
    }
    return [
        (
            (plan, formats),
            inputs(rng, input_fields, lg_dims, seeds[i], dtype),
        )
        for i in spread(rng, points, keep)
    ]


def coo(tns: np.ndarray, idx_type) -> FiberTensor:
    """A Finch COO tensor of the nonzeros of tns, indexed by idx_type."""
    fmt = element(tns.dtype.type(0), ftype(tns.dtype), ftype(idx_type))
    if not tns.ndim:
        return FiberTensor(ElementLevel(fmt, NumpyBuffer(tns.reshape(1))))
    indices = np.nonzero(tns)
    ptr = NumpyBuffer(np.array([0, indices[0].size], dtype=idx_type))
    return FiberTensor(
        SparseCOOLevel(
            ElementLevel(fmt, NumpyBuffer(tns[indices])),
            tuple(idx_type(n) for n in tns.shape),
            ptr,
            tuple(NumpyBuffer(idx.astype(idx_type)) for idx in indices),
        )
    )


def path(directory: Path, example, kind) -> Path:
    """The file of a kind, such as program.json, of an example in directory."""
    return directory / "data" / f"{example}.{kind}"


def examples(directory: Path, chunk=None) -> list[str]:
    """The names of the examples in directory, or of those of one chunk."""
    prefix = "*" if chunk is None else f"{chunk:04d}-*"
    return sorted(
        file.name.removesuffix(".program.json")
        for file in (directory / "data").glob(f"{prefix}.program.json")
    )


def load(
    directory: Path, example
) -> tuple[Plan, dict[Alias, list[LevelOption]], dict[Alias, Tensor]]:
    """
    The plan, formats, and inputs of an example which generate wrote. The inputs
    are COO, whatever their formats.
    """
    record = json.loads(path(directory, example, "program.json").read_text())
    plan = parse_logic(record["program"])
    assert isinstance(plan, Plan)
    formats: dict[Alias, list[LevelOption]] = {
        HardAlias(name): [OPTIONS[kind] for kind in kinds]
        for name, kinds in record["formats"].items()
    }
    bindings: dict[Alias, Tensor] = {
        tns: from_binsparse(
            bsp.load_binsparse(directory / "data" / record["paths"][tns.name])
        )
        for tns in (A, B)
    }
    return plan, formats, bindings


def generate(
    directory: Path, programs, chunk, lg_max_space, lg_max_dim, samples, keep, rng
) -> None:
    (directory / "data").mkdir(parents=True, exist_ok=True)
    example = 0
    for i in range(programs):
        start = time.perf_counter()
        pairs = program_inputs(rng, lg_max_space, lg_max_dim, samples, keep)
        for (plan, formats), tensors in pairs:
            name = f"{chunk:04d}-{example:05d}"
            idx_type = choose(rng, [np.int32, np.int64])
            C_dummy = np.zeros((0,) * len(formats["C"]), tensors[A].dtype)
            tensors = {**tensors, C: C_dummy}
            paths = {tns.name: f"{name}.{tns.name}.bsp.npz" for tns in tensors}
            record = {"program": str(plan), "formats": formats, "paths": paths}
            path(directory, name, "program.json").write_text(
                json.dumps(record, indent=2)
            )
            for tns, array in tensors.items():
                file = directory / "data" / paths[tns.name]
                bsp.save_binsparse(to_binsparse(coo(array, idx_type)), file)
            example += 1
        print(
            f"[{i + 1}/{programs}] {len(pairs)} examples through {name}, "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--programs", type=int, default=10)
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="the draws of the seeds of the inputs of each program",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="the draws of the inputs of each program which are kept as examples",
    )
    parser.add_argument(
        "--chunk", type=int, default=0, help="the chunk which names the examples"
    )
    parser.add_argument(
        "--lg-max-space",
        type=int,
        default=24,
        help="log2 of the most entries all the fields of a program may address",
    )
    parser.add_argument(
        "--lg-max-dim",
        type=int,
        default=11,
        help="log2 of the most values of any one field",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.keep > args.samples:
        parser.error("--keep may be at most --samples")
    # Chunks drawn with one seed still draw different programs.
    rng = np.random.default_rng(None if args.seed is None else (args.seed, args.chunk))
    generate(
        args.directory,
        args.programs,
        args.chunk,
        args.lg_max_space,
        args.lg_max_dim,
        args.samples,
        args.keep,
        rng,
    )


if __name__ == "__main__":
    main()
