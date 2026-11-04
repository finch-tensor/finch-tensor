"""
Draw programs for fitting the cost model of a stage of physical_bases, and
their inputs. Each program is a random subprogram of C_lmq += A_jlpq * B_kmpq,
which loops in the order its inputs store their fields, and its inputs are
R-MAT tensors of drawn densities. Its variants are every option of the
decision of the stage: the protocols A and B are read with under drawn
formats, the formats of A, B, and C, the loop orders of the program, with its
inputs stored in each, or, in the transpose stage, the orders a copy C of A
can be stored in, which reads A as it is stored, under drawn formats. With
--variants N, N of them are drawn without replacement when there are more.
The log sizes of the fields of a program add up to a fraction of the most
they may, under --lg-max-space and --lg-max-dim, and the log nonzeros of each
input to a fraction of its log size. Each fraction is drawn as the largest of
--size-bias or --density-bias uniform draws, which is uniform for a bias of 1
and skews toward larger or denser inputs above it. Of the samples draws of the
inputs of a program, keep are kept, spread over the densities of A, B, and C
by the same distribution, and each makes an example with the program, whose
variants compile to the same kernels, so the kernels are compiled once for
them all.
Each example is written to DIR/data as JSON, with the text of its query, its
loop order, its variants, and the R-MAT seed, log shape, and RNG seed of each
of its inputs, under the name of its chunk and its index in the chunk, so jobs
of different chunks can share DIR. No tensors are stored: load draws the
inputs from their seeds again, as COO, in memory proportional to their
nonzeros.

    pixi run -e fitting python -m scripts.fitting.generate_physical_data \\
        DIR [--stage S] [--programs N] [--samples N] [--keep N] [--chunk K] \\
        [--variants N] [--lg-max-space N] [--lg-max-dim N] [--size-bias D] \\
        [--density-bias D] [--seed S]
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import Tensor, ftype
from finch.autoschedule.cost_models import (
    allowed_options,
    read_options,
)
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS
from finch.codegen import NumpyBuffer
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
)
from finch.tensor import ElementLevel, FiberTensor, SparseCOOLevel, element
from scripts.fitting.physical_bases import (
    INPUTS,
    OPTIONS,
    STAGES,
    A,
    B,
    C,
    fused_plan,
    with_subsequence,
)


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


def rmat(S, lg_shape, dtype, idx_type, rng) -> FiberTensor:
    """
    A COO tensor of shape 2 ** lg_shape, indexed by idx_type, whose
    S.sum() * 2 ** sum(lg_shape) nonzeros are the first distinct cells which
    balls dropped by R-MAT with the seed S land in.
    """
    # Values are in (0, 1], so neither a sum nor a product of them is zero.
    shape = tuple(2**lg for lg in lg_shape)
    if not lg_shape:
        return coo(shape, (), np.array([1 - rng.random()], dtype), idx_type)
    lg_size = sum(lg_shape)
    nnz = round(np.sum(S) * 2**lg_size)
    # A ball lands in one of about 2 ** H cells, for the entropy H of the cell
    # it lands in. When nnz is well under that and under the size, balls are
    # dropped until nnz cells are hit, and otherwise every cell is ordered,
    # which takes memory for each cell.
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -sum(np.nansum(P * np.log2(P)) for _, P in levels(S, lg_shape))
    if np.log2(nnz) <= min(H - 2, lg_size - 5):
        cells = drop_balls(S, lg_shape, nnz, rng)
    else:
        # The order in which balls first land in cells is the order of
        # exponential draws over the probabilities of the cells.
        with np.errstate(divide="ignore"):
            keys = (
                rng.exponential(size=2**lg_size)
                / cell_probabilities(S, lg_shape).ravel()
            )
        cells = np.argpartition(keys, nnz - 1)[:nnz]
    cells = np.sort(cells)
    values = (1 - rng.random(nnz)).astype(dtype)
    return coo(shape, np.unravel_index(cells, shape), values, idx_type)


def program(rng, lg_max_space, lg_max_dim, size_bias):
    """
    A random subprogram of C_lmq += A_jlpq * B_kmpq, and its loop order, which
    is the order its inputs store their fields in. C has no fields of its own,
    since nothing would give them extents. The log sizes of its fields add up
    to skewed_fractions of the most they may, under lg_max_space and
    lg_max_dim.
    """
    idxs = [idx for idx in "jklmpq" if rng.random() < 0.5]
    lg_space = round(
        skewed_fractions(rng, size_bias) * min(lg_max_space, lg_max_dim * len(idxs))
    )
    # Each field has a bit, and the rest are dealt to the fields with room by
    # weights of a Dirichlet draw, so the fields of a program vary in size.
    lg_dims = dict.fromkeys(idxs, 1)
    if idxs:
        weights = rng.dirichlet(np.ones(len(idxs)))
        for _ in range(lg_space - len(idxs)):
            room = weights * [lg_dims[idx] < lg_max_dim for idx in idxs]
            lg_dims[idxs[rng.choice(len(idxs), p=room / room.sum())]] += 1
    order = shuffled(rng, idxs)
    # The inputs store their fields in loop order, since the loop orderer
    # transposes an input it would read out of order, and only the output may
    # be out of order.
    A_idxs = tuple(Field(idx) for idx in with_subsequence(order, "jlpq"))
    B_idxs = tuple(Field(idx) for idx in with_subsequence(order, "kmpq"))
    C_idxs = tuple(Field(idx) for idx in with_subsequence(order, "lmq"))
    if rng.random() < 0.1:
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
    loop_order = tuple(Field(idx) for idx in order)
    return prgm, input_fields, output_fields, lg_dims, loop_order


def skewed_fractions(rng, bias, size=None):
    """
    Fractions in [0, 1] distributed as the largest of bias uniform draws, so
    a bias of 1 is uniform and a larger bias skews toward 1.
    """
    return rng.random(size) ** (1 / bias)


def choose_seed(rng, idxs, lg_dims, density_bias):
    skews = [rng.choice([-1, 1]) * rng.random() for _ in idxs]
    lg_size = sum(lg_dims[idx.name] for idx in idxs)
    sparsity = 2 ** -((1 - skewed_fractions(rng, density_bias)) * lg_size)
    return seed(skews, sparsity)


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


def spread(rng, points, n, density_bias):
    """
    The indices of n of points, each the nearest to a draw from the unit cube
    which no earlier draw took, whose coordinates are skewed_fractions.
    """
    points = np.asarray(points)
    chosen: list[int] = []
    for target in skewed_fractions(rng, density_bias, (n, points.shape[1])):
        distance = np.linalg.norm(points - target, axis=1)
        distance[chosen] = np.inf
        chosen.append(int(np.argmin(distance)))
    return chosen


def output_kinds(query: Query, loop_order: tuple[Field, ...]) -> list[list[str]]:
    """The kinds each level of C can take under loop_order, which writes it."""
    plan = fused_plan(query, loop_order)
    return [
        [opt.kind for opt in allowed_options(C, depth, plan, LEVEL_OPTIONS)]
        for depth in range(len(query.lhs.idxs))
    ]


def read_kinds(idxs, loop_order) -> list[list[tuple[str, str]]]:
    """
    The kinds and protocols each level of an input of idxs can take under
    loop_order, as pairs of their names.
    """
    return [
        [
            (option.kind, str(proto))
            for option in LEVEL_OPTIONS
            for proto in read_options(option, idxs, 0, depth, loop_order)
        ]
        for depth in range(len(idxs))
    ]


def variant_space(
    stage: str, query: Query, input_fields, loop_order, rng
) -> tuple[list[list[Any]], Any]:
    """
    The choices of each level, or of the program, which make up the variants
    of a stage, and a function of one choice for each which makes the variant.
    The variants are the product of the choices.
    """
    if stage == "protocol":
        # The formats are drawn once, and every protocol of them is a variant.
        formats = {
            tns.name: [choose(rng, list(OPTIONS)) for _ in idxs]
            for tns, idxs in input_fields.items()
        }
        formats[C.name] = [
            choose(rng, kinds) for kinds in output_kinds(query, loop_order)
        ]
        spaces = [
            [
                proto
                for kind, proto in read_kinds(idxs, loop_order)[depth]
                if kind == formats[tns.name][depth]
            ]
            for tns, idxs in input_fields.items()
            for depth in range(len(idxs))
        ]

        def make(choices):
            protocols = {}
            for tns, idxs in input_fields.items():
                protocols[tns.name] = [choices.pop(0) for _ in idxs]
            return {"formats": formats, "protocols": protocols}

        return spaces, make
    if stage == "level":
        spaces = [list(OPTIONS) for idxs in input_fields.values() for _ in idxs]
        spaces += output_kinds(query, loop_order)

        def make(choices):
            formats = {}
            for tns, idxs in input_fields.items():
                formats[tns.name] = [choices.pop(0) for _ in idxs]
            formats[C.name] = choices
            return {"formats": formats}

        return spaces, make
    if stage == "loop":
        orders = [
            [idx.name for idx in order] for order in itertools.permutations(loop_order)
        ]
        return [orders], lambda choices: {"loop_order": choices[0]}
    if stage == "transpose":
        # C copies A, stored in each order of A's fields, under a drawn format
        # of A and, for each order, of C, whose levels the copy may laminate.
        A_idxs = input_fields[A]
        formats = {A.name: [choose(rng, list(OPTIONS)) for _ in A_idxs]}
        orders = [list(order) for order in itertools.permutations(A_idxs)]

        def make(choices):
            (order,) = choices
            copy = Query(Table(C, tuple(order)), Table(A, A_idxs))
            kinds = output_kinds(copy, tuple(order))
            return {
                "loop_order": [idx.name for idx in order],
                "formats": {**formats, C.name: [choose(rng, k) for k in kinds]},
            }

        return [orders], make
    raise ValueError(f"Unknown stage {stage}, expected one of {STAGES}")


def variants(stage, query, input_fields, loop_order, rng, limit) -> tuple[list, bool]:
    """
    The variants of the decision of stage for query under loop_order, and
    whether they are all of them, or limit drawn without replacement.
    """
    spaces, make = variant_space(stage, query, input_fields, loop_order, rng)
    total = math.prod(map(len, spaces))
    exhaustive = limit is None or total <= limit
    if exhaustive:
        indices = range(total)
    else:
        indices = sorted(rng.choice(total, size=limit, replace=False).tolist())
    result = []
    for index in indices:
        # The index of a variant is mixed radix over the choices of each space.
        choices = []
        for space in reversed(spaces):
            index, choice = divmod(index, len(space))
            choices.append(space[choice])
        result.append(make(choices[::-1]))
    return result, exhaustive


def program_inputs(
    rng,
    stage,
    lg_max_space,
    lg_max_dim,
    samples,
    keep,
    limit,
    density_bias,
    size_bias,
):
    """
    The records of the examples of a drawn program, each with its query, its
    loop order, its variants, and the seeds of its inputs, which are keep of
    samples draws, spread over the log densities of A and B and the log
    expected products of C, skewed toward denser inputs by density_bias. The
    program is skewed toward larger fields by size_bias.
    """
    prgm, input_fields, output_fields, lg_dims, loop_order = program(
        rng, lg_max_space, lg_max_dim, size_bias
    )
    dtype = choose(rng, ["float32", "float64"])
    seeds = [
        {
            tns: choose_seed(rng, idxs, lg_dims, density_bias)
            for tns, idxs in input_fields.items()
        }
        for _ in range(samples)
    ]
    points = [
        seed_lg_densities(prgm, input_fields, output_fields, lg_dims, S) for S in seeds
    ]
    query, _ = prgm.bodies
    assert isinstance(query, Query)
    if stage == "transpose":
        # The transpose stage copies A as it is stored, and its variants store
        # the copy in every order.
        query = Query(Table(C, input_fields[A]), Table(A, input_fields[A]))
        loop_order = input_fields[A]
    options, exhaustive = variants(stage, query, input_fields, loop_order, rng, limit)
    record = {
        "stage": stage,
        "query": str(Plan((query, Produces((C,))))),
        "loop_order": [idx.name for idx in loop_order],
        "exhaustive": exhaustive,
        "variants": options,
    }
    return [
        {
            **record,
            "dtype": dtype,
            "idx_type": choose(rng, ["int32", "int64"]),
            "inputs": {
                tns.name: {
                    "rmat_seed": seeds[i][tns].tolist(),
                    "lg_shape": [lg_dims[idx.name] for idx in idxs],
                    "rng_seed": int(rng.integers(2**63)),
                }
                for tns, idxs in input_fields.items()
            },
        }
        for i in spread(rng, points, keep, density_bias)
    ]


def coo(shape, coords, values: np.ndarray, idx_type) -> FiberTensor:
    """
    A Finch COO tensor of shape with values at coords, indexed by idx_type,
    with positions of intp, which counts the nonzeros of any tensor.
    """
    fmt = element(values.dtype.type(0), ftype(values.dtype), ftype(np.intp))
    if not shape:
        return FiberTensor(ElementLevel(fmt, NumpyBuffer(values)))
    ptr = NumpyBuffer(np.array([0, values.size], dtype=np.intp))
    return FiberTensor(
        SparseCOOLevel(
            ElementLevel(fmt, NumpyBuffer(values)),
            tuple(idx_type(n) for n in shape),
            ptr,
            tuple(NumpyBuffer(idx.astype(idx_type)) for idx in coords),
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


def expand(record: dict[str, Any], tns: Alias) -> Tensor:
    """The input tns of the example of record, drawn from its seeds."""
    spec = record["inputs"][tns.name]
    return rmat(
        np.asarray(spec["rmat_seed"]),
        spec["lg_shape"],
        np.dtype(record["dtype"]).type,
        np.dtype(record["idx_type"]).type,
        np.random.default_rng(spec["rng_seed"]),
    )


def load(directory: Path, example) -> tuple[dict[str, Any], dict[Alias, Tensor]]:
    """
    The record and the inputs of an example which generate wrote. The inputs
    are COO, whatever their formats.
    """
    record = json.loads(path(directory, example, "program.json").read_text())
    return record, {tns: expand(record, tns) for tns in INPUTS}


def write_example(directory: Path, name, record) -> None:
    path(directory, name, "program.json").write_text(json.dumps(record, indent=2))


def generate(
    directory: Path,
    stage,
    programs,
    chunk,
    lg_max_space,
    lg_max_dim,
    samples,
    keep,
    limit,
    density_bias,
    size_bias,
    rng,
) -> None:
    (directory / "data").mkdir(parents=True, exist_ok=True)
    example = 0
    for i in range(programs):
        start = time.perf_counter()
        records = program_inputs(
            rng,
            stage,
            lg_max_space,
            lg_max_dim,
            samples,
            keep,
            limit,
            density_bias,
            size_bias,
        )
        for record in records:
            name = f"{chunk:04d}-{example:05d}"
            write_example(directory, name, record)
            example += 1
        record = records[0]
        drawn = "all" if record["exhaustive"] else f"{len(record['variants'])} of"
        print(
            f"[{i + 1}/{programs}] {len(records)} examples through {name}, "
            f"{drawn} variants, {time.perf_counter() - start:.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--stage", choices=STAGES, default="protocol", help="the decision to vary"
    )
    parser.add_argument("--programs", type=int, default=512)
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
        "--variants",
        type=int,
        help="the most variants of a program; all of them by default",
    )
    parser.add_argument(
        "--lg-max-space",
        type=int,
        default=30,
        help="log2 of the most entries all the fields of a program may address",
    )
    parser.add_argument(
        "--lg-max-dim",
        type=int,
        default=15,
        help="log2 of the most values of any one field",
    )
    parser.add_argument(
        "--density-bias",
        type=float,
        default=3.0,
        help="the uniform draws whose largest is the log density of an input",
    )
    parser.add_argument(
        "--size-bias",
        type=float,
        default=3.0,
        help="the uniform draws whose largest is the log size of a program",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.keep > args.samples:
        parser.error("--keep may be at most --samples")
    # Chunks drawn with one seed still draw different programs.
    rng = np.random.default_rng(None if args.seed is None else (args.seed, args.chunk))
    generate(
        args.directory,
        args.stage,
        args.programs,
        args.chunk,
        args.lg_max_space,
        args.lg_max_dim,
        args.samples,
        args.keep,
        args.variants,
        args.density_bias,
        args.size_bias,
        rng,
    )


if __name__ == "__main__":
    main()
