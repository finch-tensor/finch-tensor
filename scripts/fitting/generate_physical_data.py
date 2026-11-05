"""
Draw programs for fitting the cost model of a stage of physical_bases, and
their inputs. Each program is a random subprogram of C_lmq += A_jlpq * B_kmpq,
which loops in the order its inputs store their fields, and its inputs are
Kronecker products of R-MAT blocks of drawn densities, as kronecker_stats draws
them. A and B have at most MAX_RANK fields each, which bounds the variants of
the joint stage. Its variants are every option of the decision of the stage:
the protocols A and B are read with under drawn formats, the formats and
protocols of one of A, B, and C with the protocols of the loops over its
fields, under drawn formats and protocols of the rest, the formats of one of
A, B, and C under drawn formats of the others, the loop orders of the program,
with its inputs stored in each, or, in the transpose stage, the orders a copy
C of A can be stored in, which reads A as it is stored, under drawn formats.
With --variants N, N of them are drawn without replacement when there are
more, by default all of them, except 64 of the joint stage, which has up to
6561, and 16 of the loop orders of the loop stage, which has up to 720.
The log sizes of the fields of a program add up to a fraction of the most
they may, under --lg-max-space and --lg-max-dim, and the log nonzeros of each
input to a fraction of its log size. Each fraction is drawn as the largest of
--size-bias or --density-bias uniform draws, which is uniform for a bias of 1
and skews toward larger or denser inputs above it. Of the samples draws of the
inputs of a program, keep are kept, spread over the densities of A, B, and C
by the same distribution, and each makes an example with the program. In the
protocol and joint stages, the variants of the examples of a program compile
to the same kernels, so they are compiled once for them all, and 10 are kept
by default.
The other stages choose protocols from the stats of each example, so their
kernels can differ, and one is kept by default, for more programs instead.
Each example is written to DIR/data as JSON, with the text of its query, its
loop order, its variants, and the log shape, the R-MAT seed of each block, and
the RNG seed of each of its inputs, with the version of numpy which drew its
blocks and their checksum, under the name of its chunk and its index in the
chunk, so jobs of different chunks can share DIR. No tensors are stored: load
draws the blocks of the inputs from their seeds again, checking that they
match, and materialize expands them into COO tensors.

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
from finch.algebra import ftype
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
from scripts.fitting.kronecker_stats import (
    KroneckerTensor,
    block_shapes,
    kronecker_rmat,
)
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

# The most fields of an input, which bounds the variants of the joint stage.
MAX_RANK = 3


def choose(rng, options):
    return options[rng.integers(len(options))]


def shuffled(rng, items):
    return tuple(items[p] for p in rng.permutation(len(items)))


def seed(skew, sparsity):
    S = np.ones(())
    for s in skew:
        S = np.multiply.outer(S, [0.5 - s / 2, 0.5 + s / 2])
    return S * sparsity


# Each dimension is split into digits of BLOCK_BITS bits, and the block of each
# digit is drawn by R-MAT.
BLOCK_BITS = 4


def block_seeds(S, lg_shape, block_bits) -> list[np.ndarray]:
    """
    The seed of each block of a tensor of shape 2 ** lg_shape, with the skew of
    the seed S and a share of its log density for each bit of the block, so
    the density of the tensor is that of S.
    """
    density = np.sum(S)
    lg_size = sum(lg_shape)
    return [
        S / density * density ** (math.log2(math.prod(shape)) / lg_size)
        for shape in block_shapes(lg_shape, block_bits)
    ]


def input_record(S, lg_shape, rng) -> dict[str, Any]:
    """
    The seeds of an input of shape 2 ** lg_shape with the seed S, and the
    version of numpy which draws its blocks from them and their checksum, which
    kronecker_input checks when it draws them again.
    """
    seeds = block_seeds(S, lg_shape, BLOCK_BITS)
    rng_seed = int(rng.integers(2**63))
    tensor = kronecker_rmat(
        lg_shape, seeds, np.random.default_rng(rng_seed), BLOCK_BITS
    )
    return {
        "lg_shape": lg_shape,
        "rmat_seeds": [b.tolist() for b in seeds],
        "rng_seed": rng_seed,
        "numpy": np.__version__,
        "checksum": tensor.checksum(),
    }


def program(rng, lg_max_space, lg_max_dim, size_bias):
    """
    A random subprogram of C_ilmq += A_jlpq * B_kmpq, and its loop order, which
    is the order its inputs store their fields in. C has no fields of its own,
    since nothing would give them extents. The log sizes of its fields add up
    to skewed_fractions of the most they may, under lg_max_space and
    lg_max_dim. A program whose A or B has more than MAX_RANK fields is drawn
    again.
    """
    while True:
        idxs = [idx for idx in "jklmpq" if rng.random() < 0.5]
        if all(len(set(idxs) & set(tns)) <= MAX_RANK for tns in ("jlpq", "kmpq")):
            break
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
    The log nonzeros of the inputs whose blocks block_seeds splits from seeds,
    and the log expected products of the output of prgm, each as a fraction of
    its log size. By the mixed product property, the expected products are the
    product over levels of the sum of prgm run on the seeds of the level, which
    are the same at levels where the same fields have bits left.
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
    if stage == "joint":
        # One drawn tensor takes every kind and protocol of its levels, and
        # the formatter re-chooses the protocols of the loops it changes, so
        # the levels of the other input read at the loops over its fields take
        # every protocol of their drawn kinds, and the rest take drawn ones.
        fields = {**input_fields, C: query.lhs.idxs}
        varied = choose(rng, [tns for tns, idxs in fields.items() if idxs] or [C])
        spaces = []
        for tns, idxs in input_fields.items():
            for depth, pairs in enumerate(read_kinds(idxs, loop_order)):
                if tns != varied:
                    kind = choose(rng, list(OPTIONS))
                    pairs = [pair for pair in pairs if pair[0] == kind]
                    if idxs[depth] not in fields[varied]:
                        pairs = [choose(rng, pairs)]
                spaces.append(pairs)
        kinds = output_kinds(query, loop_order)
        if varied != C:
            kinds = [[choose(rng, options)] for options in kinds]
        spaces.extend(kinds)

        def make(choices):
            formats, protocols = {}, {}
            for tns, idxs in input_fields.items():
                pairs = [choices.pop(0) for _ in idxs]
                formats[tns.name] = [kind for kind, _ in pairs]
                protocols[tns.name] = [proto for _, proto in pairs]
            formats[C.name] = choices
            return {"formats": formats, "protocols": protocols}

        return spaces, make
    if stage == "level":
        # The formatter chooses the formats of one tensor at a time, given the
        # others, so one drawn tensor takes every format of its levels, and
        # the others take drawn formats.
        kinds = {
            tns: [list(OPTIONS) for _ in idxs] for tns, idxs in input_fields.items()
        }
        kinds[C] = output_kinds(query, loop_order)
        varied = choose(rng, [tns for tns, levels in kinds.items() if levels] or [C])
        formats = {
            tns.name: [choose(rng, options) for options in levels]
            for tns, levels in kinds.items()
            if tns != varied
        }

        def make(choices):
            return {"formats": {**formats, varied.name: choices}}

        return kinds[varied], make
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
            "block_bits": BLOCK_BITS,
            "inputs": {
                tns.name: input_record(
                    seeds[i][tns], [lg_dims[idx.name] for idx in idxs], rng
                )
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


def write_json(file: Path, data) -> None:
    # A job killed while writing leaves the file as it was, rather than cut off.
    partial = file.with_name(f"{file.name}.partial")
    partial.write_text(json.dumps(data))
    partial.replace(file)


def examples(directory: Path, chunk=None, kind="program.json") -> list[str]:
    """
    The names of the examples in directory, or of those of one chunk, which
    have a file of kind.
    """
    prefix = "*" if chunk is None else f"{chunk:04d}-*"
    return sorted(
        file.name.removesuffix(f".{kind}")
        for file in (directory / "data").glob(f"{prefix}.{kind}")
    )


def kronecker_input(record: dict[str, Any], tns: Alias) -> KroneckerTensor:
    """
    The input tns of the example of record, its blocks drawn from their seeds,
    which must match the checksum of the blocks generate drew.
    """
    spec = record["inputs"][tns.name]
    tensor = kronecker_rmat(
        spec["lg_shape"],
        [np.asarray(S) for S in spec["rmat_seeds"]],
        np.random.default_rng(spec["rng_seed"]),
        record["block_bits"],
        np.dtype(record["dtype"]).type,
        np.dtype(record["idx_type"]).type,
    )
    if tensor.checksum() != spec["checksum"]:
        raise ValueError(
            f"The blocks of {tns.name} drawn with numpy {np.__version__} differ "
            f"from those generate drew with numpy {spec['numpy']}"
        )
    return tensor


def materialize(tensor: KroneckerTensor) -> FiberTensor:
    """A COO tensor of the support of tensor."""
    # Values are in (0, 1], so neither a sum nor a product of them is zero.
    values = 1 - np.random.default_rng(0).random(tensor.nnz)
    return coo(
        tensor.shape, tensor.coords(), values.astype(tensor.dtype), tensor.idx_type
    )


def load(
    directory: Path, example
) -> tuple[dict[str, Any], dict[Alias, KroneckerTensor]]:
    """The record of an example which generate wrote, and its inputs."""
    record = json.loads(path(directory, example, "program.json").read_text())
    return record, record_inputs(record)


def record_inputs(record: dict[str, Any]) -> dict[Alias, KroneckerTensor]:
    """The inputs of the example of record, drawn from their seeds."""
    return {tns: kronecker_input(record, tns) for tns in INPUTS}


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
        help="the draws of the inputs of each program which are kept as examples; "
        "10 in the protocol and joint stages and 1 in the others by default",
    )
    parser.add_argument(
        "--chunk", type=int, default=0, help="the chunk which names the examples"
    )
    parser.add_argument(
        "--variants",
        type=int,
        help="the most variants of a program; all of them by default, except 64 "
        "in the joint stage and 16 in the loop stage",
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
    if args.keep is None:
        args.keep = 10 if args.stage in ("protocol", "joint") else 1
    if args.variants is None:
        args.variants = {"joint": 64, "loop": 16}.get(args.stage)
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
