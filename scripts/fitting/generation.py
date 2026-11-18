"""
What the generators of programs for fitting share: the random draws they make,
the seeds of the Kronecker products of R-MAT blocks their inputs are drawn as,
the records of their examples, and how a step writes them to DIR/data and
reads them back. Each generator, generate_protocol_data, generate_loop_data,
and generate_factorize_data, draws its own programs, and their variants, with
these.
"""

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import finch as fl
from finch.algebra import ftype
from finch.codegen import NumpyBuffer
from finch.finch_logic import Alias, HardAlias, LogicInterpreter
from finch.tensor import ElementLevel, FiberTensor, SparseCOOLevel, element
from scripts.fitting.kronecker_stats import (
    KroneckerTensor,
    block_shapes,
    kronecker_rmat,
)

# Each dimension is split into digits of BLOCK_BITS bits, and the block of each
# digit is drawn by R-MAT.
BLOCK_BITS = 4


def choose(rng, options):
    return options[rng.integers(len(options))]


def shuffled(rng, items):
    return tuple(items[p] for p in rng.permutation(len(items)))


def skewed_fractions(rng, draws, size=None):
    """
    Fractions in [0, 1] distributed as the largest of draws uniform draws,
    which skews toward 1 the more draws there are: toward larger programs and
    denser inputs.
    """
    return rng.random(size) ** (1 / draws)


def block_seeds(S, lg_shape) -> list[np.ndarray]:
    """
    The seed of each block of a tensor of shape 2 ** lg_shape, with the skew of
    the seed S and a share of its log density for each bit of the block, so
    the density of the tensor is that of S.
    """
    density = np.sum(S)
    lg_size = sum(lg_shape)
    return [
        S / density * density ** (math.log2(math.prod(shape)) / lg_size)
        for shape in block_shapes(lg_shape, BLOCK_BITS)
    ]


def input_record(S, lg_shape, rng) -> dict[str, Any]:
    """
    The seeds of an input of shape 2 ** lg_shape with the seed S, and the
    version of numpy which draws its blocks from them and their checksum, which
    record_inputs checks when it draws them again.
    """
    seeds = block_seeds(S, lg_shape)
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


def seed_lg_densities(prgm, input_fields, lg_dims, seeds):
    """
    The log nonzeros of the inputs whose blocks block_seeds splits from seeds,
    and the log expected products of the output of prgm, each as a fraction of
    its log size. By the mixed product property, the expected products are the
    product over levels of the sum of prgm run on the seeds of the level, which
    are the same at levels where the same fields have bits left.
    """
    query, _ = prgm.bodies
    output = query.lhs.tns
    fields = {**input_fields, output: query.lhs.idxs}
    lg_size = {
        tns: sum(lg_dims[idx.name] for idx in idxs) for tns, idxs in fields.items()
    }
    nnzs = {tns: round(np.sum(seeds[tns]) * 2 ** lg_size[tns]) for tns in input_fields}
    products = math.prod(nnzs.values())
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
        *(fraction(nnzs[tns], lg_size[tns]) for tns in input_fields),
        fraction(products, lg_size[output]),
    ]


def spread(rng, points, n, draws):
    """
    The indices of n of points, each the nearest to a draw from the unit cube
    which no earlier draw took, whose coordinates are skewed_fractions of
    draws draws.
    """
    points = np.asarray(points)
    chosen: list[int] = []
    for target in skewed_fractions(rng, draws, (n, points.shape[1])):
        distance = np.linalg.norm(points - target, axis=1)
        distance[chosen] = np.inf
        chosen.append(int(np.argmin(distance)))
    return chosen


def variants(spaces, make, rng, limit) -> tuple[list, bool]:
    """
    The variants which make makes of one choice of each of spaces, all of
    them, or limit of them drawn without replacement where there are more, and
    whether they are all of them.
    """
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


def path(directory: Path, example, kind) -> Path:
    return directory / "data" / f"{example}.{kind}"


def examples(directory: Path, chunk=None, kind="program.json") -> list[str]:
    prefix = "*" if chunk is None else f"{chunk:04d}-*"
    return sorted(
        file.name.removesuffix(f".{kind}")
        for file in (directory / "data").glob(f"{prefix}.{kind}")
    )


def record_inputs(record: dict[str, Any]) -> dict[Alias, KroneckerTensor]:
    inputs = {}
    for name, spec in record["inputs"].items():
        tns = HardAlias(name)
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
        inputs[tns] = tensor
    return inputs


def materialize(tensor: KroneckerTensor) -> FiberTensor:
    # Values are in (0, 1], so neither a sum nor a product of them is zero.
    values = 1 - np.random.default_rng(0).random(tensor.nnz)
    values = values.astype(tensor.dtype)
    # Positions are intp, which counts the nonzeros of any tensor.
    fmt = element(values.dtype.type(0), ftype(values.dtype), ftype(np.intp))
    if not tensor.shape:
        return FiberTensor(ElementLevel(fmt, NumpyBuffer(values)))
    ptr = NumpyBuffer(np.array([0, values.size], dtype=np.intp))
    return FiberTensor(
        SparseCOOLevel(
            ElementLevel(fmt, NumpyBuffer(values)),
            tuple(tensor.idx_type(n) for n in tensor.shape),
            ptr,
            tuple(NumpyBuffer(idx.astype(tensor.idx_type)) for idx in tensor.coords()),
        )
    )


def generate(
    directory: Path,
    programs,
    chunk,
    rng,
    draw: Callable[[np.random.Generator], list[dict[str, Any]]],
) -> None:
    """
    Write the records of the examples of programs programs, each a list which
    draw draws, to DIR/data as JSON, under the name of their chunk and their
    index in it, so jobs of different chunks can share DIR.
    """
    (directory / "data").mkdir(parents=True, exist_ok=True)
    example = 0
    for i in range(programs):
        start = time.perf_counter()
        records = draw(rng)
        for record in records:
            name = f"{chunk:04d}-{example:05d}"
            path(directory, name, "program.json").write_text(
                json.dumps(record, indent=2)
            )
            example += 1
        record = records[0]
        drawn = "all" if record["exhaustive"] else f"{len(record['variants'])} of"
        print(
            f"[{i + 1}/{programs}] {len(records)} examples through {name}, "
            f"{drawn} variants, {time.perf_counter() - start:.1f}s",
            flush=True,
        )


def main(
    description: str,
    stages: tuple[str, ...],
    program_inputs: Callable[[np.random.Generator, str], list[dict[str, Any]]],
) -> None:
    """The command line of a generator of the examples of stages, each of a
    program whose records program_inputs draws."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--stage", choices=stages, default=stages[0], help="the decision to vary"
    )
    parser.add_argument("--programs", type=int, default=512)
    parser.add_argument(
        "--chunk", type=int, default=0, help="the chunk which names the examples"
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    # Chunks drawn with one seed still draw different programs.
    rng = np.random.default_rng(None if args.seed is None else (args.seed, args.chunk))
    generate(
        args.directory,
        args.programs,
        args.chunk,
        rng,
        lambda rng: program_inputs(rng, args.stage),
    )
