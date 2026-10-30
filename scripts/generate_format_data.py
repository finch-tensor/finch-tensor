"""
Draw programs for fitting the formatter's cost model, and their inputs. Each
program is a random subprogram of C_lmq += A_jlpq * B_kmpq, with a drawn loop
order and formats of C, and its inputs are R-MAT tensors which give C a drawn
sparsity. Each program is written to DIR as JSON, with the text of its logic
plan and the formats of C, and its inputs as binsparse, under the name of its
chunk and its index in the chunk, so jobs of different chunks can share DIR.

    pixi run -e fitting python -m scripts.generate_format_data \
        DIR [--examples N] [--chunk K] [--lg-max-space N] [--lg-max-dim N] \
        [--seed S]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import binsparse as bsp

import finch as fl
from finch import ffuncs
from finch.algebra import Tensor, ftype, ftypes
from finch.autoschedule.cost_models import LevelOption
from finch.autoschedule.default_schedulers import COMPILE_NUMBA
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
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
    parse_logic,
)
from finch.tensor import ElementLevel, FiberTensor, SparseCOOLevel, element

OPS = {"add": ffuncs.add, "mul": ffuncs.mul}
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


def rmat(S, lg_shape, rng):
    """
    A dense array of S.sum() * 2 ** sum(lg_shape) balls dropped into a tensor of
    shape 2 ** lg_shape. At each level a ball picks an entry of the seed S, and
    each axis with bits left takes that entry's bit along it, so an axis which
    has run out of bits marginalizes S over it.
    """
    # Values are in (0, 1], so neither a sum nor a product of them is zero.
    if not lg_shape:
        return np.array(1 - rng.random())
    S = np.asarray(S)
    nnz = round(S.sum() * 2 ** sum(lg_shape))
    picks = rng.choice(S.size, size=(nnz, max(lg_shape)), p=S.ravel() / S.sum())
    bits = np.unravel_index(picks, S.shape)
    coords = tuple(
        (b[:, :lg] << np.arange(lg - 1, -1, -1)).sum(axis=1)
        for b, lg in zip(bits, lg_shape, strict=True)
    )
    tns = np.zeros(tuple(2**lg for lg in lg_shape))
    tns[coords] = 1 - rng.random(nnz)
    return tns


def program(rng, lg_max_space, lg_max_dim):
    """
    A random subprogram of C_lmq += A_jlpq * B_kmpq, where * is add or mul. C
    has no fields of its own, since nothing would give them extents.
    """
    while True:
        idxs = [idx for idx in "jklmpq" if rng.random() < 0.5]
        lg_dims = {idx: int(rng.integers(1, lg_max_dim + 1)) for idx in idxs}
        if sum(lg_dims.values()) <= lg_max_space:
            break
    order = shuffled(rng, idxs)
    A_idxs = tuple(Field(idx) for idx in with_subsequence(order, "jlpq"))
    if rng.random() < 0.2:
        A_idxs = shuffled(rng, A_idxs)
    B_idxs = tuple(Field(idx) for idx in with_subsequence(order, "kmpq"))
    if rng.random() < 0.2:
        B_idxs = shuffled(rng, B_idxs)
    C_idxs = tuple(Field(idx) for idx in with_subsequence(order, "lmq"))
    if rng.random() < 0.2:
        C_idxs = shuffled(rng, C_idxs)
    r_idxs = tuple(Field(idx) for idx in with_subsequence(order, "jkp"))
    op = rng.choice([ffuncs.add, ffuncs.mul])
    prgm = Plan.from_children(
        Query(
            Table(HardAlias("C"), C_idxs),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0.0),
                MapJoin.from_children(
                    Literal(op),
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


def inputs(rng, prgm, input_fields, output_fields, lg_dims):
    samples = []
    for _ in range(100):
        seeds = {
            name: choose_seed(rng, idxs, lg_dims) for name, idxs in input_fields.items()
        }
        inputs = {
            name: rmat(S, [lg_dims[idx.name] for idx in input_fields[name]], rng)
            for name, S in seeds.items()
        }
        # The default scheduler interprets, which is too slow to run this often.
        outputs = COMPILE_NUMBA(
            prgm, {name: fl.asarray(tns) for name, tns in inputs.items()}
        )
        nnzs = {
            name: np.count_nonzero(np.asarray(output))
            for name, output in zip(output_fields, outputs, strict=True)
        }
        samples.append((inputs, nnzs))
    output_sparsities = {
        name: choose_seed(rng, idxs, lg_dims).sum()
        for name, idxs in output_fields.items()
    }

    # Select the sample with the closest output sparsity, in log. An empty
    # output counts as one nonzero.
    def distance(sample):
        _, nnzs = sample
        return sum(
            abs(
                np.log2(max(nnz, 1) / output_sparsities[name])
                - sum(lg_dims[idx.name] for idx in output_fields[name])
            )
            for name, nnz in nnzs.items()
        )

    return min(samples, key=distance)[0]


def coo(tns: np.ndarray) -> FiberTensor:
    """A Finch COO tensor of the nonzeros of tns."""
    fmt = element(tns.dtype.type(0), ftype(tns.dtype), ftypes.intp)
    if not tns.ndim:
        return FiberTensor(ElementLevel(fmt, NumpyBuffer(tns.reshape(1))))
    indices = np.nonzero(tns)
    ptr = NumpyBuffer(np.array([0, indices[0].size], dtype=np.intp))
    return FiberTensor(
        SparseCOOLevel(
            ElementLevel(fmt, NumpyBuffer(tns[indices])),
            tuple(np.intp(n) for n in tns.shape),
            ptr,
            tuple(NumpyBuffer(np.ascontiguousarray(idx)) for idx in indices),
        )
    )


def examples(directory: Path, chunk=None) -> list[str]:
    """The names of the examples in directory, or of those of one chunk."""
    prefix = "*" if chunk is None else f"{chunk:04d}-*"
    return sorted(
        path.name.removesuffix(".program.json")
        for path in directory.glob(f"{prefix}.program.json")
    )


def load(
    directory: Path, example
) -> tuple[Plan, dict[Alias, list[LevelOption]], dict[Alias, Tensor]]:
    """The plan, formats, and inputs of an example which generate wrote."""
    record = json.loads((directory / f"{example}.program.json").read_text())
    plan = parse_logic(record["program"])
    assert isinstance(plan, Plan)
    formats: dict[Alias, list[LevelOption]] = {
        HardAlias(name): [OPTIONS[kind] for kind in kinds]
        for name, kinds in record["formats"].items()
    }
    bindings: dict[Alias, Tensor] = {
        tns: from_binsparse(
            bsp.load_binsparse(directory / f"{example}.{tns.name}.bsp.npz")
        )
        for tns in (A, B)
    }
    return plan, formats, bindings


def generate(directory: Path, examples, chunk, lg_max_space, lg_max_dim, rng) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for example in range(examples):
        name = f"{chunk:04d}-{example:05d}"
        prgm, input_fields, output_fields, lg_dims = program(
            rng, lg_max_space, lg_max_dim
        )
        tensors = inputs(rng, prgm, input_fields, output_fields, lg_dims)
        query, produces = prgm.bodies
        loop_order = shuffled(rng, tuple(Field(idx) for idx in lg_dims))
        plan = Plan((fuse_loop_order(query, loop_order), produces))
        formats = {
            tns.name: [choose(rng, list(OPTIONS)) for _ in idxs]
            for tns, idxs in output_fields.items()
        }
        record = {"program": str(plan), "formats": formats}
        (directory / f"{name}.program.json").write_text(json.dumps(record, indent=2))
        for tns, array in tensors.items():
            path = directory / f"{name}.{tns.name}.bsp.npz"
            bsp.save_binsparse(to_binsparse(coo(array)), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument(
        "--chunk", type=int, default=0, help="the chunk which names the examples"
    )
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
    # Chunks drawn with one seed still draw different programs.
    rng = np.random.default_rng(None if args.seed is None else (args.seed, args.chunk))
    generate(
        args.directory,
        args.examples,
        args.chunk,
        args.lg_max_space,
        args.lg_max_dim,
        rng,
    )


if __name__ == "__main__":
    main()
