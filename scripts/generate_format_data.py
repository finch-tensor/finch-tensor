"""
Draw programs for fitting the formatter's cost model, and their inputs. Each
program reduces a map of inputs A and B into a workspace w and copies w into
C. Their fields cover each region of the Venn diagram of the fields of A, B,
and C which lies in A or B, since nothing would give a field of C alone an
extent. The layouts of all tensors follow one drawn order of the fields, and
each query has its own drawn loop order. The formats of w and C, and the
extents, densities, and dtype of the inputs are drawn too. Each program is
written to DIR as JSON, with the text of its logic plan, the formats of w and
C, and what was drawn, and its inputs as binsparse, under the name of its chunk
and its index in the chunk, so jobs of different chunks can share DIR.

    pixi run -e fitting python -m scripts.generate_format_data \
        DIR [--examples N] [--chunk K] [--lg-max-space N] [--lg-max-dim N] \
        [--seed S]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import binsparse as bsp

from finch import ffuncs
from finch.algebra import Tensor
from finch.autoschedule.cost_models import LevelOption
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS
from finch.autoschedule.util import flatten_plans, fuse_loop_order
from finch.fileio import from_binsparse
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
    parse_logic,
)

OPS = {"add": ffuncs.add, "mul": ffuncs.mul}
OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}
A, B, C, w = HardAlias("A"), HardAlias("B"), HardAlias("C"), HardAlias("w")


def with_subsequence(order, fields):
    """The fields of order which are in fields, in the order of order."""
    return tuple(idx for idx in order if idx in fields)



def interaction_range(s_x, s_y):
    return abs(s_x + s_y) - 1, 1 - abs(s_x - s_y)


def choose(rng, options):
    return options[rng.integers(len(options))]


def shuffled(rng, items):
    return tuple(items[p] for p in rng.permutation(len(items)))

def seed(skew, sparsity):
    S = np.outer([np.array([0.5 - s/2, 0.5 + s/2]) for s in skew])
    S = seed * sparsity

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
    """
        What is drawn for a random program, as JSON, which build makes a plan of.
        a random subprogram of C_ilmq += A_jlpq * B_kmpq
    """
    while True:
        idxs = [idx in "ijklmnp" if rng.random() < 0.5 ]
        lg_dims = {idx: int(rng.integers(1, lg_max_dim + 1)) for idx in idxs}
        if sum(lg_dims.values()) <= lg_max_space:
            break
    order = shuffled(rng, idxs)
    A_idxs = [Field(idx) for idx in with_subsequence("jlpq", order)]
    if rng.random() < 0.2:
        A_idxs = shuffled(rng, idxs)
    B_idxs = [Field(idx) for idx in with_subsequence("kmpq", order)]
    if rng.random() < 0.2:
        B_idxs = shuffled(rng, idxs)
    C_idxs = [Field(idx) for idx in with_subsequence("ilmq", order)]
    if rng.random() < 0.2:
        C_idxs = shuffled(rng, idxs)
    r_idxs = [Field(idx) for idx in with_subsequence("jkp", order)]
    op = rng.choice([ffuncs.add, ffuncs.mul])
    prgm = Plan(
        Query(
            Table(HardAlias("C"), C_idxs),
            Aggregate(
                Literal(op),
                Literal(0.0),
                MapJoin(
                    Table(HardAlias("A"), A_idxs),
                    Table(HardAlias("B"), B_idxs),
                ),
                r_idxs
            )
        ),
        Produces((C,))
    )
    input_fields = {
        HardAlias("A"):A_idxs,
        HardAlias("B"):B_idxs,
    }
    output_fields = {
        HardAlias("C"):C_idxs,
    }
    return prgm, input_fields, output_fields, lg_dims

def choose_seed(idxs, lg_dims):
    skews = [rng.choice([-1, 1])*rng.random() for _ in idxs]
    sparsity = 2^(-1 * rng.random(sum(lg_dims[idxs])))
    return seed(skew, sparsity)

def inputs(rng, prgm, input_fields, output_fields, lg_dims):
    samples = []
    for i in range(100):
        seeds = {name: choose_seed(idxs) for name, idxs in input_fields}
        inputs = dict(map(rmat, seeds))
        outputs = DEFAULT_SCHEDULER(prgm, inputs)
        push!(samples, {name:output.count_nonfill_values() for name:output in outputs})
    output_sparsities = {name:sum(choose_seed(idxs, lg_dims)) for name, idxs in output_fields}
    #select sample with closest output sparsity.
    

        



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
        prgm = program(rng, lg_max_space, lg_max_dim)
        plan = build(prgm)
        record = {"program": str(plan), "formats": prgm["formats"], "drawn": prgm}
        (directory / f"{name}.program.json").write_text(json.dumps(record, indent=2))
        for tns, tensor in inputs(rng, prgm).items():
            bsp.save_binsparse(tensor, directory / f"{name}.{tns}.bsp.npz")


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
