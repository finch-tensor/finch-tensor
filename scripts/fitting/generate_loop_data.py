"""
Draw programs for fitting the cost model of the loop stage of physical_bases,
and their inputs. Each program is a random subprogram of
C_lmq += A_jlpq * B_kmpq, whose inputs store their fields in the loop order it
was drawn with, and are Kronecker products of R-MAT blocks of drawn densities,
as kronecker_stats draws them. Its variants are its loop orders, or VARIANTS
of them drawn without replacement. Of SAMPLES draws of the inputs of a
program, KEEP are kept, spread over the densities of A, B, and C, and each
makes an example with the program. The programs are larger than those of
generate_protocol_data, so that most of their kernels do much more work than
a call costs, and the loop order decides much of what they cost.
Each example is written to DIR/data as JSON, with the text of its query, its
loop order, its variants, and the log shape, the R-MAT seed of each block, and
the RNG seed of each of its inputs, with the version of numpy which drew its
blocks and their checksum, under the name of its chunk and its index in the
chunk, so jobs of different chunks can share DIR. No tensors are stored:
record_inputs draws the blocks of the inputs from their seeds again, checking
that they match, and materialize expands them into COO tensors.

    pixi run -e fitting python -m scripts.fitting.generate_loop_data DIR \\
        [--programs N] [--chunk K] [--seed S]
"""

import itertools
from typing import Any

import numpy as np

from finch import ffuncs
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
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
from scripts.fitting.generation import (
    BLOCK_BITS,
    choose,
    input_record,
    main,
    record_inputs,
    seed_lg_densities,
    shuffled,
    skewed_fractions,
    spread,
    variants,
)
from scripts.fitting.physical_bases import (
    C,
    input_fields,
    record_stats_factory,
    with_subsequence,
)

STAGES = ("loop",)
# log2 of the most entries all the fields of a program may address, of the
# most values of any one field, and of the most nonzeros of any input or
# output, which bounds the memory of each step.
LG_MAX_SPACE = 34
LG_MAX_DIM = 18
LG_MAX_NNZ = 27
# The chance that a program has each field.
FIELD_PROBABILITY = 0.75
# The sizes of programs and the densities of their inputs are fractions of the
# most they may be, distributed as the largest of SKEW uniform draws.
SKEW = 8
# The draws of the seeds of the inputs of each program, of which KEEP are kept
# as examples. The loop orders of an example choose protocols from its stats,
# so their kernels can differ, and one is kept, for more programs instead.
SAMPLES = 1000
KEEP = 1
# The most loop orders of a program, of up to 720.
VARIANTS = 16


def program(rng):
    """
    A random subprogram of C_ilmq += A_jlpq * B_kmpq, and its loop order, which
    is the order its inputs store their fields in. C has no fields of its own,
    since nothing would give them extents. The log sizes of its fields add up
    to skewed_fractions of the most they may, under LG_MAX_SPACE and
    LG_MAX_DIM.
    """
    idxs = [idx for idx in "jklmpq" if rng.random() < FIELD_PROBABILITY]
    lg_space = round(
        skewed_fractions(rng, SKEW) * min(LG_MAX_SPACE, LG_MAX_DIM * len(idxs))
    )
    # Each field has a bit, and the rest are dealt to the fields with room by
    # weights of a Dirichlet draw, so the fields of a program vary in size.
    lg_dims = dict.fromkeys(idxs, 1)
    if idxs:
        weights = rng.dirichlet(np.ones(len(idxs)))
        for _ in range(lg_space - len(idxs)):
            room = weights * [lg_dims[idx] < LG_MAX_DIM for idx in idxs]
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
    loop_order = tuple(Field(idx) for idx in order)
    return prgm, input_fields, lg_dims, loop_order


def choose_seed(rng, idxs, lg_dims):
    """The seed of an input of idxs, skewed at random along each field, whose
    log nonzeros are skewed_fractions of its log size, at most LG_MAX_NNZ."""
    S = np.ones(())
    for _ in idxs:
        s = rng.choice([-1, 1]) * rng.random()
        S = np.multiply.outer(S, [0.5 - s / 2, 0.5 + s / 2])
    lg_size = sum(lg_dims[idx.name] for idx in idxs)
    lg_nnz = min(skewed_fractions(rng, SKEW) * lg_size, LG_MAX_NNZ)
    return S * 2.0 ** (lg_nnz - lg_size)


def variant_space(loop_order) -> tuple[list[list[Any]], Any]:
    """The loop orders of a program, which make up its variants, and a function
    of one of them which makes the variant."""
    orders = [
        [idx.name for idx in order] for order in itertools.permutations(loop_order)
    ]
    return [orders], lambda choices: {"loop_order": choices[0]}


def output_nonzeros(record) -> float:
    """The nonzeros of the output of the example of record, counted exactly
    from the blocks of its inputs."""
    prgm = parse_logic(record["query"])
    query, _ = prgm.bodies
    fields = input_fields(prgm)
    stats_factory = record_stats_factory(record)
    stats = {
        tns: stats_factory(tensor, fields[tns])
        for tns, tensor in record_inputs(record).items()
    }
    return StatsMachine(stats_factory, stats)(query.rhs).estimate_non_fill_values()


def program_inputs(rng, stage):
    """
    The records of the examples of a drawn program, as draw_program_inputs
    draws them, drawn again until no example has more than 2 ** LG_MAX_NNZ
    nonzeros in its output, which nothing else bounds.
    """
    while True:
        records = draw_program_inputs(rng, stage)
        if all(output_nonzeros(record) <= 2**LG_MAX_NNZ for record in records):
            return records


def draw_program_inputs(rng, stage):
    """
    The records of the examples of a drawn program, each with its query, its
    loop order, its variants, and the seeds of its inputs, which are KEEP of
    SAMPLES draws, spread over the log densities of A and B and the log
    expected products of C.
    """
    prgm, input_fields, lg_dims, loop_order = program(rng)
    dtype = choose(rng, ["float32", "float64"])
    seeds = [
        {tns: choose_seed(rng, idxs, lg_dims) for tns, idxs in input_fields.items()}
        for _ in range(SAMPLES)
    ]
    points = [seed_lg_densities(prgm, input_fields, lg_dims, S) for S in seeds]
    query, _ = prgm.bodies
    assert isinstance(query, Query)
    spaces, make = variant_space(loop_order)
    options, exhaustive = variants(spaces, make, rng, VARIANTS)
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
        for i in spread(rng, points, KEEP, SKEW)
    ]


if __name__ == "__main__":
    main(__doc__, STAGES, program_inputs)
