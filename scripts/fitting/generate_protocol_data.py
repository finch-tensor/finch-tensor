"""
Draw programs for fitting the cost model of the protocol stage of
physical_bases, and their inputs. Each program is a random
subprogram of C_lmq += A_jlpq * B_kmpq, which loops in the order its inputs
store their fields, and its inputs are Kronecker products of R-MAT blocks of
drawn densities, as kronecker_stats draws them. Its variants are every option
of the decision of the stage, or VARIANTS of them drawn without replacement.
Of SAMPLES draws of the inputs of a program, KEEP of the stage are kept,
spread over the densities of A, B, and C, and each makes an example with the
program.
Each example is written to DIR/data as JSON, with the text of its query, its
loop order, its variants, and the log shape, the R-MAT seed of each block, and
the RNG seed of each of its inputs, with the version of numpy which drew its
blocks and their checksum, under the name of its chunk and its index in the
chunk, so jobs of different chunks can share DIR. No tensors are stored:
record_inputs draws the blocks of the inputs from their seeds again, checking
that they match, and materialize expands them into COO tensors.

    pixi run -e fitting python -m scripts.fitting.generate_protocol_data DIR \\
        [--stage STAGE] [--programs N] [--chunk K] [--seed S]
"""

from typing import Any

import numpy as np

from finch import ffuncs
from finch.autoschedule.cost_models import allowed_options, read_options
from finch.autoschedule.formatter.smart_formatter import LEVEL_OPTIONS
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
)
from scripts.fitting.generation import (
    BLOCK_BITS,
    choose,
    input_record,
    main,
    seed_lg_densities,
    shuffled,
    skewed_fractions,
    spread,
    variants,
)
from scripts.fitting.physical_bases import (
    OPTIONS,
    C,
    fused_plan,
    with_subsequence,
)

STAGES = ("protocol",)
# The most fields of an input, which bounds the variants of the protocol stage.
MAX_RANK = 3
# log2 of the most entries all the fields of a program may address, and of the
# most values of any one field.
LG_MAX_SPACE = 32
LG_MAX_DIM = 16
# The chance that a program has each field.
FIELD_PROBABILITY = 0.5
# The sizes of programs and the densities of their inputs are fractions of the
# most they may be, distributed as the largest of SKEW uniform draws.
SKEW = 5
# The draws of the seeds of the inputs of each program, of which KEEP of its
# stage are kept as examples. In the protocol stage, the variants of the
# examples of a program compile to the same kernels, so they are compiled
# once for them all, and more are kept.
SAMPLES = 1000
KEEP = {"protocol": 10}
# The most variants of a program, where a stage has too many to time: the
# protocol stage has up to 6561.
VARIANTS = {"protocol": 64}


def program(rng):
    """
    A random subprogram of C_ilmq += A_jlpq * B_kmpq, and its loop order, which
    is the order its inputs store their fields in. C has no fields of its own,
    since nothing would give them extents. The log sizes of its fields add up
    to skewed_fractions of the most they may, under LG_MAX_SPACE and
    LG_MAX_DIM. A program whose A or B has more than MAX_RANK fields is drawn
    again.
    """
    while True:
        idxs = [idx for idx in "jklmpq" if rng.random() < FIELD_PROBABILITY]
        if all(len(set(idxs) & set(tns)) <= MAX_RANK for tns in ("jlpq", "kmpq")):
            break
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
    S = np.ones(())
    for _ in idxs:
        s = rng.choice([-1, 1]) * rng.random()
        S = np.multiply.outer(S, [0.5 - s / 2, 0.5 + s / 2])
    lg_size = sum(lg_dims[idx.name] for idx in idxs)
    return S * 2 ** -((1 - skewed_fractions(rng, SKEW)) * lg_size)


def output_kinds(query: Query, loop_order: tuple[Field, ...]) -> list[list[str]]:
    plan = fused_plan(query, loop_order)
    return [
        [opt.kind for opt in allowed_options(C, depth, plan, LEVEL_OPTIONS)]
        for depth in range(len(query.lhs.idxs))
    ]


def read_kinds(idxs, loop_order) -> list[list[tuple[str, str]]]:
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
    raise ValueError(f"Unknown stage {stage}, expected one of {STAGES}")


def program_inputs(rng, stage):
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
    spaces, make = variant_space(stage, query, input_fields, loop_order, rng)
    options, exhaustive = variants(spaces, make, rng, VARIANTS.get(stage))
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
        for i in spread(rng, points, KEEP[stage], SKEW)
    ]


if __name__ == "__main__":
    main(__doc__, STAGES, program_inputs)
