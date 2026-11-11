"""
Draw programs for fitting the cost model of the factorize stage of
physical_bases, and their inputs. Each program sums the product of two to
four inputs, the first of A, B, C and D, into E over the fields E lacks. Each
input has each of the fields of FIELDS with chance FIELD_PROBABILITY, and E
has each field of the inputs with chance OUTPUT_PROBABILITY, so that the
inputs are connected by the fields they share, and the program sums over at
least two of them. The inputs are Kronecker products of R-MAT blocks of drawn
densities, as kronecker_stats draws them. Its variants are the distinct
factorizations the factorizer makes of it, one for each order of the fields
it sums over, or VARIANTS of them drawn without replacement, leaving out those
with an intermediate or output of more than 2 ** LG_MAX_NNZ nonzeros, which
bounds the memory of each query, and those estimated to run for more than
MAX_SECONDS. Of SAMPLES draws of the inputs, spread over the densities of the
inputs and E, up to RETRIES are tried in turn, and the first with two
variants whose mean is estimated to run for MIN_MEAN_SECONDS to
MAX_MEAN_SECONDS, so that its queries do much more work than a call costs,
makes the example of the program. A program none of whose tries does is
drawn again.
Each example is written to DIR/data as JSON, with the text of its query, its
variants, as the text of their plans, and the log shape, the R-MAT seed of
each block, and the RNG seed of each of its inputs, with the version of numpy
which drew its blocks and their checksum, under the name of its chunk and its
index in the chunk, so jobs of different chunks can share DIR. No tensors are
stored: record_inputs draws the blocks of the inputs from their seeds again,
checking that they match, and materialize expands them into COO tensors.

    pixi run -e fitting python -m scripts.fitting.generate_factorize_data DIR \\
        [--programs N] [--chunk K] [--seed S]
"""

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np

from finch import ffuncs
from finch.autoschedule.cost_models import FlopsBasis
from finch.autoschedule.factorizer.galley_factorizer.annotated_query import (
    AnnotatedQuery,
)
from finch.autoschedule.factorizer.galley_factorizer.query_normalization import (
    postprocess_plan_after_galley,
    preprocess_plan_for_galley,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import flatten_plans
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
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import PostWalk, Rewrite
from scripts.fitting.generation import (
    BLOCK_BITS,
    choose,
    input_record,
    main,
    record_inputs,
    seed_lg_densities,
    skewed_fractions,
    spread,
    variants,
)
from scripts.fitting.physical_bases import (
    factorized_plan,
    input_stats,
    record_stats_factory,
)

STAGES = ("factorize",)
INPUTS = tuple(HardAlias(name) for name in "ABCD")
E = HardAlias("E")
FIELDS = "ijklmn"
# The fewest and most inputs of a program, of equal chance, and the most
# fields of an input.
MIN_INPUTS = 2
MAX_INPUTS = 4
MAX_RANK = 3
# log2 of the most entries all the fields of a program may address, of the
# most values of any one field, and of the most nonzeros of any input,
# intermediate, or output, which bounds the memory of each query.
LG_MAX_SPACE = 44
LG_MAX_DIM = 22
LG_MAX_NNZ = 27
# The chance that an input has each field, and that the output has each field
# of the inputs.
FIELD_PROBABILITY = 0.5
OUTPUT_PROBABILITY = 0.3
# The sizes of programs and the densities of their inputs are fractions of the
# most they may be, distributed as the largest of SKEW uniform draws.
SKEW = 8
# The draws of the seeds of the inputs of each program, of which up to RETRIES
# are tried, and one is kept as its example. The queries of each
# factorization choose loop orders and protocols from the stats of its
# example, so their kernels can differ, and one is kept, for more programs
# instead.
SAMPLES = 200
RETRIES = 32
# The most factorizations of a program, of up to 720.
VARIANTS = 16
# The seconds the variants of an example are estimated to run for: the most
# of any, and the fewest and most of their mean. The seconds of an iteration
# and of an output are the median rates of the variants of the factorize data
# of 2026-11-08 which ran for more than 0.1 s, about which they ran for half
# to twice as long.
MAX_SECONDS = 60.0
MIN_MEAN_SECONDS = 1.0
MAX_MEAN_SECONDS = 10.0
SECONDS_PER_ITERATION = 1.4e-9
SECONDS_PER_OUTPUT = 6.5e-8


def connected(field_sets: list[set[str]]) -> bool:
    """Whether the inputs of field_sets are connected by the fields they share."""
    reached, rest = set(field_sets[0]), field_sets[1:]
    while rest:
        joined = [fields for fields in rest if reached & fields]
        if not joined:
            return False
        for fields in joined:
            reached |= fields
            rest.remove(fields)
    return True


def program(rng, inputs):
    """
    A random sum of the product of inputs into E, connected by the fields
    they share, which sums over at least two fields, and the log size of each
    field. The log sizes of its fields add up to skewed_fractions of the most
    they may, under LG_MAX_SPACE and LG_MAX_DIM.
    """
    while True:
        input_idxs = {
            tns: [idx for idx in FIELDS if rng.random() < FIELD_PROBABILITY]
            for tns in inputs
        }
        if not all(1 <= len(idxs) <= MAX_RANK for idxs in input_idxs.values()):
            continue
        used = [idx for idx in FIELDS if any(idx in i for i in input_idxs.values())]
        out = [idx for idx in used if rng.random() < OUTPUT_PROBABILITY]
        summed = [idx for idx in used if idx not in out]
        if len(summed) >= 2 and connected([set(i) for i in input_idxs.values()]):
            break
    lg_space = round(
        skewed_fractions(rng, SKEW) * min(LG_MAX_SPACE, LG_MAX_DIM * len(used))
    )
    # Each field has a bit, and the rest are dealt to the fields with room by
    # weights of a Dirichlet draw, so the fields of a program vary in size.
    lg_dims = dict.fromkeys(used, 1)
    weights = rng.dirichlet(np.ones(len(used)))
    for _ in range(lg_space - len(used)):
        room = weights * [lg_dims[idx] < LG_MAX_DIM for idx in used]
        lg_dims[used[rng.choice(len(used), p=room / room.sum())]] += 1
    input_fields = {
        tns: tuple(Field(idx) for idx in idxs) for tns, idxs in input_idxs.items()
    }
    prgm = Plan.from_children(
        Query(
            Table(E, tuple(Field(idx) for idx in out)),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0.0),
                MapJoin.from_children(
                    Literal(ffuncs.mul),
                    *(Table(tns, idxs) for tns, idxs in input_fields.items()),
                ),
                tuple(Field(idx) for idx in summed),
            ),
        ),
        Produces((E,)),
    )
    return prgm, input_fields, lg_dims


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


def named_in_order(plan: Plan) -> Plan:
    """plan, with the intermediates named T1, T2, and so on, in the order it
    defines them, so factorizations which differ only in their names match."""
    names: dict[Alias, Alias] = {}
    for body in plan.bodies:
        match body:
            case Query(Table(tns, _), _) if tns != E:
                names[tns] = HardAlias(f"T{len(names) + 1}")

    def rename(node):
        match node:
            case Alias() if node in names:
                return names[node]
        return None

    result = Rewrite(PostWalk(rename))(plan)
    assert isinstance(result, Plan)
    return result


def factorizations(
    prgm: Plan, stats: Mapping[Alias, TensorStats], stats_factory: StatsFactory
) -> list[Plan]:
    """
    The distinct factorizations of prgm which the factorizer makes, as
    GalleyLogicFactorizer makes them, one for each order of eliminating the
    fields it sums over.
    """
    query, produces = preprocess_plan_for_galley(prgm).bodies
    annotated = AnnotatedQuery(stats_factory, query, dict(stats))
    distinct: dict[str, Plan] = {}
    for order in itertools.permutations(annotated.reduce_idxs):
        aq = annotated.copy()
        # Eliminating a field eliminates those which only it covers too, which
        # are then skipped.
        queries = [
            aq.reduce_idx(idx) for idx in order if idx in aq.get_reducible_idxs()
        ]
        queries.append(aq.get_remaining_query())
        plan = postprocess_plan_after_galley(Plan((*queries, produces)))
        plan = named_in_order(flatten_plans(plan))
        distinct.setdefault(str(plan), plan)
    return list(distinct.values())


def largest_nonzeros(plan, stats, stats_factory) -> float:
    """The most nonzeros of any result of plan, counted exactly from stats."""
    machine = StatsMachine(stats_factory, dict(stats))
    return max(machine(body).estimate_non_fill_values() for body in plan.bodies[:-1])


def estimated_seconds(plan: Plan, stats, stats_factory) -> float:
    """The seconds the queries of plan which are timed run for, from their
    iterations and outputs, counted exactly from stats."""
    rows = FlopsBasis().get_features(
        factorized_plan({"plan": str(plan)}), {}, stats, stats_factory
    )
    return float(rows.sum(axis=0) @ [SECONDS_PER_ITERATION, SECONDS_PER_OUTPUT])


def program_inputs(rng, stage) -> list[dict[str, Any]]:
    """
    The record of the example of a drawn program of the first MIN_INPUTS to
    MAX_INPUTS of INPUTS, as draw_program_inputs draws it, drawn again until
    one of its tries makes one. The inputs are drawn first, so each number of
    them is as likely, however often the rest is drawn.
    """
    inputs = INPUTS[: rng.integers(MIN_INPUTS, MAX_INPUTS + 1)]
    while True:
        records = draw_program_inputs(rng, stage, inputs)
        if records is not None:
            return records


def draw_program_inputs(rng, stage, inputs) -> list[dict[str, Any]] | None:
    """
    The record of the example of a drawn program, with its query, its
    variants, and the seeds of its inputs, the first of RETRIES of SAMPLES
    draws, spread over the log densities of the inputs and the log expected
    products of E, with two variants under the caps whose mean is estimated
    to run for MIN_MEAN_SECONDS to MAX_MEAN_SECONDS, or None if none has.
    """
    prgm, input_fields, lg_dims = program(rng, inputs)
    dtype = choose(rng, ["float32", "float64"])
    idx_type = choose(rng, ["int32", "int64"])
    seeds = [
        {tns: choose_seed(rng, idxs, lg_dims) for tns, idxs in input_fields.items()}
        for _ in range(SAMPLES)
    ]
    points = [seed_lg_densities(prgm, input_fields, lg_dims, S) for S in seeds]
    for i in spread(rng, points, RETRIES, SKEW):
        record = {
            "stage": stage,
            "query": str(prgm),
            "dtype": dtype,
            "idx_type": idx_type,
            "block_bits": BLOCK_BITS,
            "inputs": {
                tns.name: input_record(
                    seeds[i][tns], [lg_dims[idx.name] for idx in idxs], rng
                )
                for tns, idxs in input_fields.items()
            },
        }
        stats_factory = record_stats_factory(record)
        stats = input_stats(prgm, record_inputs(record), stats_factory)
        plans = factorizations(prgm, stats, stats_factory)
        seconds = {
            str(plan): estimated_seconds(plan, stats, stats_factory)
            for plan in plans
            if largest_nonzeros(plan, stats, stats_factory) <= 2**LG_MAX_NNZ
        }
        kept = [plan for plan, time in seconds.items() if time <= MAX_SECONDS]
        if len(kept) < 2:
            continue
        mean = np.mean([seconds[plan] for plan in kept])
        if not MIN_MEAN_SECONDS <= mean <= MAX_MEAN_SECONDS:
            continue
        options, exhaustive = variants(
            [kept], lambda choices: {"plan": choices[0]}, rng, VARIANTS
        )
        return [
            {
                **record,
                "exhaustive": exhaustive and len(kept) == len(plans),
                "variants": options,
            }
        ]
    return None


if __name__ == "__main__":
    main(__doc__, STAGES, program_inputs)
