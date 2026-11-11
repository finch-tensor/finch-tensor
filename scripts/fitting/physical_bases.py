"""
The stages of fitting the cost models of the physical optimizers, from the
innermost decision out. Each stage enumerates every option of its own decision
for random programs, and the stages below it, which are already fitted, make
theirs, so the runtime of each option is what the optimizer would get by
choosing it, and the best option of each example is known. Each stage names
the basis its model is fitted to, which fit_physical_data extracts from the
examples coalesce_physical_data collects, so a new basis needs no new timings.

- protocol: for a random program, every assignment of the formats and
  protocols of one of its tensors, with the protocols of the loops over its
  fields, under random formats and protocols of the rest, as the formatter
  re-chooses the protocols of the loops a format changes. Its model, the
  protocol cost model, is the one the formatter, the workspacer, and the
  protocol optimizer price with.
- loop: for a random program, every loop order, with its inputs stored in
  that order, so that the program is concordant, and the output stored as the
  program stores it, which the loops may scatter into. Each order is passed
  to the workspacer, which stages the writes which scatter in a workspace,
  and copies inputs into new formats, where that costs less, then to the
  formatter, which chooses the formats from the supports of its inputs, and
  then to the protocol optimizer. Its features are those of the program as
  ordered, before any of them run, from the stats of its inputs alone,
  whatever their formats, with every table galloping, as the loop orderer
  prices it, so the model learns what a loop order costs once they have run.
- factorize: for a random sum of products, every factorization the factorizer
  can make of it, by the order it eliminates the fields it sums over. Each
  query of a factorization which computes something is timed on its own, as
  an example of the loop stage, in the loop order the loop orderer chooses
  for it with the loop model, with every tensor it reads copied into that
  order, and the results of earlier queries drawn from their exact supports,
  so no copy is timed, as the factorizer prices queries whatever their order.
  A variant's runtime is the sum of those of its queries, and its features
  are those of its queries, from the stats of its inputs alone, as the
  factorizer prices them.
"""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from finch.algebra import StaticFillFType, TensorFType, ftype, ftypes
from finch.autoschedule import optimize_protocols, protocolize
from finch.autoschedule.cost_models import (
    LOOP_COST_MODEL,
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    Affine,
    Basis,
    Concat,
    CostModel,
    ElementLevelOption,
    FlopsBasis,
    LevelOption,
    PigeonBasis,
    WriteBasis,
    WriteClasses,
    WriteStrata,
    level_options,
    loop_basis,
    loop_cost_model,
)
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    format_ftype,
    optimize_formats,
)
from finch.autoschedule.loop_orderer import BFSLoopOrderer
from finch.autoschedule.loop_orderer.loop_ordering import (
    drop_internal_reorders,
    galloping,
)
from finch.autoschedule.tensor_stats.memo_stats import MemoStats, MemoStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.autoschedule.workspacer import (
    add_cost_effective_reformats,
    add_cost_effective_workspaces,
)
from finch.finch_logic import (
    Alias,
    Field,
    Fuse,
    HardAlias,
    LogicStatement,
    Plan,
    Produces,
    Query,
    StatsFactory,
    Table,
    TensorStats,
    parse_logic,
)
from finch.finch_logic.parser import PROTOCOLS
from finch.finch_logic.simplification import simplify_logic
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite
from scripts.fitting.kronecker_stats import (
    KroneckerStatsFactory,
    KroneckerTensor,
    support_tensor,
)

STAGES = ("protocol", "loop", "factorize")
OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}
A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")


BASES: dict[str, Basis] = {
    "protocol": PROTOCOL_BASIS,
    "loop": loop_basis(),
    "factorize": Affine(FlopsBasis()),
}


# The rows of the loop basis before its writes are classed, from which
# stratified gives the rows of the loop basis under any cutoff without
# featurizing again.
TUNING_BASIS = Affine(Concat(PigeonBasis(), WriteBasis()))


def stratified(rows: np.ndarray, classes: WriteClasses) -> np.ndarray:
    """The rows of loop_basis(classes) of rows of TUNING_BASIS."""
    start = PigeonBasis().n_features
    end = start + WriteBasis().n_features
    writes = WriteStrata(classes)
    return np.hstack(
        [rows[:, :start], writes.transform(rows[:, start:end]), rows[:, end:]]
    )


def parse_models(args: list[str] | None) -> dict[str, CostModel]:
    """
    The models of NAME=PATH arguments, which fit_physical_data wrote, over the
    models the optimizers use by default. A loop model whose
    region cutoff was tuned classes its writes by it.
    """
    models = {"protocol": PROTOCOL_COST_MODEL, "loop": LOOP_COST_MODEL}
    for arg in args or ():
        name, _, path = arg.partition("=")
        if name not in BASES or not path:
            raise ValueError(
                f"Expected NAME=PATH with NAME of {list(BASES)}, got {arg}"
            )
        data = json.loads(Path(path).read_text())
        if name == "loop":
            models[name] = loop_cost_model(data)
        else:
            basis = BASES[name]
            untrained = CostModel(basis, np.zeros(basis.n_features))
            models[name] = untrained.from_json(data)
    return models


def with_subsequence(order, fields):
    return tuple(idx for idx in order if idx in fields)


def fused_plan(query: Query, loop_order: tuple[Field, ...]) -> Plan:
    return Plan((fuse_loop_order(query, loop_order), Produces((query.lhs.tns,))))


def fused_order(stmt: LogicStatement) -> tuple[Field, ...]:
    """The loop order of the Fuse nodes around a statement."""
    match stmt:
        case Fuse(idx, body):
            return (idx, *fused_order(body))
    return ()


def variant_plan(record: Mapping[str, Any], variant: Mapping[str, Any]) -> Plan:
    """
    The plan of a variant of the program of record, under the loop order of
    the variant or of the record, reading its inputs with the protocols of the
    variant, if any. In the loop stage the inputs are stored in the loop
    order.
    """
    prgm = parse_logic(record["query"])
    assert isinstance(prgm, Plan)
    query, _ = prgm.bodies
    assert isinstance(query, Query)
    order = tuple(
        Field(name) for name in variant.get("loop_order", record["loop_order"])
    )
    protocols = variant.get("protocols", {})
    stage = record["stage"]

    def read(node):
        match node:
            case Table(HardAlias(name) as tns, idxs):
                if stage == "loop":
                    idxs = with_subsequence(order, idxs)
                if name in protocols:
                    protos = tuple(PROTOCOLS[proto] for proto in protocols[name])
                    return Table(tns, idxs, protos)
                return Table(tns, idxs)
        return None

    query = Query(query.lhs, Rewrite(PostWalk(read))(query.rhs))
    return fused_plan(query, order)


def galloped(plan: Plan) -> Plan:
    """plan, with every query reading its tables as the loop orderer prices
    them, galloping."""

    def query(node):
        match node:
            case Query(lhs, rhs):
                return Query(lhs, galloping(rhs))
        return None

    result = Rewrite(PostWalk(query))(plan)
    assert isinstance(result, Plan)
    return result


def input_perms(
    record: Mapping[str, Any], plan: LogicStatement
) -> dict[Alias, tuple[int, ...]]:
    """
    The permutation of the fields of each input of the example of record which
    stores it in the order plan reads it, from the order the example stores
    it in.
    """
    stored = input_fields(parse_logic(record["query"]))
    return {
        tns: tuple(stored[tns].index(idx) for idx in idxs)
        for tns, idxs in input_fields(plan).items()
        if tns in stored
    }


def input_fields(plan: LogicStatement) -> dict[Alias, tuple]:
    fields = {}
    for node in PostOrderDFS(plan):
        match node:
            case Table(Alias() as tns, idxs):
                fields.setdefault(tns, idxs)
    return fields


def input_ftype(options: list[LevelOption], tensor: KroneckerTensor):
    shape_type = (ftype(tensor.idx_type),) * tensor.ndim
    return format_ftype(options, StaticFillFType(tensor.dtype(0)), shape_type)


def read_inputs(
    record: Mapping[str, Any],
    plan: LogicStatement,
    inputs: Mapping[Alias, KroneckerTensor],
) -> dict[Alias, KroneckerTensor]:
    """
    The inputs of the example of record which plan reads, each storing its
    fields in the order plan reads them.
    """
    perms = input_perms(record, plan)
    return {
        tns: tensor.transpose(perms[tns])
        for tns, tensor in inputs.items()
        if tns in perms
    }


def record_stats_factory(record: Mapping[str, Any]) -> StatsFactory:
    """
    The exact stats of the inputs of the example of record, memoized, since
    the optimizers rebuild the same stats for every option they price.
    """
    return MemoStatsFactory(KroneckerStatsFactory(record["block_bits"]))


def input_stats(
    plan: LogicStatement,
    inputs: Mapping[Alias, KroneckerTensor],
    stats_factory: StatsFactory,
) -> dict[Alias, TensorStats]:
    fields = input_fields(plan)
    return {tns: stats_factory(tensor, fields[tns]) for tns, tensor in inputs.items()}


def choose_formats(
    plan: LogicStatement,
    inputs: Mapping[Alias, KroneckerTensor],
    stats_factory: StatsFactory,
    cost_model: CostModel,
    bindings: Mapping[Alias, TensorFType] | None = None,
) -> dict[Alias, list[LevelOption]]:
    """
    The level options of each tensor of plan without a format in bindings
    under cost_model, as the formatter would choose them from the supports of
    inputs, which store their fields in the order plan reads them. Without
    bindings, the inputs are formatted too. A tensor which plan fuses stores
    the fields its first definition doesn't fuse.
    """
    bindings = bindings or {}
    stats = input_stats(plan, inputs, stats_factory)
    fill_types = plan.infer_fill_type(
        {tns: StaticFillFType(tensor.dtype(0)) for tns, tensor in inputs.items()}
    )
    stored = {}
    for tns, idxs in input_fields(plan).items():
        if tns.unfused not in bindings:
            stored.setdefault(tns.unfused, (ftypes.intp,) * (len(idxs) - tns.nfused))
    ftypes_2, _ = optimize_formats(
        plan,
        bindings,
        stats,
        stats_factory,
        stored,
        fill_types,
        cost_model,
        LEVEL_OPTIONS,
    )
    formats = {}
    for tns, fmt in ftypes_2.items():
        options = level_options(fmt)
        assert options is not None
        formats[tns] = options
    return formats


def resolve(
    record: Mapping[str, Any],
    variant: Mapping[str, Any],
    inputs: Mapping[Alias, KroneckerTensor],
    models: Mapping[str, CostModel],
) -> tuple[LogicStatement, dict[Alias, list[LevelOption]]]:
    """
    The program and formats of a variant of the example of record, once the
    stages below its stage have made their choices from the exact stats of
    inputs: the formats of every tensor, unless the variant gives them, and
    the protocols each input is read with, unless the variant gives those too,
    as the variants of the protocol stage do. Where the formats are chosen, the
    inputs are formatted first, as they are when the program runs, then the
    workspacer stages the writes which scatter in workspaces, and copies
    inputs into new formats, where that costs less, and then the formatter
    formats the outputs and copies of the program it leaves. The inputs the
    program reads are stored in the order it reads them, which is the loop
    order in the loop stage.
    """
    plan = variant_plan(record, variant)
    formats = {
        HardAlias(name): [OPTIONS[kind] for kind in kinds]
        for name, kinds in variant.get("formats", {}).items()
    }
    inputs = read_inputs(record, plan, inputs)
    stats_factory = record_stats_factory(record)
    if "formats" not in variant:
        formats = choose_formats(plan, inputs, stats_factory, models["protocol"])
        bindings = {
            tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()
        }
        for staging in (add_cost_effective_workspaces, add_cost_effective_reformats):
            plan = staging(
                plan,
                bindings,
                input_stats(plan, inputs, stats_factory),
                stats_factory,
                models["protocol"],
            )
        formats = {
            **{tns: formats[tns] for tns in inputs},
            **choose_formats(plan, inputs, stats_factory, models["protocol"], bindings),
        }
    if "protocols" not in variant:
        plan = optimize_protocols(
            plan,
            {tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()},
            input_stats(plan, inputs, stats_factory),
            stats_factory,
            models["protocol"],
        )
    return plan, formats


def factorized_plan(variant: Mapping[str, Any]) -> Plan:
    """
    The plan of a variant of the factorize stage, as the loop orderer receives
    it, without its copies, which happen outside the timer.
    """
    plan = drop_internal_reorders(simplify_logic(parse_logic(variant["plan"])))
    assert isinstance(plan, Plan)

    def computes(body: LogicStatement) -> bool:
        match body:
            case Query(_, Table()):
                return False
        return True

    return Plan(tuple(filter(computes, plan.bodies)))


def unmemoized(stats: TensorStats) -> TensorStats:
    match stats:
        case MemoStats():
            return stats.inner
    return stats


def factorized_queries(
    record: Mapping[str, Any],
    variant: Mapping[str, Any],
    inputs: Mapping[Alias, KroneckerTensor],
    models: Mapping[str, CostModel],
) -> Iterator[tuple[dict[str, Any], dict[Alias, KroneckerTensor]]]:
    """
    Each query of the factorized_plan of a variant of the example of record,
    as the record of an example of the loop stage of that query alone, in the
    loop order the loop orderer chooses for it with models["loop"], and the
    tensors it reads, under their names in the query: the inputs of record,
    and the results of earlier queries, as KroneckerTensors of their exact
    supports, which are drawn when they are reached. Each query stores its
    result as the factorization does.
    """
    plan = factorized_plan(variant)
    stats_factory = record_stats_factory(record)
    stats = input_stats(plan, inputs, stats_factory)
    orderer = BFSLoopOrderer(cost_model=models["loop"])
    ordered = orderer.set_loop_orders(plan, dict(stats), stats_factory)
    stored = input_fields(parse_logic(record["query"]))
    machine = StatsMachine(stats_factory, dict(stats))
    dtype = np.dtype(record["dtype"]).type
    idx_type = np.dtype(record["idx_type"]).type
    for query, fused in zip(plan.bodies[:-1], ordered.bodies[:-1], strict=True):
        assert isinstance(query, Query)
        tensors = {}
        for tns, idxs in input_fields(query.rhs).items():
            if tns in inputs:
                # The factorizer keeps the fields of the inputs as they are.
                if idxs != stored[tns]:
                    raise ValueError(f"Expected {tns} read as {stored[tns]}")
                tensors[tns] = inputs[tns]
            else:
                support = unmemoized(machine(Table(tns, idxs)))
                tensors[tns] = support_tensor(support, dtype, idx_type)
        machine(query)
        yield (
            {
                "stage": "loop",
                "query": str(Plan((query, Produces((query.lhs.tns,))))),
                "loop_order": [idx.name for idx in fused_order(fused)],
                "dtype": record["dtype"],
                "idx_type": record["idx_type"],
                "block_bits": record["block_bits"],
            },
            tensors,
        )


def format_kinds(formats: Mapping[Alias, list[LevelOption]]) -> dict[str, list[str]]:
    return {
        tns.name: [option.kind for option in options]
        for tns, options in formats.items()
    }


def features(
    basis: Basis, plan, formats, inputs, stats_factory, protocols=True
) -> np.ndarray:
    """
    The features of basis of plan, with the level options of formats, from the
    stats of inputs, which store their fields in the order plan reads them.
    Unless protocols is false, each read without a protocol takes the default
    of its level, as Finch.jl reads it.
    """
    stats = input_stats(plan, inputs, stats_factory)
    if protocols:
        ftypes = {
            tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()
        }
        plan = protocolize(plan, ftypes)
    formats = {
        alias: [*options, ElementLevelOption()] for alias, options in formats.items()
    }
    return basis.get_features(plan, formats, stats, stats_factory)


def featurize_record(
    record: Mapping[str, Any],
    inputs: Mapping[Alias, KroneckerTensor],
    name: str,
    models: Mapping[str, CostModel],
    basis: Basis | None = None,
) -> list[np.ndarray]:
    """
    The features of basis, by default the basis name, of each variant of the
    example of record,
    whose inputs are inputs, as it was timed: the stages below make their
    choices with models, as time_physical_data makes them, so nothing is
    compiled or run. The loop stage orders loops before the workspacer and the
    formatter run, so its features are those of the program as ordered, from
    the stats of its inputs alone, whatever their formats, as the loop orderer
    prices it, galloping, and it chooses nothing. The factorize stage factors
    before any of them, so its features are those of the queries of the
    factorization, from the stats of its inputs alone, as the factorizer
    prices them.
    """
    stats_factory = record_stats_factory(record)
    stage = record["stage"]
    rows = []
    for variant in record["variants"]:
        match stage:
            case "loop":
                plan, formats = galloped(variant_plan(record, variant)), {}
            case "factorize":
                plan, formats = factorized_plan(variant), {}
            case _:
                plan, formats = resolve(record, variant, inputs, models)
        rows.append(
            features(
                BASES[name] if basis is None else basis,
                plan,
                formats,
                read_inputs(record, plan, inputs),
                stats_factory,
                protocols=stage not in ("loop", "factorize"),
            )
        )
    return rows
