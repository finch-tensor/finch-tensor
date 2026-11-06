"""
The stages of fitting the cost models of the physical optimizers, from the
innermost decision out. Each stage enumerates every option of its own decision
for random programs, and the stages below it, which are already fitted, make
theirs, so the runtime of each option is what the optimizer would get by
choosing it, and the best option of each example is known. Each stage names
the basis its model is fitted to, which fit_physical_data extracts from the
examples coalesce_physical_data collects, so a new basis needs no new timings.

- protocol: for a random program with random formats, every assignment of
  the protocols its inputs are read with.
- format: for a random program, every assignment of the formats and protocols
  of one of its tensors, with the protocols of the loops over its fields,
  under random formats and protocols of the rest, as the formatter re-chooses
  the protocols of the loops a format changes. It fits the protocol basis
  from the same examples as the protocol stage, with formats too.
- transpose: for a random input, every order a copy of it can be stored in,
  which the copy reads out of order, as the loop orderer transposes an input
  it would read out of order. The loop orderer charges such copies with the
  transpose model, on top of the loop model.
- loop: for a random program, every loop order, with its inputs stored in
  that order, so that the program is concordant, and the output stored as the
  program stores it, which the loops may scatter into. Each order is passed
  to the workspacer, which stages the writes which scatter in a workspace
  where that costs less, then to the formatter, which chooses the formats
  from the supports of its inputs, and then to the protocol optimizer. Its
  features are those of the program as ordered, before any of them run, with
  its inputs read as their formats read them by default and its outputs
  unformatted, as the loop orderer prices it, so the model learns what a loop
  order costs once they have run.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from finch.algebra import StaticFillFType, TensorFType, ftype, ftypes
from finch.autoschedule import optimize_protocols, protocolize
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    Affine,
    Basis,
    Concat,
    CostModel,
    ElementLevelOption,
    LevelOption,
    PigeonBasis,
    TransposeBasis,
    WriteBasis,
    level_options,
)
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    format_ftype,
    optimize_formats,
)
from finch.autoschedule.util import fuse_loop_order
from finch.autoschedule.workspacer import add_workspaces
from finch.finch_logic import (
    Alias,
    Field,
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
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite
from scripts.fitting.kronecker_stats import KroneckerStatsFactory, KroneckerTensor

STAGES = ("protocol", "format", "transpose", "loop")
OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}
A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")
INPUTS = (A, B)

# The formatter and the protocol optimizer price the same basis, so the format
# stage fits it again, to examples whose formats vary.
BASES: dict[str, Basis] = {
    "protocol": PROTOCOL_BASIS,
    "format": PROTOCOL_BASIS,
    "transpose": Affine(TransposeBasis()),
    "loop": Affine(Concat(PigeonBasis(), WriteBasis())),
}


def parse_models(args: list[str] | None) -> dict[str, CostModel]:
    """
    The models of NAME=PATH arguments, which fit_physical_data wrote, over the
    models the optimizers use until they are fitted.
    """
    models = {"protocol": PROTOCOL_COST_MODEL, "format": PROTOCOL_COST_MODEL}
    for arg in args or ():
        name, _, path = arg.partition("=")
        if name not in BASES or not path:
            raise ValueError(
                f"Expected NAME=PATH with NAME of {list(BASES)}, got {arg}"
            )
        basis = BASES[name]
        untrained = CostModel(basis, np.zeros(basis.n_features))
        models[name] = untrained.from_json(json.loads(Path(path).read_text()))
    return models


def with_subsequence(order, fields):
    return tuple(idx for idx in order if idx in fields)


def fused_plan(query: Query, loop_order: tuple[Field, ...]) -> Plan:
    return Plan((fuse_loop_order(query, loop_order), Produces((C,))))


def variant_plan(record: Mapping[str, Any], variant: Mapping[str, Any]) -> Plan:
    """
    The plan of a variant of the program of record, under the loop order of
    the variant or of the record, reading its inputs with the protocols of the
    variant, if any. In the loop stage the inputs are stored in the loop
    order, and in the transpose stage the copy is.
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
            case Table(HardAlias(name) as tns, idxs) if tns in INPUTS:
                if stage == "loop":
                    idxs = with_subsequence(order, idxs)
                if name in protocols:
                    protos = tuple(PROTOCOLS[proto] for proto in protocols[name])
                    return Table(tns, idxs, protos)
                return Table(tns, idxs)
        return None

    query = Query(query.lhs, Rewrite(PostWalk(read))(query.rhs))
    if stage == "transpose":
        query = Query(Table(C, order), query.rhs)
    return fused_plan(query, order)


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
    ftypes_2 = optimize_formats(
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
    as the variants of the protocol and format stages do. Where the formats
    are chosen, the inputs are formatted first, as they are when the program
    runs, then the workspacer stages the writes which scatter in workspaces
    where that costs less, and then the formatter formats the outputs of the
    program it leaves. The inputs the program reads are stored in the order it
    reads them, which is the loop order in the loop stage.
    """
    plan = variant_plan(record, variant)
    formats = {
        HardAlias(name): [OPTIONS[kind] for kind in kinds]
        for name, kinds in variant.get("formats", {}).items()
    }
    inputs = read_inputs(record, plan, inputs)
    stats_factory = KroneckerStatsFactory(record["block_bits"])
    if "formats" not in variant:
        formats = choose_formats(plan, inputs, stats_factory, models["format"])
        bindings = {
            tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()
        }
        plan = add_workspaces(
            plan,
            bindings,
            input_stats(plan, inputs, stats_factory),
            stats_factory,
            models["format"],
        )
        formats = {
            **{tns: formats[tns] for tns in inputs},
            **choose_formats(plan, inputs, stats_factory, models["format"], bindings),
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
) -> list[np.ndarray]:
    """
    The features of the basis name of each variant of the example of record,
    whose inputs are inputs, as it was timed: the stages below make their
    choices with models, as time_physical_data makes them, so nothing is
    compiled or run. The loop stage orders loops before the workspacer and
    the formatter run, so its features are those of the program as ordered,
    reading its inputs as their formats read them by default, as the loop
    orderer prices them, and its outputs without protocols.
    """
    stage = record["stage"]
    stats_factory = KroneckerStatsFactory(record["block_bits"])
    rows = []
    for variant in record["variants"]:
        plan, formats = resolve(record, variant, inputs, models)
        if stage == "loop":
            # The loop orderer prices the concordant program before the
            # workspacer, the formatter and the protocol optimizer run.
            plan = variant_plan(record, variant)
        rows.append(
            features(
                BASES[name],
                plan,
                formats,
                read_inputs(record, plan, inputs),
                stats_factory,
            )
        )
    return rows
