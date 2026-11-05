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
- level: for a random program, every assignment of the formats of its tensors,
  with the protocols the protocol optimizer chooses.
- transpose: for a random input, every order a copy of it can be stored in,
  which the copy reads out of order, as the loop orderer transposes an input
  it would read out of order. The loop orderer charges such copies with the
  transpose model, on top of the loop model.
- loop: for a random program, every loop order, with its inputs stored in
  that order, so that the program is concordant, the formats the formatter
  chooses, from the supports of its inputs, and the protocols the protocol
  optimizer chooses. Its features are those of the supports, before the
  formatter and the protocol optimizer run. loop_orderer_model puts the loop
  and transpose models together for the loop orderer.
"""

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from finch.algebra import StaticFillFType, ftype, ftypes
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
    LinearCostModel,
    PigeonBasis,
    TransposeBasis,
    level_options,
)
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_BASIS,
    LEVEL_BUILDERS,
    LEVEL_OPTIONS,
    STORAGE_COST_MODEL,
    optimize_format,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
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
    parse_logic,
)
from finch.finch_logic.parser import PROTOCOLS
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite
from finch.tensor import element, fiber_tensor
from scripts.fitting.kronecker_stats import KroneckerStatsFactory, KroneckerTensor

STAGES = ("protocol", "level", "transpose", "loop")
OPTIONS = {option.kind: option for option in LEVEL_OPTIONS}
A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")
INPUTS = (A, B)

BASES: dict[str, Basis] = {
    "protocol": Affine(PROTOCOL_BASIS),
    "level": Affine(LEVEL_BASIS),
    "transpose": Affine(TransposeBasis()),
    "loop": Affine(PigeonBasis()),
}

# The models the optimizers of each stage use until they are fitted.
DEFAULT_MODELS: dict[str, CostModel] = {
    "protocol": PROTOCOL_COST_MODEL,
    "level": STORAGE_COST_MODEL,
}


def load_model(name: str, path: Path) -> LinearCostModel:
    """The model of the basis name which fit_physical_data wrote to path."""
    basis = BASES[name]
    untrained = LinearCostModel(basis, np.zeros(basis.n_features))
    return untrained.from_json(json.loads(path.read_text()))


def parse_models(args: list[str] | None) -> dict[str, CostModel]:
    """The models of NAME=PATH arguments, over the defaults."""
    models = dict(DEFAULT_MODELS)
    for arg in args or ():
        name, _, path = arg.partition("=")
        if name not in BASES or not path:
            raise ValueError(
                f"Expected NAME=PATH with NAME of {list(BASES)}, got {arg}"
            )
        models[name] = load_model(name, Path(path))
    return models


def loop_orderer_model(
    loop: LinearCostModel, transpose: LinearCostModel
) -> LinearCostModel:
    """
    The model of the loop orderer, which prices a loop order by the loop model
    on the concordant program and the transpose model on the inputs it would
    have to transpose, as LOOP_COST_MODEL does with the built-in weights.
    """
    return LinearCostModel(
        Concat(loop.basis, transpose.basis),
        np.concatenate([loop.coefficients, transpose.coefficients]),
    )


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


def variant_formats(variant: Mapping[str, Any]) -> dict[Alias, list[LevelOption]]:
    """The level options of each tensor which a variant gives a format."""
    return {
        HardAlias(name): [OPTIONS[kind] for kind in kinds]
        for name, kinds in variant.get("formats", {}).items()
    }


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
    """The fields of the table of each alias plan reads, in field order."""
    fields = {}
    for node in PostOrderDFS(plan):
        match node:
            case Table(Alias() as tns, idxs):
                fields.setdefault(tns, idxs)
    return fields


def format_ftype(options: list[LevelOption], fill_type, shape_type):
    """The ftype of a tensor of the level options, fill_type, and shape_type."""
    lvl = element(fill_type, fill_type.element_type, ftypes.intp)
    for option, dim in reversed(list(zip(options, shape_type, strict=True))):
        lvl = LEVEL_BUILDERS[option.kind](lvl, dim)
    return fiber_tensor(lvl)


def input_ftype(options: list[LevelOption], tensor: KroneckerTensor):
    """The ftype of tensor stored with the level options."""
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


def choose_formats(
    plan: LogicStatement,
    inputs: Mapping[Alias, KroneckerTensor],
    stats_factory: StatsFactory,
    cost_model: CostModel,
) -> dict[Alias, list[LevelOption]]:
    """
    The level options of each tensor of plan under cost_model, as the
    formatter would choose them from the supports of inputs, which store their
    fields in the order plan reads them, for the inputs too, which it takes to
    be unformatted.
    """
    fields = input_fields(plan)
    stats = OrderedDict(
        (tns, stats_factory(tensor, fields[tns])) for tns, tensor in inputs.items()
    )
    fill_types = plan.infer_fill_type(
        {tns: StaticFillFType(tensor.dtype(0)) for tns, tensor in inputs.items()}
    )
    formats = {}
    for tns, idxs in fields.items():
        lvl = optimize_format(
            tns,
            (ftypes.intp,) * len(idxs),
            StatsMachine(stats_factory, OrderedDict(stats)),
            fill_types[tns],
            cost_model,
            plan,
            LEVEL_OPTIONS,
        )
        options = level_options(fiber_tensor(lvl))
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
    inputs: the formats of every tensor, and the protocols each input is read
    with, which the plan of the protocol stage already gives. The inputs the
    program reads are stored in the order it reads them, which is the loop
    order in the loop stage.
    """
    stage = record["stage"]
    plan = variant_plan(record, variant)
    formats = variant_formats(variant)
    inputs = read_inputs(record, plan, inputs)
    stats_factory = KroneckerStatsFactory(record["block_bits"])
    if stage == "loop":
        formats = choose_formats(plan, inputs, stats_factory, models["level"])
    if stage != "protocol":
        fields = input_fields(plan)
        plan = optimize_protocols(
            plan,
            {tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()},
            {tns: stats_factory(tensor, fields[tns]) for tns, tensor in inputs.items()},
            stats_factory,
            models["protocol"],
        )
    return plan, formats


def features(
    bases: Mapping[str, Basis], plan, formats, inputs, stats_factory, protocols=True
) -> dict[str, np.ndarray]:
    """
    The features of each of bases of plan, with the level options of formats,
    from the stats of inputs, which store their fields in the order plan reads
    them. Unless protocols is false, each read without a protocol takes the
    default of its level, as Finch.jl reads it.
    """
    fields = input_fields(plan)
    stats = {tns: stats_factory(tensor, fields[tns]) for tns, tensor in inputs.items()}
    if protocols:
        ftypes = {
            tns: input_ftype(formats[tns], tensor) for tns, tensor in inputs.items()
        }
        plan = protocolize(plan, ftypes)
    formats = {
        alias: [*options, ElementLevelOption()] for alias, options in formats.items()
    }
    return {
        name: basis.get_features(plan, formats, stats, stats_factory)
        for name, basis in bases.items()
    }


def featurize_record(
    record: Mapping[str, Any],
    inputs: Mapping[Alias, KroneckerTensor],
    names,
    models: Mapping[str, CostModel],
) -> dict[str, list[np.ndarray]]:
    """
    The features of each basis of names of each variant of the example of
    record, whose inputs are inputs, as it was timed: the stages below make
    their choices with models, as time_physical_data makes them, so nothing is
    compiled or run. The loop stage orders loops before its inputs are
    formatted, so it reads them without protocols.
    """
    stage = record["stage"]
    bases = {name: BASES[name] for name in names}
    stats_factory = KroneckerStatsFactory(record["block_bits"])
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for variant in record["variants"]:
        plan, formats = resolve(record, variant, inputs, models)
        if stage == "loop":
            # The loop orderer prices the concordant program before the
            # formatter and the protocol optimizer run, without protocols.
            plan = variant_plan(record, variant)
        found = features(
            bases,
            plan,
            formats,
            read_inputs(record, plan, inputs),
            stats_factory,
            protocols=stage != "loop",
        )
        for name in names:
            rows[name].append(found[name])
    return rows
