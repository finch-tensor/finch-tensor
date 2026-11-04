"""
The stages of fitting the cost models of the physical optimizers, from the
innermost decision out. Each stage enumerates every option of its own decision
for random programs, and the stages below it, which are already fitted, make
theirs, so the runtime of each option is what the optimizer would get by
choosing it, and the best option of each example is known. Each stage names
the basis its model is fitted to, which featurize_physical_data extracts and
condense_physical_data collects under the name of the stage, so several stages
can share DIR, and a new basis needs no new timings.

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

from finch.algebra import Tensor, ftypes
from finch.autoschedule import optimize_protocols
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    Affine,
    Basis,
    Concat,
    CostModel,
    LevelOption,
    LinearCostModel,
    PigeonBasis,
    TransposeBasis,
    level_options,
)
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_BASIS,
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
from finch.tensor import fiber_tensor

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


def features_kind(name: str) -> str:
    return f"{name}.features.json"


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


def choose_formats(
    plan: LogicStatement,
    bindings: Mapping[Alias, Tensor],
    stats_factory: StatsFactory,
    cost_model: CostModel,
) -> dict[Alias, list[LevelOption]]:
    """
    The level options of each tensor of plan under cost_model, as the
    formatter would choose them from the supports of the inputs of bindings,
    for the inputs too, which it takes to be unformatted.
    """
    fields = input_fields(plan)
    stats = OrderedDict(
        (tns, stats_factory(arg, fields[tns])) for tns, arg in bindings.items()
    )
    fill_types = plan.infer_fill_type(
        {tns: arg.ftype.fill_type for tns, arg in bindings.items()}
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
    bindings: Mapping[Alias, Tensor],
    stats_factory: StatsFactory,
    models: Mapping[str, CostModel],
    reformat,
) -> tuple[LogicStatement, dict[Alias, list[LevelOption]], dict[Alias, Tensor]]:
    """
    The program, formats, and inputs of a variant of the example of record,
    once the stages below its stage have made their choices: the formats of
    every tensor, the inputs of bindings copied into theirs by reformat, in the
    order the program reads them, which is the loop order in the loop stage,
    and the protocols each input is read with, which the plan of the protocol
    stage already gives. reformat takes a tensor, its level options, and the
    permutation of its fields to store.
    """
    stage = record["stage"]
    plan = variant_plan(record, variant)
    formats = variant_formats(variant)
    if stage == "loop":
        formats = choose_formats(plan, bindings, stats_factory, models["level"])
    # The inputs the program reads are stored in the order it reads them.
    perms = input_perms(record, plan)
    bindings = {
        tns: reformat(arg, formats[tns], perms[tns])
        for tns, arg in bindings.items()
        if tns in perms
    }
    if stage != "protocol":
        fields = input_fields(plan)
        stats = {tns: stats_factory(arg, fields[tns]) for tns, arg in bindings.items()}
        plan = optimize_protocols(
            plan,
            {tns: arg.ftype for tns, arg in bindings.items()},
            stats,
            stats_factory,
            models["protocol"],
        )
    return plan, formats, bindings
