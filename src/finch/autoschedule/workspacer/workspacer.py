from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence

from finch.algebra import ftype, ftypes
from finch.algebra.tensor import TensorFType
from finch.autoschedule.cost_models import (
    PROTOCOL_COST_MODEL,
    CostModel,
    LevelOption,
    ProtocolChoice,
)
from finch.autoschedule.formatter.galley_formatter import GalleyFormatter
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    optimize_formats,
)
from finch.autoschedule.protocolizer.protocolize import choose_protocols
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import flatten_plans, fuse_loop_order
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import (
    Alias,
    Field,
    Fuse,
    FusedAlias,
    HardAlias,
    LogicLoader,
    LogicStatement,
    MockLogicLoader,
    Plan,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import Namespace, PostOrderDFS, PostWalk, Rewrite


def workspace_depth(prgm: LogicStatement, alias: Alias) -> int | None:
    """
    Return the number of leading fields that the query defining alias writes in
    loop order, when it scatters into a later field. A workspace fused over
    those fields can then take the scattered writes. Only an alias which one
    Query defines, and nothing updates, has a workspace.
    """
    writes: list[tuple[LogicStatement, tuple[Field, ...]]] = []

    def visit(node: LogicStatement, loops: tuple[Field, ...]):
        match node:
            case Plan(bodies):
                for body in bodies:
                    visit(body, loops)
            case Fuse(idx, body):
                visit(body, (*loops, idx))
            case Query(Table(Alias() as lhs, _), _) | QueryInto(
                Table(Alias() as lhs, _), _, _
            ) if lhs.unfused == alias:
                writes.append((node, loops))

    visit(prgm, ())
    match writes:
        case [(Query(Table(HardAlias(), idxs), _), loops)]:
            depth = 0
            while loops[depth : depth + 1] == idxs[depth : depth + 1] != ():
                depth += 1
            if depth < len(idxs):
                return depth
    return None


def add_workspace(
    prgm: LogicStatement, alias: HardAlias, depth: int, ws: HardAlias
) -> LogicStatement:
    """
    Redirect the query which defines alias into ws, fused over its first depth
    fields. Right after the statement within depth Fuse nodes which holds the
    query, ws is copied into alias in order.
    """
    fused = FusedAlias(ws, depth)

    # Each rewrite returns the copy until the depth where it is inserted.
    def rewrite(
        node: LogicStatement, nloops: int
    ) -> tuple[LogicStatement, LogicStatement | None]:
        match node:
            case Plan(bodies):
                stmts: list[LogicStatement] = []
                pending = None
                for body in bodies:
                    body, copy = rewrite(body, nloops)
                    stmts.append(body)
                    if copy is not None and nloops == depth:
                        stmts.append(copy)
                    elif copy is not None:
                        pending = copy
                return Plan(tuple(stmts)), pending
            case Fuse(idx, body):
                body, copy = rewrite(body, nloops + 1)
                if copy is not None and nloops + 1 == depth:
                    return Fuse(idx, Plan((body, copy))), None
                return Fuse(idx, body), copy
            case Query(Table(lhs, idxs), rhs) if lhs == alias:
                copy = fuse_loop_order(
                    Query(Table(alias, idxs), Table(fused, idxs)), idxs[depth:]
                )
                return Query(Table(fused, idxs), rhs), copy
            case _:
                return node, None

    prgm, copy = rewrite(prgm, 0)
    if copy is not None:
        prgm = Plan((prgm, copy))
    return prgm


def stage_workspaces(
    prgm: Plan,
    bindings: Mapping[Alias, TensorFType],
    keep: Callable[[Plan, Plan, HardAlias, HardAlias], bool],
) -> Plan:
    """
    Redirect each query of prgm which scatters into an output of prgm into a
    workspace fused over the fields it writes in loop order, as add_workspace
    does, if the workspace stores fewer fields than the output, and keep, given
    prgm, the program with the workspace, the output, and the workspace, keeps
    it. The inputs of bindings have no workspaces.
    """
    prgm = flatten_plans(prgm)
    spc = Namespace(prgm)
    outputs: dict[HardAlias, None] = {}
    for node in PostOrderDFS(prgm):
        match node:
            case Query(Table(HardAlias() as lhs, _), _) if lhs not in bindings:
                outputs[lhs] = None
    for alias in outputs:
        depth = workspace_depth(prgm, alias)
        if not depth:
            continue
        ws = HardAlias(spc.freshen(alias.name, "ws"))
        staged = add_workspace(prgm, alias, depth, ws)
        assert isinstance(staged, Plan)
        if keep(prgm, staged, alias, ws):
            prgm = staged
    return prgm


def format_cost(
    prgm: LogicStatement,
    bindings: Mapping[Alias, TensorFType],
    stats: Mapping[Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice],
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
) -> tuple[dict[Alias, TensorFType], float]:
    """
    The formats of the aliases prgm defines which bindings doesn't format, as
    optimize_formats chooses them, and the cost of prgm with them under
    cost_model, once choose_protocols chooses its protocols.
    """
    shape_types = prgm.infer_shape_type(
        {var: val.shape_type for var, val in bindings.items()}
    )
    # Each new alias stores the fields its first definition doesn't fuse.
    nfused: dict[Alias, int] = {}
    for node in PostOrderDFS(prgm):
        match node:
            case Query(Table(Alias() as lhs, _), _) | QueryInto(
                Table(Alias() as lhs, _), _, _
            ) if lhs.unfused not in bindings:
                nfused.setdefault(lhs.unfused, lhs.nfused)
    stored = {
        alias: tuple(
            ftype(dim) if dim is not None else ftypes.intp
            for dim in shape_types[alias][n:]
        )
        for alias, n in nfused.items()
    }
    fill_types = prgm.infer_fill_type(
        {var: val.fill_type for var, val in bindings.items()}
    )
    return optimize_formats(
        prgm, bindings, stats, stats_factory, stored, fill_types, cost_model, options
    )


def galley_format_cost(
    prgm: LogicStatement,
    bindings: Mapping[Alias, TensorFType],
    stats: Mapping[Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice],
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
) -> tuple[dict[Alias, TensorFType], float]:
    """
    format_cost with the formats GalleyFormatter chooses from the stats of each
    alias, rather than those optimize_formats searches for. The program is
    priced once, so a workspace costs one choice of protocols rather than a
    search over formats; the formats are those a GalleyFormatter downstream
    gives the program. options is unused.
    """
    _, formatted = GalleyFormatter(MockLogicLoader()).lower(
        prgm, dict(bindings), dict(stats), stats_factory
    )
    _, costs = choose_protocols(prgm, formatted, stats, stats_factory, cost_model)
    formats = {alias: fmt for alias, fmt in formatted.items() if alias not in bindings}
    return formats, sum(costs)


FormatCost = Callable[..., tuple[dict[Alias, TensorFType], float]]


def is_dense(stats: TensorStats) -> bool:
    assert isinstance(stats, NumericStats)
    return stats.estimate_non_fill_values() >= 0.5 * math.prod(stats.dim_sizes.values())


def add_cost_effective_workspaces(
    prgm: Plan,
    bindings: Mapping[Alias, TensorFType],
    stats: Mapping[Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice],
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
    format_cost: FormatCost = format_cost,
) -> Plan:
    """
    Add workspaces that improve cost, as defined by the format_cost function.
    Because this relies on further optimization, we need to be a bit careful
    to only reoptimize formats for the current alias and workspace.
    """
    # The formats and cost of the program so far, found when a workspace is
    # first priced, and updated with each workspace kept.
    formats: dict[Alias, TensorFType] | None = None
    cost = math.inf

    def keep(prgm, staged, alias, ws) -> bool:
        nonlocal formats, cost
        machine = StatsMachine(stats_factory, OrderedDict(stats))
        machine(prgm)
        if is_dense(machine.bindings[alias]):
            return False
        if formats is None:
            formats, cost = format_cost(
                prgm, bindings, stats, stats_factory, cost_model, options
            )
        fixed = {tns: fmt for tns, fmt in formats.items() if tns != alias}
        staged_formats, staged_cost = format_cost(
            staged, {**bindings, **fixed}, stats, stats_factory, cost_model, options
        )
        if staged_cost < cost:
            formats, cost = {**fixed, **staged_formats}, staged_cost
            return True
        return False

    return stage_workspaces(prgm, bindings, keep)


def add_reformat(
    prgm: Plan, alias: HardAlias, idxs: tuple[Field, ...], copy: HardAlias
) -> Plan:
    """
    Copy alias into copy before prgm, looping over idxs, the fields of a table
    which reads alias, in the order alias stores them, and read copy wherever
    prgm reads alias, so that the formatter may store copy in any format.
    """

    def read(node):
        match node:
            case Table(tns, idxs_2, protocols) if tns == alias:
                return Table(copy, idxs_2, protocols)
        return None

    reformat = fuse_loop_order(Query(Table(copy, idxs), Table(alias, idxs)), idxs)
    reads = Rewrite(PostWalk(read))(prgm)
    assert isinstance(reads, Plan)
    return Plan((reformat, *reads.bodies))


def stage_reformats(
    prgm: Plan,
    bindings: Mapping[Alias, TensorFType],
    keep: Callable[[Plan, Plan, HardAlias, HardAlias], bool],
) -> Plan:
    """
    Copy each input of bindings which prgm reads, and doesn't write, into a
    new alias before prgm, as add_reformat does, if keep, given prgm, the
    program with the copy, the input, and the copy, keeps it. A scalar has no
    levels to store otherwise.
    """
    prgm = flatten_plans(prgm)
    spc = Namespace(prgm)
    reads: dict[HardAlias, tuple[Field, ...]] = {}
    written: set[Alias] = set()
    for node in PostOrderDFS(prgm):
        match node:
            case Query(Table(Alias() as lhs, _), _) | QueryInto(
                Table(Alias() as lhs, _), _, _
            ):
                written.add(lhs.unfused)
            case Table(HardAlias() as tns, idxs) if tns in bindings and idxs:
                reads.setdefault(tns, idxs)
    for alias, idxs in reads.items():
        if alias in written:
            continue
        copy = HardAlias(spc.freshen(alias.name, "copy"))
        staged = add_reformat(prgm, alias, idxs, copy)
        if keep(prgm, staged, alias, copy):
            prgm = staged
    return prgm


def add_cost_effective_reformats(
    prgm: Plan,
    bindings: Mapping[Alias, TensorFType],
    stats: Mapping[Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice],
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
    format_cost: FormatCost = format_cost,
) -> Plan:
    """
    Copy the inputs of prgm into new formats, as stage_reformats does, where
    the program costs less with the copy under cost_model, once format_cost
    formats the copy, keeping the formats of the other aliases prgm defines.
    """
    # The formats and cost of the program so far, found when a copy is first
    # priced, and updated with each copy kept.
    formats: dict[Alias, TensorFType] | None = None
    cost = math.inf

    def keep(prgm, staged, alias, copy) -> bool:
        nonlocal formats, cost
        if formats is None:
            formats, cost = format_cost(
                prgm, bindings, stats, stats_factory, cost_model, options
            )
        staged_formats, staged_cost = format_cost(
            staged, {**bindings, **formats}, stats, stats_factory, cost_model, options
        )
        if staged_cost < cost:
            formats, cost = {**formats, **staged_formats}, staged_cost
            return True
        return False

    return stage_reformats(prgm, bindings, keep)


class LogicWorkspacer(LoopOrderedForm, LogicLoader):
    """
    Pass a program in LoopOrderedForm to ctx, after redirecting the queries
    which scatter into their outputs into workspaces where they cost less
    under cost_model, by default PROTOCOL_COST_MODEL, as
    add_cost_effective_workspaces does, and then copying its inputs into new
    formats where that costs less, as add_cost_effective_reformats does. A
    loop orderer wraps it, so that the copies aren't propagated, and the
    formatter it wraps should price programs with the same model. format_cost
    prices a program with the formats its formatter would choose: format_cost
    searches them as a CostFormatter does, and galley_format_cost takes those
    of a GalleyFormatter, far more cheaply.
    """

    def __init__(
        self,
        ctx: LogicLoader,
        cost_model: CostModel[ProtocolChoice] | None = None,
        options: Sequence[LevelOption] = LEVEL_OPTIONS,
        format_cost: FormatCost = format_cost,
    ):
        self.ctx = ctx
        if cost_model is None:
            cost_model = PROTOCOL_COST_MODEL
        self.cost_model = cost_model
        self.options = options
        self.format_cost = format_cost

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[Alias, TensorFType]]:
        assert isinstance(prgm, Plan)
        prgm = add_cost_effective_workspaces(
            prgm,
            bindings,
            stats,
            stats_factory,
            self.cost_model,
            self.options,
            self.format_cost,
        )
        prgm = add_cost_effective_reformats(
            prgm,
            bindings,
            stats,
            stats_factory,
            self.cost_model,
            self.options,
            self.format_cost,
        )
        return self.ctx(prgm, bindings, stats, stats_factory)


def add_1d_workspaces(prgm: Plan, bindings: Mapping[Alias, TensorFType]) -> Plan:
    """
    Stage the workspaces of prgm, as stage_workspaces does, which store one
    field, the only one the query scatters into, without pricing them.
    """

    def keep(_, staged, __, ws) -> bool:
        # The workspace stores the fields of its table which it isn't fused over.
        for node in PostOrderDFS(staged):
            match node:
                case Table(FusedAlias(alias=tns, n=n), idxs) if tns == ws:
                    return len(idxs) - n == 1
        return False

    return stage_workspaces(prgm, bindings, keep)


class DefaultLogicWorkspacer(LoopOrderedForm, LogicLoader):
    """
    Pass a program in LoopOrderedForm to ctx, after redirecting the queries
    which scatter into one field of their outputs into workspaces, as
    add_1d_workspaces does, without statistics. A loop orderer wraps it,
    so that the copies out of workspaces aren't propagated.
    """

    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[Alias, TensorFType]]:
        assert isinstance(prgm, Plan)
        prgm = add_1d_workspaces(prgm, bindings)
        return self.ctx(prgm, bindings, stats, stats_factory)
