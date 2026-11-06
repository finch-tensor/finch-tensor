from __future__ import annotations

from collections.abc import Mapping, Sequence

from finch.algebra.tensor import TensorFType
from finch.autoschedule.cost_models import (
    PROTOCOL_COST_MODEL,
    CostModel,
    LevelOption,
    ProtocolChoice,
)
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    program_cost,
)
from finch.autoschedule.stages import LoopOrderedForm
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
    Plan,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import Namespace, PostOrderDFS


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


def add_workspaces(
    prgm: Plan,
    bindings: Mapping[Alias, TensorFType],
    stats: Mapping[Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice],
    options: Sequence[LevelOption] = LEVEL_OPTIONS,
) -> Plan:
    """
    Redirect each query of prgm which scatters into an output of prgm into a
    workspace fused over the fields it writes in loop order, as add_workspace
    does, if the workspace stores fewer fields than the output, and prgm costs
    less with it than without it under cost_model, once the formats and
    protocols of each are chosen, as program_cost prices them. The inputs of
    bindings have no workspaces.
    """

    def cost(stmt: Plan) -> float:
        return program_cost(stmt, bindings, stats, stats_factory, cost_model, options)

    prgm = flatten_plans(prgm)
    spc = Namespace(prgm)
    outputs: dict[HardAlias, None] = {}
    for node in PostOrderDFS(prgm):
        match node:
            case Query(Table(HardAlias() as lhs, _), _) if lhs not in bindings:
                outputs[lhs] = None
    best = None
    for alias in outputs:
        depth = workspace_depth(prgm, alias)
        if not depth:
            continue
        staged = add_workspace(
            prgm, alias, depth, HardAlias(spc.freshen(alias.name, "ws"))
        )
        assert isinstance(staged, Plan)
        if best is None:
            best = cost(prgm)
        staged_cost = cost(staged)
        if staged_cost < best:
            prgm, best = staged, staged_cost
    return prgm


class LogicWorkspacer(LoopOrderedForm, LogicLoader):
    """
    Pass a program in LoopOrderedForm to ctx, after redirecting the queries
    which scatter into their outputs into workspaces where they cost less
    under cost_model, by default PROTOCOL_COST_MODEL, as add_workspaces
    does. A loop orderer wraps it, so that the copies out of workspaces aren't
    propagated, and the formatter it wraps should price programs with the same
    model.
    """

    def __init__(
        self,
        ctx: LogicLoader,
        cost_model: CostModel[ProtocolChoice] | None = None,
        options: Sequence[LevelOption] = LEVEL_OPTIONS,
    ):
        self.ctx = ctx
        if cost_model is None:
            cost_model = PROTOCOL_COST_MODEL
        self.cost_model = cost_model
        self.options = options

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[Alias, TensorFType]]:
        assert isinstance(prgm, Plan)
        prgm = add_workspaces(
            prgm, bindings, stats, stats_factory, self.cost_model, self.options
        )
        return self.ctx(prgm, bindings, stats, stats_factory)
