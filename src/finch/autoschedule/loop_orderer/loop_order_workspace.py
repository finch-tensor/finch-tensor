from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.autoschedule.cost_models import (
    WORKSPACE_LOOP_COST_MODEL,
    CostModel,
    LevelOption,
    Prefix,
    ProtocolComponent,
)
from finch.autoschedule.cost_models.loop_cost_models import WORKSPACE_LIMIT
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    fitted_cost_model,
    program_cost,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import flatten_plans, fuse_loop_order
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

from .loop_order_bnb import BFSLoopOrderer
from .loop_ordering import with_copy_loop_orders


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


class WorkspaceLoopOrderer(BFSLoopOrderer):
    """
    Search for the loop order of each query like BFSLoopOrderer, pricing the
    writes which scatter into an output by WORKSPACE_LOOP_COST_MODEL, as cheaply
    as a workspace takes them. Then redirect each query whose loop order
    scatters into its output into a workspace fused over the fields it writes
    in order, which is copied into the output in order after each slice, if
    the workspace stores fewer fields than the output, and a slice of it has
    at most workspace_limit entries, as the WriteBasis of the cost model
    assumes. The formatter chooses the format of each workspace.
    """

    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        k: int | None = None,
        cost_model: CostModel[Prefix] | None = None,
        workspace_limit: float = WORKSPACE_LIMIT,
    ):
        if cost_model is None:
            cost_model = WORKSPACE_LOOP_COST_MODEL
        super().__init__(ctx, k=k, cost_model=cost_model)
        self.workspace_limit = workspace_limit

    def add_workspaces(
        self,
        prgm: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        # Copies are given their loop orders first, so that a copy in order
        # isn't mistaken for one which scatters.
        prgm = with_copy_loop_orders(flatten_plans(prgm))
        spc = Namespace(prgm)
        outputs: dict[HardAlias, tuple[Field, ...]] = {}
        for node in PostOrderDFS(prgm):
            match node:
                case Query(Table(HardAlias() as lhs, idxs), _) if lhs not in bindings:
                    outputs[lhs] = idxs
        for alias, idxs in outputs.items():
            depth = workspace_depth(prgm, alias)
            if depth is None or depth == 0:
                continue
            ws = HardAlias(spc.freshen(alias.name, "ws"))
            staged = add_workspace(prgm, alias, depth, ws)
            assert isinstance(staged, Plan)
            if self.keep_workspace(
                prgm, staged, alias, idxs[depth:], bindings, stats, stats_factory
            ):
                prgm = staged
        return prgm

    def keep_workspace(
        self,
        prgm: Plan,
        staged: Plan,
        alias: HardAlias,
        slice_idxs: tuple[Field, ...],
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> bool:
        """
        Whether staged, which redirects the query defining alias into a
        workspace storing slice_idxs, replaces prgm: if a slice of the
        workspace has at most workspace_limit entries.
        """
        # The stats of the output give the extent of each of its fields.
        machine = StatsMachine(stats_factory, OrderedDict(stats))
        machine(prgm)
        output = machine.bindings[alias]
        size = np.prod([output.get_dim_size(idx) for idx in slice_idxs])
        return bool(size <= self.workspace_limit)


class CostedWorkspaceLoopOrderer(WorkspaceLoopOrderer):
    """
    Order loops and propose workspaces like WorkspaceLoopOrderer, but keep a
    workspace only if the program with it costs less than the program without
    it under format_cost_model, by default the fitted format model, once the
    formats of each are chosen and its protocols, as program_cost prices them.
    The formatter downstream should price programs with the same model.
    """

    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        k: int | None = None,
        cost_model: CostModel[Prefix] | None = None,
        format_cost_model: CostModel[ProtocolComponent] | None = None,
        options: Sequence[LevelOption] = LEVEL_OPTIONS,
    ):
        super().__init__(ctx, k=k, cost_model=cost_model)
        if format_cost_model is None:
            format_cost_model = fitted_cost_model()
        self.format_cost_model = format_cost_model
        self.options = options

    def keep_workspace(
        self,
        prgm: Plan,
        staged: Plan,
        alias: HardAlias,
        slice_idxs: tuple[Field, ...],
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> bool:
        def cost(stmt: Plan) -> float:
            return program_cost(
                stmt,
                bindings,
                stats,
                stats_factory,
                self.format_cost_model,
                self.options,
            )

        return cost(staged) < cost(prgm)
