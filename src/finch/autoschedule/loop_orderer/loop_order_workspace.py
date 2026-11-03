from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from finch.algebra import AbstractFillFType, FType
from finch.algebra.tensor import TensorFType
from finch.autoschedule.cost_models import CostModel, LevelChoice, LevelOption, Prefix
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    WORKSPACE_COST_MODEL,
    optimize_format,
    stored_shape_types,
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
from finch.symbolic import Namespace

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
    Search for the loop order of each query like BFSLoopOrderer, then redirect
    each query whose loop order scatters into its output into a workspace fused
    over the fields it writes in order, when that costs less. The workspace is
    copied into the output in order after each slice. A workspace costs less
    when the cheapest formats of the workspace and of the output it is copied
    into cost less under format_cost_model than the cheapest format of the
    output alone. The formatter then chooses the formats.
    """

    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        k: int | None = None,
        cost_model: CostModel[Prefix] | None = None,
        format_cost_model: CostModel[LevelChoice] = WORKSPACE_COST_MODEL,
        format_options: Sequence[LevelOption] = LEVEL_OPTIONS,
    ):
        super().__init__(ctx, k=k, cost_model=cost_model)
        self.format_cost_model = format_cost_model
        self.format_options = format_options

    def format_cost(
        self,
        alias: Alias,
        term: LogicStatement,
        machine: StatsMachine,
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
    ) -> float:
        cost, _ = optimize_format(
            alias,
            shape_type,
            machine,
            fill_type,
            self.format_cost_model,
            term,
            self.format_options,
        )
        return cost

    def add_workspaces(
        self,
        prgm: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        # The format costs need the loop order of every query, copies included.
        prgm = with_copy_loop_orders(flatten_plans(prgm))
        initial = StatsMachine(stats_factory, OrderedDict(stats))
        stored = stored_shape_types(
            prgm, bindings, StatsMachine(stats_factory, OrderedDict(stats))
        )
        fill_types = prgm.infer_fill_type(
            {var: val.fill_type for var, val in bindings.items()}
        )
        spc = Namespace(prgm)
        for alias, shape_type in stored.items():
            depth = workspace_depth(prgm, alias)
            if depth is None:
                continue
            assert isinstance(alias, HardAlias)
            ws = HardAlias(spc.freshen(alias.name, "ws"))
            staged = add_workspace(prgm, alias, depth, ws)
            assert isinstance(staged, Plan)
            fill_type = fill_types[alias]
            direct = self.format_cost(alias, prgm, initial, fill_type, shape_type)
            copied = self.format_cost(alias, staged, initial, fill_type, shape_type)
            written = self.format_cost(
                ws, staged, initial, fill_type, shape_type[depth:]
            )
            if copied + written < direct:
                prgm = staged
        return prgm
