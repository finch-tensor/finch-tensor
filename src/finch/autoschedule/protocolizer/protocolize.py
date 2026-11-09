import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from itertools import product
from typing import TypeVar

from finch import finch_logic as lgc
from finch.algebra import TensorFType
from finch.autoschedule.compiler import lower_protocol
from finch.autoschedule.cost_models import (
    PROTOCOL_COST_MODEL,
    CostModel,
    LevelOption,
    Prefix,
    ProtocolChoice,
    level_accesses,
    level_formats,
    level_reads,
    read_options,
    rescans_a_root,
)
from finch.autoschedule.stages import FormattedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import LogicLoader, StatsFactory, TensorStats
from finch.symbolic import PostOrderDFS, PostWalk, PreWalk, Rewrite

_PROTOCOLS = (lgc.DefaultProtocol(), lgc.Walk(), lgc.Gallop(), lgc.Follow())


def protocolize(
    prgm: lgc.LogicStatement, bindings: Mapping[lgc.Alias, TensorFType]
) -> lgc.LogicStatement:
    """
    Read each level of a tensor of bindings with the protocol which the level
    uses by default, where the table which reads it gives none, so that every
    read of a formatted tensor states the protocol it is lowered with.
    """

    def read(node):
        match node:
            case lgc.Table(lgc.Alias() as tns, idxs, protocols) if (
                tns.unfused in bindings
            ):
                defaults = bindings[tns.unfused].default_protocols()
                if not defaults:
                    return None
                protocols_2: list[lgc.Protocol] = [*protocols] or [
                    lgc.DefaultProtocol()
                ] * len(idxs)
                # The fused fields of a table precede those its tensor stores.
                for pos, default in enumerate(defaults, tns.nfused):
                    if protocols_2[pos] == lgc.DefaultProtocol():
                        protocols_2[pos] = next(
                            proto
                            for proto in _PROTOCOLS
                            if lower_protocol(proto) == default
                        )
                if protocols_2 != [lgc.DefaultProtocol()] * len(idxs):
                    return lgc.Table(tns, idxs, tuple(protocols_2))
        return None

    # A query writes its table, which takes no protocols.
    def query(node):
        match node:
            case lgc.Query(lhs, rhs):
                return lgc.Query(lhs, Rewrite(PostWalk(read))(rhs))
            case lgc.QueryInto(lhs, op, rhs):
                return lgc.QueryInto(lhs, op, Rewrite(PostWalk(read))(rhs))
        return None

    return Rewrite(PreWalk(query))(prgm)


class LogicProtocolizer(FormattedForm, LogicLoader):
    """
    Pass a program in FormattedForm to ctx, after reading each level of a
    formatted tensor with an explicit protocol, as protocolize does.
    """

    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, TensorFType]]:
        return self.ctx(protocolize(prgm, bindings), bindings, stats, stats_factory)


def with_protocol(
    expr: lgc.LogicExpression,
    tns: lgc.Alias,
    idxs: tuple[lgc.Field, ...],
    depth: int,
    protocol: lgc.Protocol,
) -> lgc.LogicExpression:
    """expr, where each table of tns and idxs reads its level at a zero-based
    stored depth with protocol."""

    def read(node):
        match node:
            case lgc.Table(tns_2, idxs_2, protocols) if (tns_2, idxs_2) == (tns, idxs):
                protocols_2: list[lgc.Protocol] = [*protocols] or [
                    lgc.DefaultProtocol()
                ] * len(idxs)
                protocols_2[tns.nfused + depth] = protocol
                return lgc.Table(tns, idxs, tuple(protocols_2))
        return None

    return Rewrite(PostWalk(read))(expr)


def touches(stmt: lgc.LogicStatement, aliases: set[lgc.Alias]) -> bool:
    """Whether stmt reads or writes a tensor of aliases."""
    for node in PostOrderDFS(stmt):
        match node:
            case lgc.Table(lgc.Alias() as tns) if tns.unfused in aliases:
                return True
    return False


def given_protocols(prgm: lgc.LogicStatement) -> set[tuple]:
    """The reads of prgm which give their protocol, each as the tensor and
    fields of its table and the position of the field it reads."""
    given = set()
    for node in PostOrderDFS(prgm):
        match node:
            case lgc.Table(tns, idxs, protocols):
                given.update(
                    (tns, idxs, pos)
                    for pos, proto in enumerate(protocols)
                    if proto != lgc.DefaultProtocol()
                )
    return given


def rebuilder(
    stmt: lgc.Query | lgc.QueryInto,
) -> Callable[[lgc.LogicExpression], lgc.Query | lgc.QueryInto]:
    """The function which rebuilds stmt with another right-hand side."""
    match stmt:
        case lgc.Query(lhs, _):
            return partial(lgc.Query, lhs)
        case lgc.QueryInto(lhs, op, _):
            return partial(lgc.QueryInto, lhs, op)


def merged_expr(stmt: lgc.Query | lgc.QueryInto) -> lgc.LogicExpression:
    """The expression the loops of stmt merge: its right-hand side, or the
    aggregate an update reduces into its table, as its aggregate query says."""
    match stmt:
        case lgc.QueryInto() as update:
            return update.as_query().rhs
        case lgc.Query(_, rhs):
            return rhs


def choose_protocols(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice] = PROTOCOL_COST_MODEL,
    cache: dict | None = None,
) -> tuple[lgc.LogicStatement, list[float]]:
    """
    Read each level of each formatted tensor of bindings with the protocol
    which costs least under cost_model, and return the program with the cost
    of each of its queries, in program order. The reads at a loop are chosen
    together, over the product of the protocols each supports, by the cost of
    the accesses at that loop alone, since the protocols at a loop don't
    change where the loops within it are entered. A read which gives its
    protocol keeps it. The calls of one search may share the cache of their
    stats machines.
    """
    formats = level_formats(bindings)
    given = given_protocols(prgm)
    prgm = protocolize(prgm, bindings)
    machine = StatsMachine(stats_factory, OrderedDict(stats), cache)
    found: list[float] = []

    def choose(stmt: lgc.Query | lgc.QueryInto, loop_order: tuple[lgc.Field, ...]):
        """stmt, reading each level with the protocol which costs least, and
        its cost."""
        match stmt:
            case lgc.Query(lgc.Table(_, lhs_idxs) as lhs, rhs) | lgc.QueryInto(
                lgc.Table(_, lhs_idxs) as lhs, _, rhs
            ):
                pass
        rebuild = rebuilder(stmt)
        cost = 0.0
        for n in range(len(loop_order)):
            loops = loop_order[: n + 1]
            reads = [
                read
                for read in level_reads(
                    Prefix(loops, merged_expr(rebuild(rhs)), lhs_idxs), formats
                )
                if (read.alias, read.table.idxs, read.alias.nfused + read.depth)
                not in given
            ]
            options = [
                [
                    (read.alias, read.table.idxs, read.depth, proto)
                    for proto in read_options(
                        read.option,
                        read.table.idxs,
                        read.alias.nfused,
                        read.depth,
                        loop_order,
                    )
                ]
                for read in reads
            ]
            best: tuple[float, lgc.LogicExpression] = (float("inf"), rhs)
            for choice in product(*options):
                if rescans_a_root(loops, reads, [c[3] for c in choice]):
                    continue
                rhs_2 = rhs
                for tns, idxs, depth, proto in choice:
                    rhs_2 = with_protocol(rhs_2, tns, idxs, depth, proto)
                prefix = Prefix(
                    loops,
                    merged_expr(rebuild(rhs_2)),
                    lhs_idxs,
                    formats,
                    output=lhs.tns,
                )
                total = sum(
                    cost_model.predict_component_cost(component, machine)
                    for component in level_accesses(prefix, formats)
                )
                if total < best[0]:
                    best = (total, rhs_2)
            total, rhs = best
            cost += total
        return rebuild(rhs), cost

    def visit(stmt: lgc.LogicStatement, loop_order: tuple[lgc.Field, ...]):
        match stmt:
            case lgc.Plan(bodies):
                return lgc.Plan(tuple(visit(body, loop_order) for body in bodies))
            case lgc.Fuse(idx, body):
                return lgc.Fuse(idx, visit(body, (*loop_order, idx)))
            case lgc.Produces():
                return stmt
            case lgc.QueryInto(lgc.Table(), _, lgc.Literal()):
                pass
            case lgc.Query() | lgc.QueryInto():
                stmt, cost = choose(stmt, loop_order)
                found.append(cost)
            case _:
                raise ValueError(f"Unsupported statement type: {stmt}")
        machine(stmt)
        return stmt

    return visit(prgm, ()), found


K = TypeVar("K", bound=Hashable)


def alias_costs(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice],
    alias: lgc.Alias,
    candidates: Mapping[K, TensorFType],
    costs: Sequence[float],
    cache: dict | None = None,
) -> dict[K, list[float]]:
    """
    The cost of each query of prgm, in program order, as choose_protocols
    gives it, for each format of candidates for alias, with the other tensors
    formatted as bindings says. The queries which don't touch alias cost what
    costs says. Formats whose levels agree on whether they filter and on their
    default protocols give prgm the same loop spaces, so each group of them is
    priced in one pass: at each loop, every protocol which a format of the
    group can read each level of alias with is tried with those of the other
    reads, each access of another tensor is priced once for each choice, and
    each access of alias once for the kinds of the levels it depends on. Each
    format then adds up its accesses in order and takes the cheapest choice
    whose protocols it can read.
    """
    given = given_protocols(prgm)
    options = {
        key: level_formats({alias: fmt})[alias] for key, fmt in candidates.items()
    }
    groups: dict[tuple, list[K]] = {}
    for key, opts in options.items():
        groups.setdefault(
            tuple((opt.filters, opt.default_protocol) for opt in opts), []
        ).append(key)
    found: dict[K, list[float]] = {key: [] for key in candidates}
    for members in groups.values():
        group = {**bindings, alias: candidates[members[0]]}
        machine = StatsMachine(stats_factory, OrderedDict(stats), cache)
        query_costs = _group_costs(
            protocolize(prgm, group),
            level_formats(group),
            machine,
            cost_model,
            alias,
            {key: options[key] for key in members},
            given,
        )
        for key in members:
            found[key] = [
                costs[query] if cost is None else cost[key]
                for query, cost in enumerate(query_costs)
            ]
    return found


def _group_costs(
    prgm: lgc.LogicStatement,
    formats: Mapping[lgc.Alias, Sequence[LevelOption]],
    machine: StatsMachine,
    cost_model: CostModel[ProtocolChoice],
    alias: lgc.Alias,
    options: Mapping[K, Sequence[LevelOption]],
    given: set[tuple],
) -> list[dict[K, float] | None]:
    """
    The cost of each query of prgm under each level options of options for
    alias, as alias_costs prices a group, with prgm read as formats says,
    whose options for alias agree with those of the group on which levels
    filter and on their default protocols, or None for a query which doesn't
    touch alias.
    """
    found: list[dict[K, float] | None] = []

    def protocols(read: ProtocolChoice, opts, loop_order) -> list[lgc.Protocol]:
        return read_options(
            opts[read.depth], read.table.idxs, read.alias.nfused, read.depth, loop_order
        )

    def price(
        stmt: lgc.Query | lgc.QueryInto, loop_order: tuple[lgc.Field, ...]
    ) -> dict[K, float]:
        match stmt:
            case lgc.Query(lgc.Table(_, lhs_idxs) as lhs, rhs) | lgc.QueryInto(
                lgc.Table(_, lhs_idxs) as lhs, _, rhs
            ):
                pass
        rebuild = rebuilder(stmt)
        totals = dict.fromkeys(options, 0.0)
        for n in range(len(loop_order)):
            loops = loop_order[: n + 1]
            reads = [
                read
                for read in level_reads(
                    Prefix(loops, merged_expr(stmt), lhs_idxs), formats
                )
                if (read.alias, read.table.idxs, read.alias.nfused + read.depth)
                not in given
            ]
            mine = [read.alias.unfused == alias for read in reads]
            # The protocols each format reads each level of alias with here.
            able = {
                key: [
                    protocols(read, opts, loop_order) if m else []
                    for read, m in zip(reads, mine, strict=True)
                ]
                for key, opts in options.items()
            }
            choices = [
                list(dict.fromkeys(p for key in options for p in able[key][i]))
                if m
                else protocols(read, read.options, loop_order)
                for i, (read, m) in enumerate(zip(reads, mine, strict=True))
            ]
            best = dict.fromkeys(options, math.inf)
            for choice in product(*choices):
                if rescans_a_root(loops, reads, choice):
                    continue
                keys = [
                    key
                    for key in options
                    if all(
                        not m or proto in allowed
                        for m, allowed, proto in zip(
                            mine, able[key], choice, strict=True
                        )
                    )
                ]
                if not keys:
                    continue
                rhs_2 = rhs
                for read, proto in zip(reads, choice, strict=True):
                    rhs_2 = with_protocol(
                        rhs_2, read.alias, read.table.idxs, read.depth, proto
                    )
                prefix = Prefix(
                    loops,
                    merged_expr(rebuild(rhs_2)),
                    lhs_idxs,
                    formats,
                    output=lhs.tns,
                )
                accesses = [
                    (access, 0.0)
                    if access.alias.unfused == alias
                    else (None, cost_model.predict_component_cost(access, machine))
                    for access in level_accesses(prefix, formats)
                ]
                # A read costs the same for the formats which agree on the kind
                # of its level, and a write for those which agree on the kinds
                # of the levels down to its own.
                priced: dict[tuple, float] = {}
                for key in keys:
                    opts = options[key]
                    access_costs = []
                    for i, (access, cost) in enumerate(accesses):
                        if access is not None:
                            kinds = opts[access.depth : access.depth + 1]
                            if access.write:
                                kinds = opts[: access.depth + 1]
                            memo = (i, tuple(opt.kind for opt in kinds))
                            if memo not in priced:
                                priced[memo] = cost_model.predict_component_cost(
                                    replace(access, options=opts), machine
                                )
                            cost = priced[memo]
                        access_costs.append(cost)
                    total = sum(access_costs)
                    if total < best[key]:
                        best[key] = total
            for key in options:
                totals[key] += best[key]
        return totals

    def visit(stmt: lgc.LogicStatement, loop_order: tuple[lgc.Field, ...]):
        match stmt:
            case lgc.Plan(bodies):
                for body in bodies:
                    visit(body, loop_order)
                return
            case lgc.Fuse(idx, body):
                visit(body, (*loop_order, idx))
                return
            case lgc.Produces():
                return
            case lgc.QueryInto(lgc.Table(), _, lgc.Literal()):
                pass
            case lgc.Query() | lgc.QueryInto():
                found.append(
                    price(stmt, loop_order) if touches(stmt, {alias}) else None
                )
            case _:
                raise ValueError(f"Unsupported statement type: {stmt}")
        machine(stmt)

    visit(prgm, ())
    return found


def optimize_protocols(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice] = PROTOCOL_COST_MODEL,
) -> lgc.LogicStatement:
    """prgm, reading each level with the protocol which costs least under
    cost_model, as choose_protocols chooses them."""
    return choose_protocols(prgm, bindings, stats, stats_factory, cost_model)[0]


class CostProtocolizer(FormattedForm, LogicLoader):
    """
    Pass a program in FormattedForm to ctx, after choosing the protocol which
    each table reads each level of its formatted tensor with, as
    optimize_protocols does under cost_model.
    """

    def __init__(
        self,
        ctx: LogicLoader,
        cost_model: CostModel[ProtocolChoice] = PROTOCOL_COST_MODEL,
    ):
        self.ctx = ctx
        self.cost_model = cost_model

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, TensorFType]]:
        prgm = optimize_protocols(prgm, bindings, stats, stats_factory, self.cost_model)
        return self.ctx(prgm, bindings, stats, stats_factory)
