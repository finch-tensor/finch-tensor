from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import product

from finch import finch_logic as lgc
from finch.algebra import TensorFType
from finch.autoschedule.compiler import lower_protocol
from finch.autoschedule.cost_models import (
    PROTOCOL_COST_MODEL,
    CostModel,
    Prefix,
    ProtocolChoice,
    level_accesses,
    level_formats,
    level_reads,
    read_options,
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


def choose_protocols(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    cost_model: CostModel[ProtocolChoice] = PROTOCOL_COST_MODEL,
    only: set[lgc.Alias] | None = None,
    costs: Sequence[float] | None = None,
) -> tuple[lgc.LogicStatement, list[float]]:
    """
    Read each level of each formatted tensor of bindings with the protocol
    which costs least under cost_model, and return the program with the cost
    of each of its queries, in program order. The reads at a loop are chosen
    together, over the product of the protocols each supports, by the cost of
    the accesses at that loop alone, since the protocols at a loop don't
    change where the loops within it are entered. A read which gives its
    protocol keeps it. With only, the queries which touch none of its aliases
    keep their protocols and take their costs from costs, those of every
    query in order.
    """
    formats = level_formats(bindings)
    given = set()
    for node in PostOrderDFS(prgm):
        match node:
            case lgc.Table(tns, idxs, protocols):
                given.update(
                    (tns, idxs, pos)
                    for pos, proto in enumerate(protocols)
                    if proto != lgc.DefaultProtocol()
                )
    prgm = protocolize(prgm, bindings)
    machine = StatsMachine(stats_factory, OrderedDict(stats))
    found: list[float] = []

    def choose(stmt: lgc.Query | lgc.QueryInto, loop_order: tuple[lgc.Field, ...]):
        """stmt, reading each level with the protocol which costs least, and
        its cost."""
        match stmt:
            case lgc.Query(lgc.Table(_, lhs_idxs) as lhs, rhs):
                rebuild = partial(lgc.Query, lhs)
            case lgc.QueryInto(lgc.Table(_, lhs_idxs) as lhs, op, rhs):
                rebuild = partial(lgc.QueryInto, lhs, op)

        def expr(rhs: lgc.LogicExpression) -> lgc.LogicExpression:
            # An update reduces into its table, as its aggregate query says.
            match rebuild(rhs):
                case lgc.QueryInto() as update:
                    return update.as_query().rhs
            return rhs

        cost = 0.0
        for n in range(len(loop_order)):
            loops = loop_order[: n + 1]
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
                for read in level_reads(Prefix(loops, expr(rhs), lhs_idxs), formats)
                if (read.alias, read.table.idxs, read.alias.nfused + read.depth)
                not in given
            ]
            best: tuple[float, lgc.LogicExpression] = (float("inf"), rhs)
            for choice in product(*options):
                rhs_2 = rhs
                for tns, idxs, depth, proto in choice:
                    rhs_2 = with_protocol(rhs_2, tns, idxs, depth, proto)
                prefix = Prefix(loops, expr(rhs_2), lhs_idxs, formats, output=lhs.tns)
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
                if only is None or touches(stmt, only):
                    stmt, cost = choose(stmt, loop_order)
                elif costs is not None:
                    cost = costs[len(found)]
                else:
                    raise ValueError("Expected the costs of the queries not chosen")
                found.append(cost)
            case _:
                raise ValueError(f"Unsupported statement type: {stmt}")
        machine(stmt)
        return stmt

    return visit(prgm, ()), found


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
