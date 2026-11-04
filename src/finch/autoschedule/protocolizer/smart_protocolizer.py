import itertools
from collections import OrderedDict
from collections.abc import Mapping

from finch import finch_logic as lgc
from finch.algebra import TensorFType
from finch.autoschedule.compiler import lower_protocol
from finch.autoschedule.cost_models import LOOP_COST_MODEL, CostModel, Prefix
from finch.autoschedule.stages import FormattedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import LogicLoader, StatsFactory, TensorStats
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite

# Walks come first, so that a tie keeps the protocol which generates the least
# code.
_READS = (lgc.Walk(), lgc.Gallop(), lgc.Follow())


def protocol_options(
    table: lgc.Table,
    prefix: tuple[lgc.Field, ...],
    bindings: Mapping[lgc.Alias, TensorFType],
) -> tuple[lgc.Protocol, ...]:
    """
    The protocols which table can read the last field of prefix with: those
    which the level of its tensor storing that field supports, once prefix
    reaches the level in storage order. A fused field, a tensor without a
    format, or a level which reads every protocol the same way is read by
    default.
    """
    match table:
        case lgc.Table(lgc.Alias() as tns, idxs, _) if tns.unfused in bindings:
            supported = bindings[tns.unfused].protocols()
            stored = idxs[tns.nfused :]
            idx = prefix[-1]
            if supported and idx in stored:
                depth = stored.index(idx)
                if set(stored[:depth]) <= set(prefix) and set(
                    stored[depth + 1 :]
                ).isdisjoint(prefix):
                    options = tuple(
                        proto
                        for proto in _READS
                        if lower_protocol(proto) in supported[depth]
                    )
                    if options:
                        return options
    return (lgc.DefaultProtocol(),)


def with_protocols(
    node: lgc.LogicNode,
    idx: lgc.Field,
    protocols: Mapping[lgc.Table, lgc.Protocol],
) -> lgc.LogicNode:
    """Read each table of protocols with its protocol for idx."""

    def rule(table):
        match table:
            case lgc.Table(tns, idxs, protos) if table in protocols:
                protos_2: list[lgc.Protocol] = [*protos] or [
                    lgc.DefaultProtocol()
                ] * len(idxs)
                protos_2[idxs.index(idx)] = protocols[table]
                return lgc.Table(tns, idxs, tuple(protos_2))
        return None

    return Rewrite(PostWalk(rule))(node)


def _with_rhs(query, rhs):
    match query:
        case lgc.Query(lhs, _):
            return lgc.Query(lhs, rhs)
        case lgc.QueryInto(lhs, op, _):
            return lgc.QueryInto(lhs, op, rhs)
    raise ValueError(f"Expected a query, got {query}")


def _expr(query) -> lgc.LogicExpression:
    """The right-hand side of query, as a Query, which the cost model prices."""
    match query:
        case lgc.Query(_, rhs):
            return rhs
        case lgc.QueryInto():
            return query.as_query().rhs
    raise ValueError(f"Expected a query, got {query}")


def cheapest_protocols(
    query: lgc.Query | lgc.QueryInto,
    loop_order: tuple[lgc.Field, ...],
    machine: StatsMachine,
    bindings: Mapping[lgc.Alias, TensorFType],
    cost_model: CostModel[Prefix],
    max_gallops: int | None,
):
    """
    Read the inputs of query with the protocols which cost least under
    cost_model, loop by loop of loop_order. A loop prefix charges only the
    protocols of its last field, so each loop chooses them for the tables which
    read its field independently, with at most max_gallops gallops.
    """
    lhs_idxs = query.lhs.idxs
    for n in range(1, len(loop_order) + 1):
        prefix = loop_order[:n]
        tables: dict[lgc.Table, None] = {}
        for node in PostOrderDFS(query.rhs):
            match node:
                case lgc.Table(_, idxs) if prefix[-1] in idxs:
                    tables[node] = None
        best_cost, best = float("inf"), {}
        options = (protocol_options(table, prefix, bindings) for table in tables)
        for protos in itertools.product(*options):
            if max_gallops is not None and protos.count(lgc.Gallop()) > max_gallops:
                continue
            protocols = dict(zip(tables, protos, strict=True))
            candidate = _with_rhs(
                query, with_protocols(query.rhs, prefix[-1], protocols)
            )
            component = Prefix(prefix, _expr(candidate), lhs_idxs)
            cost = cost_model.predict_component_cost(component, machine)
            if cost < best_cost:
                best_cost, best = cost, protocols
        query = _with_rhs(query, with_protocols(query.rhs, prefix[-1], best))
    return query


def smart_protocolize(
    prgm: lgc.LogicStatement,
    bindings: Mapping[lgc.Alias, TensorFType],
    stats: Mapping[lgc.Alias, TensorStats],
    stats_factory: StatsFactory,
    *,
    cost_model: CostModel[Prefix] = LOOP_COST_MODEL,
    max_gallops: int | None = 1,
) -> lgc.LogicStatement:
    """
    Read the formatted tensors of each query of prgm with the protocols which
    cost least under cost_model, given the statistics of the tensors before the
    query, as cheapest_protocols chooses them. A query's loop order is the
    fields of its enclosing Fuse nodes.
    """
    machine = StatsMachine(stats_factory, OrderedDict(stats))

    def visit(node, loop_order):
        match node:
            case lgc.Plan(bodies):
                return lgc.Plan(tuple(visit(body, loop_order) for body in bodies))
            case lgc.Fuse(idx, body):
                return lgc.Fuse(idx, visit(body, (*loop_order, idx)))
            case lgc.QueryInto(_, _, lgc.Literal()):
                machine(node)
            case lgc.Query() | lgc.QueryInto():
                new = cheapest_protocols(
                    node, loop_order, machine, bindings, cost_model, max_gallops
                )
                machine(node)
                return new
        return node

    return visit(prgm, ())


class SmartProtocolizer(FormattedForm, LogicLoader):
    """
    Pass a program in FormattedForm to ctx, after reading each formatted tensor
    with the protocols which cost least under cost_model, as smart_protocolize
    chooses them.
    """

    def __init__(
        self,
        ctx: LogicLoader,
        *,
        cost_model: CostModel[Prefix] = LOOP_COST_MODEL,
        max_gallops: int | None = 1,
    ):
        self.ctx = ctx
        self.cost_model = cost_model
        self.max_gallops = max_gallops

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, TensorFType]]:
        prgm = smart_protocolize(
            prgm,
            bindings,
            stats,
            stats_factory,
            cost_model=self.cost_model,
            max_gallops=self.max_gallops,
        )
        return self.ctx(prgm, bindings, stats, stats_factory)
