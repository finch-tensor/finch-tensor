import logging
from abc import abstractmethod
from copy import deepcopy
from functools import reduce
from itertools import chain as join_chains

from finch.algebra.tensor import TensorFType
from finch.algebra.utils import intersect, is_subsequence, setdiff, with_subsequence
from finch.autoschedule.factorizer.optimize import with_unique_lhs
from finch.autoschedule.stages import LogicLoopOrderer
from finch.autoschedule.util import (
    flatten_plans,
    fuse_loop_order,
    propagate_copy_queries,
    push_relabels,
)
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Fuse,
    HardAlias,
    LogicExpression,
    LogicLoader,
    LogicStatement,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import Namespace, PostOrderDFS, PostWalk, Rewrite
from finch.util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


def concordize(
    root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> LogicStatement:
    """
    Read every Table of a query in the loop order of its enclosing Fuse nodes.
    A Table stored in another order reads a transposed copy of its tensor
    instead, which is made after each statement that writes the tensor. Copies
    may read their argument in any order.
    """
    needed_swizzles: dict[Alias, dict[tuple[int, ...], Alias]] = {}
    namespace = Namespace(root)

    def swizzle(rhs: LogicExpression, loop_order: tuple[Field, ...]):
        def rule(ex):
            match ex:
                case Table(Alias(_) as var, idxs_1) if not is_subsequence(
                    intersect(idxs_1, loop_order), loop_order
                ):
                    idxs_subseq = with_subsequence(
                        intersect(loop_order, idxs_1), idxs_1
                    )
                    perm = tuple(idxs_1.index(idx) for idx in idxs_subseq)
                    return Table(
                        needed_swizzles.setdefault(var, {}).setdefault(
                            perm, HardAlias(namespace.freshen(var.name))
                        ),
                        idxs_subseq,
                    )

        return Rewrite(PostWalk(rule))(rhs)

    def concordant(stmt: LogicStatement, loop_order: tuple[Field, ...]):
        match stmt:
            case Plan(bodies):
                return Plan(tuple(concordant(body, loop_order) for body in bodies))
            case Fuse(idx, body):
                return Fuse(idx, concordant(body, (*loop_order, idx)))
            case Query(_, Table()):
                return stmt
            case Query(lhs, rhs):
                return Query(lhs, swizzle(rhs, loop_order))
            case QueryInto(lhs, op, rhs):
                return QueryInto(lhs, op, swizzle(rhs, loop_order))
        return stmt

    def _get_swizzle_queries(lhs: Alias) -> tuple[Query, ...]:
        ndims = len(next(iter(needed_swizzles[lhs].items()))[0])
        idxs = tuple([Field(f"i_{i}") for i in range(ndims)])
        return tuple(
            Query(Table(alias, tuple(idxs[p] for p in perm)), Table(lhs, idxs))
            for perm, alias in needed_swizzles[lhs].items()
        )

    assert isinstance(root, Plan)
    root = flatten_plans(root)
    match root:
        case Plan((*bodies, Produces() as prod)):
            bodies = [concordant(body, ()) for body in bodies]
            # Consider also aliases from input arguments.
            stmts: list[LogicStatement] = [
                query
                for alias in bindings
                if alias in needed_swizzles
                for query in _get_swizzle_queries(alias)
            ]
            for body in bodies:
                stmts.append(body)
                written = dict.fromkeys(
                    node.lhs.tns
                    for node in PostOrderDFS(body)
                    if isinstance(node, Query | QueryInto)
                )
                for lhs in written:
                    assert isinstance(lhs, Alias)
                    if lhs in needed_swizzles:
                        stmts.extend(_get_swizzle_queries(lhs))
            return Plan((*stmts, prod))
        case _:
            raise Exception(f"Invalid root: {root}")


def drop_internal_reorders(root: LogicStatement) -> LogicStatement:
    """
    Remove the Reorders within the right-hand side of each aggregate query and
    in-place update, whose fields are ordered by their left-hand tables.
    """

    def reorder_remover(ex):
        match ex:
            case Reorder(arg_2, _):
                return arg_2

    def rule(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, arg, idxs_2)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(lhs, Aggregate(op, init, arg_1, idxs_2))
            case QueryInto(lhs, op, arg):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return QueryInto(lhs, op, arg_1)

    return Rewrite(PostWalk(rule))(root)


def with_copy_loop_orders(root: Plan) -> Plan:
    """
    Set the loop order of each copy, which visits the fields of its left-hand
    table in order and then the fields it reduces. A copy may read its argument
    in another order, representing a transpose.
    """

    def loop_ordered(stmt: LogicStatement) -> LogicStatement:
        match stmt:
            case Query(Table(_, lhs_idxs), Table(_, idxs)):
                return fuse_loop_order(stmt, (*lhs_idxs, *setdiff(idxs, lhs_idxs)))
        return stmt

    return Plan(tuple(loop_ordered(body) for body in root.bodies))


class CycleInFields(Exception): ...


def toposort(chains: list[list[Field]]) -> tuple[Field, ...]:
    chains = deepcopy(chains)
    chains = [c for c in chains if len(c) > 0]
    parents = {chain[0]: 0 for chain in chains}
    for chain in chains:
        for f in chain[1:]:
            parents[f] = parents.get(f, 0) + 1
    roots = [f for f in parents if parents[f] == 0]
    perm = []
    while len(parents) > 0:
        if len(roots) == 0:
            raise CycleInFields("Cycle detected in fields' orders")
        perm.append(roots.pop())
        for chain in chains:
            if len(chain) > 0 and chain[0] == perm[-1]:
                chain.pop(0)
                if len(chain) > 0:
                    parents[chain[0]] -= 1
                    if parents[chain[0]] == 0:
                        roots.append(chain[0])
        parents.pop(perm[-1])
    return tuple(perm)


def _heuristic_loop_order(root: LogicExpression) -> tuple[Field, ...]:
    chains = []
    for node in PostOrderDFS(root):
        match node:
            case Table(_, idxs_1):
                chains.append(list(intersect(idxs_1, root.fields())))
    chains.extend([f] for f in root.fields())

    need_fix = False
    try:
        result = toposort(chains)
    except CycleInFields:
        logger.warning("Cycle in fields detected, need to permute.")
        need_fix = True
        result = root.fields()
    if need_fix or reduce(max, [len(c) for c in chains], 0) < len(
        set(join_chains(*chains))
    ):
        counts: dict[Field, int] = {}
        for chain in chains:
            for f in chain:
                counts[f] = counts.get(f, 0) + 1
        result = tuple(sorted(result, key=lambda x: counts[x], reverse=True))
    return result


def with_loop_order(
    stmt: LogicStatement, loop_order: tuple[Field, ...]
) -> LogicStatement:
    """
    Set the loop order of an aggregate query or in-place update by wrapping it
    in a Fuse node for each field of the loop order. A query can't change the
    layout of the table it writes, so its loop order is permuted to visit the
    fields of that table in order, and includes the fields which only that
    table has, such as output-only unit dimensions.
    """
    match stmt:
        case Query(Table(_, lhs_idxs), Aggregate()) | QueryInto(
            Table(_, lhs_idxs), _, _
        ):
            loop_order = with_subsequence(intersect(lhs_idxs, loop_order), loop_order)
            loop_order = toposort([list(loop_order), list(lhs_idxs)])
            return fuse_loop_order(stmt, loop_order)
        case _:
            raise ValueError(f"Expected an aggregate or in-place query, got {stmt}")


def heuristic_loop_order(plan: Plan) -> Plan:
    new_queries = []
    for query in plan.bodies[:-1]:

        def rule_1(query):
            match query:
                case Query(_, Aggregate(_, _, arg, _)) | QueryInto(_, _, arg):
                    return with_loop_order(query, _heuristic_loop_order(arg))
                case Query(_, Table(Alias(), _)) as q:
                    return q
                case _:
                    raise Exception(f"Invalid node: {query} in set_loop_order")

        new_queries.append(rule_1(query))
    return Plan(tuple(new_queries + [plan.bodies[-1]]))


class AbstractLoopOrderer(LogicLoopOrderer):
    def __init__(self, ctx: LogicLoader | None = None):
        if ctx is None:
            ctx = MockLogicLoader()
        self.ctx = ctx

    @abstractmethod
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        pass

    def lower(
        self,
        prgm: Plan,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        def loop_order_transform(prgm, bindings):
            prgm = drop_internal_reorders(prgm)
            assert isinstance(prgm, Plan)
            prgm = self.set_loop_orders(prgm, stats, stats_factory)
            prgm = push_relabels(prgm)
            assert isinstance(prgm, Plan)
            prgm = concordize(prgm, bindings)
            prgm = propagate_copy_queries(prgm, bindings)
            prgm = flatten_plans(prgm)
            return prgm, bindings

        stmt, bindings = with_unique_lhs(loop_order_transform, prgm, bindings)
        assert isinstance(stmt, Plan)
        # Copies include those which write renamed inputs back.
        stmt = with_copy_loop_orders(flatten_plans(stmt))
        return self.ctx(stmt, bindings, stats, stats_factory)


class DefaultLoopOrderer(AbstractLoopOrderer):
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        return heuristic_loop_order(prgm)
