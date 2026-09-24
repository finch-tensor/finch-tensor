import logging
from abc import abstractmethod
from copy import deepcopy
from functools import reduce
from itertools import chain as join_chains

from finch.algebra.tensor import TensorFType
from finch.algebra.utils import intersect, is_subsequence, with_subsequence
from finch.autoschedule.factorizer.optimize import with_unique_lhs
from finch.autoschedule.stages import LogicLoopOrderer
from finch.autoschedule.util import flatten_plans, propagate_copy_queries, push_fields
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
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
    needed_swizzles: dict[Alias, dict[tuple[int, ...], Alias]] = {}
    namespace = Namespace(root)

    def rule_0(ex):
        match ex:
            case Reorder(Table(Alias(_) as var, idxs_1), idxs_2):
                if not is_subsequence(intersect(idxs_1, idxs_2), idxs_2):
                    idxs_subseq = with_subsequence(intersect(idxs_2, idxs_1), idxs_1)
                    perm = tuple(idxs_1.index(idx) for idx in idxs_subseq)
                    return Reorder(
                        Table(
                            needed_swizzles.setdefault(var, {}).setdefault(
                                perm, Alias(namespace.freshen(var.name))
                            ),
                            idxs_subseq,
                        ),
                        idxs_2,
                    )
                return None

    def _get_swizzle_queries(lhs: Alias) -> tuple[Query, ...]:
        ndims = len(next(iter(needed_swizzles[lhs].items()))[0])
        idxs = tuple([Field(f"i_{i}") for i in range(ndims)])
        return tuple(
            Query(Table(alias, tuple(idxs[p] for p in perm)), Table(lhs, idxs))
            for perm, alias in needed_swizzles[lhs].items()
        )

    def rule_1(ex):
        match ex:
            case (
                Query(Table(Alias() as lhs, _), _)
                | QueryInto(Table(Alias() as lhs, _), _, _)
            ) as q if lhs in needed_swizzles:
                swizzle_queries = _get_swizzle_queries(lhs)
                return Plan((q, *swizzle_queries))

    assert isinstance(root, Plan)
    root = flatten_plans(root)
    match root:
        case Plan(bodies) if isinstance(bodies[-1], Produces):
            prod = bodies[-1]
            root = Plan(bodies[:-1])
            root = Rewrite(PostWalk(rule_0))(root)
            root = Rewrite(PostWalk(rule_1))(root)
            # Consider also aliases from input arguments.
            for alias in bindings:
                if alias in needed_swizzles:
                    swizzle_queries = _get_swizzle_queries(alias)
                    root = Plan((*swizzle_queries, root))
            return flatten_plans(Plan((root, prod)))
        case _:
            raise Exception(f"Invalid root: {root}")


def drop_internal_reorders(
    root: LogicStatement, keep_loop_orders: bool
) -> LogicStatement:
    def reorder_remover(ex):
        match ex:
            case Reorder(arg_2, _):
                return arg_2

    def rule_1(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, arg, idxs_2)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(lhs, Aggregate(op, init, arg_1, idxs_2))
            case QueryInto(lhs, op, Aggregate(op1, init, arg, ag_idxs)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return QueryInto(lhs, op, Aggregate(op1, init, arg_1, ag_idxs))
            case QueryInto(lhs, op, arg):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return QueryInto(lhs, op, arg_1)

    def rule_2(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, Reorder(arg, idxs_1), idxs_2)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(lhs, Aggregate(op, init, Reorder(arg_1, idxs_1), idxs_2))
            case QueryInto(
                lhs, op, Aggregate(op1, init, Reorder(arg, idxs_1), ag_idxs)
            ):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return QueryInto(
                    lhs, op, Aggregate(op1, init, Reorder(arg_1, idxs_1), ag_idxs)
                )
            case QueryInto(lhs, op, Reorder(arg, idxs_1)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return QueryInto(lhs, op, Reorder(arg_1, idxs_1))

    if keep_loop_orders:
        return Rewrite(PostWalk(rule_2))(root)
    return Rewrite(PostWalk(rule_1))(root)


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
    Set the loop order of a query. An aggregate query holds its loop order in
    a Reorder of the aggregate's argument, and a pointwise in-place update
    holds it in a Reorder of its right-hand side.
    """
    match stmt:
        case Query(lhs, Aggregate(op, init, arg, idxs)):
            return Query(lhs, Aggregate(op, init, Reorder(arg, loop_order), idxs))
        case QueryInto(lhs, update_op, Aggregate(op, init, arg, idxs)):
            return QueryInto(
                lhs, update_op, Aggregate(op, init, Reorder(arg, loop_order), idxs)
            )
        case QueryInto(lhs, op, arg):
            return QueryInto(lhs, op, Reorder(arg, loop_order))
        case _:
            raise ValueError(f"Expected an aggregate or in-place query, got {stmt}")


def heuristic_loop_order(plan: Plan) -> Plan:
    new_queries = []
    for query in plan.bodies[:-1]:

        def rule_1(query):
            match query:
                case Query(_, Aggregate(_, _, arg, _)) | QueryInto(
                    _, _, Aggregate(_, _, arg, _)
                ):
                    return with_loop_order(query, _heuristic_loop_order(arg))
                case QueryInto(Table(_, idxs), _, _):
                    # A pointwise update loops in the order of the table it
                    # updates.
                    return with_loop_order(query, idxs)
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
            prgm = drop_internal_reorders(prgm, keep_loop_orders=False)
            assert isinstance(prgm, Plan)
            prgm = self.set_loop_orders(prgm, stats, stats_factory)
            prgm = push_fields(prgm)
            assert isinstance(prgm, Plan)
            prgm = concordize(prgm, bindings)
            prgm = drop_internal_reorders(prgm, keep_loop_orders=True)
            prgm = propagate_copy_queries(prgm, bindings)
            prgm = flatten_plans(prgm)
            return prgm, bindings

        stmt, bindings = with_unique_lhs(loop_order_transform, prgm, bindings)
        assert isinstance(stmt, Plan)
        stmt = flatten_plans(stmt)
        return self.ctx(stmt, bindings, stats, stats_factory)


class DefaultLoopOrderer(AbstractLoopOrderer):
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        return heuristic_loop_order(prgm)
