from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicExpression,
    Plan,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)

from .loop_order_cost import (
    get_conjunctive_and_disjunctive_inputs,
    get_loop_lookups,
)
from .loop_ordering import DefaultLoopOrderer


def connected_loop_candidates(
    prefix: tuple[Field, ...],
    remaining: set[Field],
    conjunct_stats: list[TensorStats],
    disjunct_stats: list[TensorStats],
) -> set[Field]:
    if not prefix:
        return set(remaining)

    prefix_set = set(prefix)
    connected: set[Field] = set()
    for stat in conjunct_stats + disjunct_stats:
        index_set = set(stat.index_order)
        if index_set & prefix_set:
            connected |= index_set

    candidates = connected & remaining
    return candidates if candidates else set(remaining)


def greedy_loop_order(
    expr: LogicExpression,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    if not all_vars:
        return ()

    stats_bindings_2 = stats_bindings.copy()
    cache: dict[object, TensorStats] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings_2, cache
    )

    prefix: list[Field] = []
    remaining = set(all_vars)

    while remaining:
        candidates = connected_loop_candidates(
            tuple(prefix), remaining, conjunct_stats, disjunct_stats
        )
        best = min(
            candidates,
            key=lambda field: get_loop_lookups(
                tuple(prefix) + (field,),
                conjunct_stats,
                disjunct_stats,
                stats_factory,
            ),
        )
        prefix.append(best)
        remaining.remove(best)

    return tuple(prefix)


# Same as set_loop_order in loop_ordering.py
def set_greedy_loop_order(
    plan: Plan,
    stats_factory: StatsFactory,
    stats: dict[Alias, TensorStats],
) -> Plan:
    new_queries = []
    for query in plan.bodies[:-1]:
        match query:
            case Query(lhs, Aggregate(op, init, arg, idxs)):
                idxs_2 = greedy_loop_order(arg, stats_factory, stats)
                aggregate_2 = Aggregate(op, init, Reorder(arg, idxs_2), idxs)
                new_queries.append(Query(lhs, aggregate_2))
            case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs)):
                idxs_2 = greedy_loop_order(arg, stats_factory, stats)
                reorder_2 = Reorder(
                    Aggregate(op, init, Reorder(arg, idxs_2), ag_idxs), idxs
                )
                new_queries.append(Query(lhs, reorder_2))
            case Query(_, Reorder(Table(Alias(), _), _)) as q:
                new_queries.append(q)
            case _:
                raise Exception(f"Invalid node: {query} in set_greedy_loop_order")

    return Plan(tuple(new_queries + [plan.bodies[-1]]))


class GreedyLoopOrderer(DefaultLoopOrderer):
    def _set_loop_order(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        return set_greedy_loop_order(prgm, stats_factory, stats)
