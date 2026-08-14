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

from .galley.logical_optimizer import insert_statistics
from .loop_order_cost import (
    cost_of_reformat,
    get_conjunctive_and_disjunctive_inputs,
    get_prefix_cost,
    get_reformat_set,
)
from .loop_ordering import AbstractLoopOrderer


def connected_loop_candidates(
    prefix: tuple[Field, ...],
    remaining: list[Field],
    conjunct_stats: list[TensorStats],
    disjunct_stats: list[TensorStats],
) -> list[Field]:
    """Fields in ``remaining`` that share a tensor with ``prefix``.

    ``remaining`` is kept ordered, and the result preserves that order, so that
    ties in the greedy cost comparison always break the same way. Iterating a
    set here would make the chosen loop order depend on ``Field`` hash values,
    which vary between processes.
    """
    if not prefix:
        return list(remaining)

    prefix_set = set(prefix)
    connected: set[Field] = set()
    for stat in conjunct_stats + disjunct_stats:
        index_set = set(stat.index_order)
        if index_set & prefix_set:
            connected |= index_set

    candidates = [field for field in remaining if field in connected]
    return candidates if candidates else list(remaining)


def transpose_penalty(
    input_stats: list[TensorStats],
    prefix: tuple[Field, ...],
    charged: frozenset[int],
) -> float:
    """Cost of the reformats that ``prefix`` newly forces.

    ``needs_reformat`` is monotone in the prefix: an index placed before one
    that precedes it in a tensor's storage order already forces that tensor to
    be reformatted, and extending the prefix cannot undo it. So each tensor is
    charged exactly once, at the step where its reformat becomes unavoidable,
    and the total over a full order matches ``loop_order_cost``.
    """
    return sum(
        cost_of_reformat(input_stats[i])
        for i in get_reformat_set(input_stats, prefix) - charged
    )


def greedy_loop_order(
    expr: LogicExpression,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    output_vars: tuple[Field, ...] | None = None,
) -> tuple[Field, ...]:
    """Build a loop order one index at a time, appending the cheapest candidate.

    Each step is scored with the same terms ``loop_order_cost`` charges for that
    prefix -- ``get_prefix_cost`` for the reads and writes, plus the reformats
    the new index forces -- so the summed score of the returned order equals its
    ``loop_order_cost``.
    """
    all_vars = tuple(dict.fromkeys(expr.fields()))
    if not all_vars:
        return ()

    stats_bindings_2 = stats_bindings.copy()
    cache: dict[object, TensorStats] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings_2, cache
    )

    # Deduplicated by identity, matching how loop_order_cost charges reformats.
    input_stats: list[TensorStats] = []
    for stat in conjunct_stats + disjunct_stats:
        if not any(stat is seen for seen in input_stats):
            input_stats.append(stat)

    prefix: list[Field] = []
    remaining = list(all_vars)
    charged: frozenset[int] = frozenset()

    def candidate_cost(field: Field) -> float:
        new_prefix = tuple(prefix) + (field,)
        prefix_cost = get_prefix_cost(
            new_prefix, conjunct_stats, disjunct_stats, stats_factory, output_vars
        )
        return prefix_cost + transpose_penalty(input_stats, new_prefix, charged)

    while remaining:
        candidates = connected_loop_candidates(
            tuple(prefix), remaining, conjunct_stats, disjunct_stats
        )
        best = min(candidates, key=candidate_cost)
        prefix.append(best)
        remaining.remove(best)
        charged = get_reformat_set(input_stats, tuple(prefix))

    return tuple(prefix)


def set_greedy_loop_order(
    plan: Plan,
    stats_factory: StatsFactory,
    stats: dict[Alias, TensorStats],
    *,
    output_fields: dict[Alias, tuple[Field, ...]] | None = None,
) -> Plan:
    if output_fields is None:
        output_fields = {}
    stats_bindings = dict(stats)
    cache: dict[object, TensorStats] = {}

    new_queries = []
    for query in plan.bodies[:-1]:
        # The query's result layout decides whether writing an index is
        # sequential or random, so pass it to the cost model.
        match query:
            case Query(lhs, Aggregate(op, init, arg, idxs) as rhs):
                idxs_2 = greedy_loop_order(
                    arg, stats_factory, stats_bindings, rhs.fields()
                )
                output_idxs = output_fields.get(lhs, rhs.fields())
                aggregate_2 = Reorder(
                    Aggregate(op, init, Reorder(arg, idxs_2), idxs),
                    output_idxs,
                )
                new_queries.append(Query(lhs, aggregate_2))
            case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs) as rhs):
                idxs_2 = greedy_loop_order(
                    arg, stats_factory, stats_bindings, rhs.fields()
                )
                output_idxs = output_fields.get(lhs, rhs.fields())
                reorder_2 = Reorder(
                    Aggregate(op, init, Reorder(arg, idxs_2), ag_idxs),
                    output_idxs,
                )
                new_queries.append(Query(lhs, reorder_2))
            case Query(_, Reorder(Table(Alias(), _), _)) as q:
                new_queries.append(q)
            case _:
                raise Exception(f"Invalid node: {query} in set_greedy_loop_order")

        insert_statistics(
            stats_factory, query, stats_bindings, replace=False, cache=cache
        )

    return Plan(tuple(new_queries + [plan.bodies[-1]]))


class GreedyLoopOrderer(AbstractLoopOrderer):
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
        *,
        output_fields: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> Plan:
        return set_greedy_loop_order(
            prgm, stats_factory, stats, output_fields=output_fields
        )
