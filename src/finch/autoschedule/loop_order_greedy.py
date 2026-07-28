from finch.finch_logic import Alias, Field, LogicExpression, StatsFactory, TensorStats

from .loop_order_cost import (
    get_conjunctive_and_disjunctive_inputs,
    get_loop_lookups,
)


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
