from __future__ import annotations

import itertools
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Generic, TypeVar

from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicExpression,
    LogicLoader,
    Plan,
    Query,
    Reorder,
    StatsFactory,
    Table,
)

from .galley.logical_optimizer import insert_statistics
from .loop_order_cost import (
    cost_of_reformat,
    get_conjunctive_and_disjunctive_inputs,
    get_prefix_cost,
    loop_order_cost,
    needs_reformat,
)
from .loop_order_greedy import connected_loop_candidates, greedy_loop_order
from .loop_ordering import AbstractLoopOrderer

from .tensor_stats import NumericStats, TensorStats

TS = TypeVar("TS", bound=TensorStats)
NS = TypeVar("NS", bound=NumericStats)


def loop_order_bfs(
    expr: LogicExpression,
    stats_factory: StatsFactory[NS],
    stats_bindings: Mapping[Alias, NS],
    output_vars: tuple[Field, ...] | None = None,
    *,
    k: int | None = None,
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    if not all_vars:
        return ()

    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, dict(stats_bindings), {}
    )
    unique_stats = list(
        {id(s): s for s in itertools.chain(conjunct_stats, disjunct_stats)}.values()
    )

    best_order = greedy_loop_order(expr, stats_factory, stats_bindings, output_vars)
    best_cost = loop_order_cost(
        expr, best_order, stats_factory, stats_bindings, output_vars
    )
    prev_new_optimal_orders: list[tuple[tuple[Field, ...], float]] = [((), 0.0)]

    for _ in all_vars:
        new_optimal_orders: list[tuple[tuple[Field, ...], float]] = []
        for prefix, prefix_cost in prev_new_optimal_orders:
            prefix_set = set(prefix)
            remaining = [field for field in all_vars if field not in prefix_set]
            # Preserves remaining / all_vars order (see connected_loop_candidates).
            candidates = connected_loop_candidates(
                prefix, remaining, conjunct_stats, disjunct_stats
            )
            for field in candidates:
                new_prefix = prefix + (field,)
                new_cost = prefix_cost + get_prefix_cost(
                    new_prefix,
                    conjunct_stats,
                    disjunct_stats,
                    stats_factory,
                    output_vars,
                )
                if new_cost >= best_cost:
                    continue
                if len(new_prefix) == len(all_vars):
                    reformat_cost = sum(
                        cost_of_reformat(stat)
                        for stat in unique_stats
                        if needs_reformat(stat, new_prefix)
                    )
                    total_cost = new_cost + reformat_cost
                    if total_cost < best_cost:
                        best_order, best_cost = new_prefix, total_cost
                else:
                    new_optimal_orders.append((new_prefix, new_cost))
        if k is not None:
            new_optimal_orders.sort(key=lambda state: state[1])
            new_optimal_orders = new_optimal_orders[:k]
        prev_new_optimal_orders = new_optimal_orders

    return best_order


def loop_order_dfs(
    expr: LogicExpression,
    stats_factory: StatsFactory[NS],
    stats_bindings: MutableMapping[Alias, NS],
    output_vars: tuple[Field, ...] | None = None,
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    if not all_vars:
        return ()

    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings, {}
    )
    unique_stats = list(
        {id(s): s for s in itertools.chain(conjunct_stats, disjunct_stats)}.values()
    )

    best_order = greedy_loop_order(expr, stats_factory, stats_bindings, output_vars)
    best_cost = loop_order_cost(
        expr, best_order, stats_factory, stats_bindings, output_vars
    )
    stack: list[tuple[tuple[Field, ...], float]] = [((), 0.0)]

    while stack:
        prefix, prefix_cost = stack.pop()
        prefix_set = set(prefix)
        remaining = [field for field in all_vars if field not in prefix_set]
        children: list[tuple[tuple[Field, ...], float]] = []

        # Preserves remaining / all_vars order (see connected_loop_candidates).
        candidates = connected_loop_candidates(
            prefix, remaining, conjunct_stats, disjunct_stats
        )
        for field in candidates:
            new_prefix = prefix + (field,)
            new_cost = prefix_cost + get_prefix_cost(
                new_prefix,
                conjunct_stats,
                disjunct_stats,
                stats_factory,
                output_vars,
            )
            if new_cost >= best_cost:
                continue
            if len(new_prefix) == len(all_vars):
                reformat_cost = sum(
                    cost_of_reformat(stat)
                    for stat in unique_stats
                    if needs_reformat(stat, new_prefix)
                )
                total_cost = new_cost + reformat_cost
                if total_cost < best_cost:
                    best_order, best_cost = new_prefix, total_cost
            else:
                children.append((new_prefix, new_cost))

        children.sort(key=lambda child: child[1], reverse=True)
        stack.extend(children)

    return best_order


def loop_order_brute_force(
    expr: LogicExpression,
    stats_factory: StatsFactory[NS],
    stats_bindings: Mapping[Alias, NS],
    output_vars: tuple[Field, ...] | None = None,
) -> tuple[Field, ...]:
    """Pick the loop order with the lowest ``loop_order_cost`` over all permutations.

    Ties keep the first permutation in ``expr.fields()`` order. Exponential in the
    number of indices; intended as an oracle for small nests and benchmarks.
    """
    all_vars = tuple(expr.fields())
    if len(all_vars) == 0:
        return ()

    best_order = all_vars
    best_cost = loop_order_cost(
        expr, best_order, stats_factory, stats_bindings, output_vars
    )
    for order in itertools.permutations(all_vars):
        cost = loop_order_cost(expr, order, stats_factory, stats_bindings, output_vars)
        if cost < best_cost:
            best_order, best_cost = order, cost
    return best_order


class BFSLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        k: int | None = None,
    ):
        super().__init__(ctx)
        self.k = k

    def set_loop_orders(
        self,
        prgm: Plan,
        stats: MutableMapping[Alias, NS],
        stats_factory: StatsFactory[NS],
        *,
        output_fields: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> Plan:
        if output_fields is None:
            output_fields = {}
        stats_bindings = dict(stats)
        cache: dict[object, NS] = {}

        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case Query(lhs, Aggregate(op, init, arg, idxs) as rhs):
                    idxs_2 = loop_order_bfs(
                        arg, stats_factory, stats_bindings, rhs.fields(), k=self.k
                    )
                    output_idxs = output_fields.get(lhs, rhs.fields())
                    aggregate_2 = Reorder(
                        Aggregate(op, init, Reorder(arg, idxs_2), idxs),
                        output_idxs,
                    )
                    new_queries.append(Query(lhs, aggregate_2))
                case Query(
                    lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs) as rhs
                ):
                    idxs_2 = loop_order_bfs(
                        arg, stats_factory, stats_bindings, rhs.fields(), k=self.k
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
                    raise Exception(f"Invalid node: {query} in BFSLoopOrderer")

            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache=cache
            )

        return Plan(tuple(new_queries + [prgm.bodies[-1]]))


class DFSLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: MutableMapping[Alias, NS],
        stats_factory: StatsFactory[NS],
        *,
        output_fields: MutableMapping[Alias, tuple[Field, ...]] | None = None,
    ) -> Plan:
        if output_fields is None:
            output_fields = {}
        stats_bindings = dict(stats)
        cache: dict[object, NS] = {}

        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case Query(lhs, Aggregate(op, init, arg, idxs) as rhs):
                    idxs_2 = loop_order_dfs(
                        arg, stats_factory, stats_bindings, rhs.fields()
                    )
                    output_idxs = output_fields.get(lhs, rhs.fields())
                    aggregate_2 = Reorder(
                        Aggregate(op, init, Reorder(arg, idxs_2), idxs),
                        output_idxs,
                    )
                    new_queries.append(Query(lhs, aggregate_2))
                case Query(
                    lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs) as rhs
                ):
                    idxs_2 = loop_order_dfs(
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
                    raise Exception(f"Invalid node: {query} in DFSLoopOrderer")

            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache=cache
            )

        return Plan(tuple(new_queries + [prgm.bodies[-1]]))


class BruteForceLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: MutableMapping[Alias, NS],
        stats_factory: StatsFactory[NS],
        *,
        output_fields: MutableMapping[Alias, tuple[Field, ...]] | None = None,
    ) -> Plan:
        if output_fields is None:
            output_fields = {}
        stats_bindings = dict(stats)
        cache: dict[object, NS] = {}

        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case Query(lhs, Aggregate(op, init, arg, idxs) as rhs):
                    idxs_2 = loop_order_brute_force(
                        arg, stats_factory, stats_bindings, rhs.fields()
                    )
                    output_idxs = output_fields.get(lhs, rhs.fields())
                    aggregate_2 = Reorder(
                        Aggregate(op, init, Reorder(arg, idxs_2), idxs),
                        output_idxs,
                    )
                    new_queries.append(Query(lhs, aggregate_2))
                case Query(
                    lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs) as rhs
                ):
                    idxs_2 = loop_order_brute_force(
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
                    raise Exception(f"Invalid node: {query} in BruteForceLoopOrderer")

            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache=cache
            )

        return Plan(tuple(new_queries + [prgm.bodies[-1]]))
