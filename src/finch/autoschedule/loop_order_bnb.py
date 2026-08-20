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
    TensorStats,
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


def loop_order_bfs(
    expr: LogicExpression,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    output_vars: tuple[Field, ...] | None = None,
    *,
    k: int | None = None,
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    if not all_vars:
        return ()

    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings.copy(), {}
    )
    unique_stats = list({id(s): s for s in conjunct_stats + disjunct_stats}.values())

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
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    output_vars: tuple[Field, ...] | None = None,
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    if not all_vars:
        return ()

    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings.copy(), {}
    )
    unique_stats = list({id(s): s for s in conjunct_stats + disjunct_stats}.values())

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


def set_bfs_loop_order(
    plan: Plan,
    stats_factory: StatsFactory,
    stats: dict[Alias, TensorStats],
    *,
    k: int | None = None,
    output_fields: dict[Alias, tuple[Field, ...]] | None = None,
) -> Plan:
    if output_fields is None:
        output_fields = {}
    stats_bindings = dict(stats)
    cache: dict[object, TensorStats] = {}

    new_queries = []
    for query in plan.bodies[:-1]:
        match query:
            case Query(lhs, Aggregate(op, init, arg, idxs) as rhs):
                idxs_2 = loop_order_bfs(
                    arg, stats_factory, stats_bindings, rhs.fields(), k=k
                )
                output_idxs = output_fields.get(lhs, rhs.fields())
                aggregate_2 = Reorder(
                    Aggregate(op, init, Reorder(arg, idxs_2), idxs),
                    output_idxs,
                )
                new_queries.append(Query(lhs, aggregate_2))
            case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs) as rhs):
                idxs_2 = loop_order_bfs(
                    arg, stats_factory, stats_bindings, rhs.fields(), k=k
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
                raise Exception(f"Invalid node: {query} in set_bfs_loop_order")

        insert_statistics(
            stats_factory, query, stats_bindings, replace=False, cache=cache
        )

    return Plan(tuple(new_queries + [plan.bodies[-1]]))


def set_dfs_loop_order(
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
            case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs) as rhs):
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
                raise Exception(f"Invalid node: {query} in set_dfs_loop_order")

        insert_statistics(
            stats_factory, query, stats_bindings, replace=False, cache=cache
        )

    return Plan(tuple(new_queries + [plan.bodies[-1]]))


class BFSLoopOrderer(AbstractLoopOrderer):
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
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
        *,
        output_fields: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> Plan:
        return set_bfs_loop_order(
            prgm,
            stats_factory,
            stats,
            k=self.k,
            output_fields=output_fields,
        )


class DFSLoopOrderer(AbstractLoopOrderer):
    def set_loop_orders(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
        *,
        output_fields: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> Plan:
        return set_dfs_loop_order(
            prgm, stats_factory, stats, output_fields=output_fields
        )
