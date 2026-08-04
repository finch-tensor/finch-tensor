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

from .loop_order_cost import (
    cost_of_reformat,
    get_conjunctive_and_disjunctive_inputs,
    get_prefix_cost,
    loop_order_cost,
    needs_reformat,
)
from .loop_order_greedy import connected_loop_candidates, greedy_loop_order
from .loop_ordering import DefaultLoopOrderer


def loop_order_bfs(
    expr: LogicExpression,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    *,
    k: float = float("inf"),
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    cache: dict[object, TensorStats] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings.copy(), cache
    )
    if not all_vars:
        return ()

    best_order = greedy_loop_order(expr, stats_factory, stats_bindings)
    best_cost = loop_order_cost(expr, best_order, stats_factory, stats_bindings)
    prev_new_optimal_orders: list[tuple[tuple[Field, ...], float]] = [((), 0.0)]

    for _ in all_vars:
        new_optimal_orders: list[tuple[tuple[Field, ...], float]] = []
        for prefix, prefix_cost in prev_new_optimal_orders:
            remaining = set(all_vars) - set(prefix)
            candidates = connected_loop_candidates(
                prefix, remaining, conjunct_stats, disjunct_stats
            )
            for field in (field for field in all_vars if field in candidates):
                new_prefix = prefix + (field,)
                new_cost = prefix_cost + get_prefix_cost(
                    new_prefix,
                    conjunct_stats,
                    disjunct_stats,
                    stats_factory,
                )
                if new_cost >= best_cost:
                    continue
                if len(new_prefix) == len(all_vars):
                    # transpose cost
                    reformat_cost = 0.0
                    seen: list[TensorStats] = []
                    for stat in conjunct_stats + disjunct_stats:
                        if any(stat is other for other in seen):
                            continue
                        seen.append(stat)
                        if needs_reformat(stat, new_prefix):
                            reformat_cost += cost_of_reformat(stat)
                    total_cost = new_cost + reformat_cost
                    if total_cost < best_cost:
                        best_order, best_cost = new_prefix, total_cost
                else:
                    new_optimal_orders.append((new_prefix, new_cost))
        new_optimal_orders.sort(key=lambda state: state[1])
        if k != float("inf"):
            new_optimal_orders = new_optimal_orders[: int(k)]
        prev_new_optimal_orders = new_optimal_orders

    return best_order


def loop_order_dfs(
    expr: LogicExpression,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
) -> tuple[Field, ...]:
    all_vars = tuple(expr.fields())
    cache: dict[object, TensorStats] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings.copy(), cache
    )
    if not all_vars:
        return ()

    best_order = greedy_loop_order(expr, stats_factory, stats_bindings)
    best_cost = loop_order_cost(expr, best_order, stats_factory, stats_bindings)
    stack: list[tuple[tuple[Field, ...], float]] = [((), 0.0)]

    while stack:
        prefix, prefix_cost = stack.pop()
        remaining = set(all_vars) - set(prefix)
        children: list[tuple[tuple[Field, ...], float]] = []

        candidates = connected_loop_candidates(
            prefix, remaining, conjunct_stats, disjunct_stats
        )
        for field in (field for field in all_vars if field in candidates):
            new_prefix = prefix + (field,)
            new_cost = prefix_cost + get_prefix_cost(
                new_prefix,
                conjunct_stats,
                disjunct_stats,
                stats_factory,
            )
            if new_cost >= best_cost:
                continue
            if len(new_prefix) == len(all_vars):
                # transpose cost
                reformat_cost = 0.0
                seen: list[TensorStats] = []
                for stat in conjunct_stats + disjunct_stats:
                    if any(stat is other for other in seen):
                        continue
                    seen.append(stat)
                    if needs_reformat(stat, new_prefix):
                        reformat_cost += cost_of_reformat(stat)
                total_cost = new_cost + reformat_cost
                if total_cost < best_cost:
                    best_order, best_cost = new_prefix, total_cost
            else:
                children.append((new_prefix, new_cost))

        children.sort(key=lambda child: child[1], reverse=True)
        stack.extend(children)

    return best_order


class BFSLoopOrderer(DefaultLoopOrderer):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        k: float = float("inf"),
    ):
        super().__init__(ctx)
        self.k = k

    def _set_loop_order(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case Query(lhs, Aggregate(op, init, arg, idxs)):
                    idxs_2 = loop_order_bfs(
                        arg,
                        stats_factory,
                        stats,
                        k=self.k,
                    )
                    aggregate_2 = Aggregate(op, init, Reorder(arg, idxs_2), idxs)
                    new_queries.append(Query(lhs, aggregate_2))
                case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs)):
                    idxs_2 = loop_order_bfs(
                        arg,
                        stats_factory,
                        stats,
                        k=self.k,
                    )
                    reorder_2 = Reorder(
                        Aggregate(op, init, Reorder(arg, idxs_2), ag_idxs), idxs
                    )
                    new_queries.append(Query(lhs, reorder_2))
                case Query(_, Reorder(Table(Alias(), _), _)) as q:
                    new_queries.append(q)
                case _:
                    raise Exception(f"Invalid node: {query} in BFS loop ordering")
        return Plan(tuple(new_queries + [prgm.bodies[-1]]))


class DFSLoopOrderer(DefaultLoopOrderer):
    def _set_loop_order(
        self,
        prgm: Plan,
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> Plan:
        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case Query(lhs, Aggregate(op, init, arg, idxs)):
                    idxs_2 = loop_order_dfs(
                        arg,
                        stats_factory,
                        stats,
                    )
                    aggregate_2 = Aggregate(op, init, Reorder(arg, idxs_2), idxs)
                    new_queries.append(Query(lhs, aggregate_2))
                case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs)):
                    idxs_2 = loop_order_dfs(
                        arg,
                        stats_factory,
                        stats,
                    )
                    reorder_2 = Reorder(
                        Aggregate(op, init, Reorder(arg, idxs_2), ag_idxs), idxs
                    )
                    new_queries.append(Query(lhs, reorder_2))
                case Query(_, Reorder(Table(Alias(), _), _)) as q:
                    new_queries.append(q)
                case _:
                    raise Exception(f"Invalid node: {query} in DFS loop ordering")
        return Plan(tuple(new_queries + [prgm.bodies[-1]]))
