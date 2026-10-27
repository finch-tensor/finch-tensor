from __future__ import annotations

import itertools
from collections.abc import Mapping, MutableMapping
from typing import Generic, TypeVar

from finch.autoschedule.cost_models import LOOP_COST_MODEL, CostModel, Prefix
from finch.autoschedule.factorizer.galley_factorizer import insert_statistics
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicLoader,
    Plan,
    Query,
    QueryInto,
    StatsFactory,
    Table,
)

from .loop_order_greedy import connected_loop_candidates, greedy_loop_order
from .loop_ordering import AbstractLoopOrderer, with_loop_order

NS = TypeVar("NS", bound=NumericStats)


def loop_order_bfs(
    expr: LogicExpression,
    stats_factory: StatsFactory[NS],
    stats_bindings: Mapping[Alias, NS],
    output_vars: tuple[Field, ...] | None = None,
    *,
    k: int | None = None,
    cost_model: CostModel[Prefix] | None = None,
) -> tuple[Field, ...]:
    """Search prefixes with nonnegative incremental model costs as lower bounds."""
    if cost_model is None:
        cost_model = LOOP_COST_MODEL
    machine = StatsMachine(stats_factory, dict(stats_bindings))
    match expr:
        case Aggregate(arg=arg):
            pass
        case _:
            arg = expr
    fields = (*arg.fields(), *(expr.fields() if output_vars is None else output_vars))
    all_vars = tuple(dict.fromkeys(fields))
    if not all_vars:
        return ()

    best_order = greedy_loop_order(
        expr, stats_factory, stats_bindings, output_vars, cost_model=cost_model
    )
    best_cost = sum(
        cost_model.predict_component_cost(
            Prefix(best_order[:n], expr, output_vars), machine
        )
        for n in range(len(best_order) + 1)
    )
    initial_cost = cost_model.predict_component_cost(
        Prefix((), expr, output_vars), machine
    )
    prev_new_optimal_orders: list[tuple[tuple[Field, ...], float]] = [
        ((), initial_cost)
    ]

    for _ in all_vars:
        new_optimal_orders: list[tuple[tuple[Field, ...], float]] = []
        for prefix, prefix_cost in prev_new_optimal_orders:
            prefix_set = set(prefix)
            remaining = [field for field in all_vars if field not in prefix_set]
            # Preserves remaining / all_vars order (see connected_loop_candidates).
            candidates = connected_loop_candidates(prefix, remaining, output_vars)
            for field in candidates:
                new_prefix = prefix + (field,)
                new_cost = prefix_cost + cost_model.predict_component_cost(
                    Prefix(new_prefix, expr, output_vars), machine
                )
                if new_cost >= best_cost:
                    continue
                if len(new_prefix) == len(all_vars):
                    best_order, best_cost = new_prefix, new_cost
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
    *,
    cost_model: CostModel[Prefix] | None = None,
) -> tuple[Field, ...]:
    """Search prefixes with nonnegative incremental model costs as lower bounds."""
    if cost_model is None:
        cost_model = LOOP_COST_MODEL
    machine = StatsMachine(stats_factory, dict(stats_bindings))
    match expr:
        case Aggregate(arg=arg):
            pass
        case _:
            arg = expr
    fields = (*arg.fields(), *(expr.fields() if output_vars is None else output_vars))
    all_vars = tuple(dict.fromkeys(fields))
    if not all_vars:
        return ()

    best_order = greedy_loop_order(
        expr, stats_factory, stats_bindings, output_vars, cost_model=cost_model
    )
    best_cost = sum(
        cost_model.predict_component_cost(
            Prefix(best_order[:n], expr, output_vars), machine
        )
        for n in range(len(best_order) + 1)
    )
    initial_cost = cost_model.predict_component_cost(
        Prefix((), expr, output_vars), machine
    )
    stack: list[tuple[tuple[Field, ...], float]] = [((), initial_cost)]

    while stack:
        prefix, prefix_cost = stack.pop()
        prefix_set = set(prefix)
        remaining = [field for field in all_vars if field not in prefix_set]
        children: list[tuple[tuple[Field, ...], float]] = []

        # Preserves remaining / all_vars order (see connected_loop_candidates).
        candidates = connected_loop_candidates(prefix, remaining, output_vars)
        for field in candidates:
            new_prefix = prefix + (field,)
            new_cost = prefix_cost + cost_model.predict_component_cost(
                Prefix(new_prefix, expr, output_vars), machine
            )
            if new_cost >= best_cost:
                continue
            if len(new_prefix) == len(all_vars):
                best_order, best_cost = new_prefix, new_cost
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
    *,
    cost_model: CostModel[Prefix] | None = None,
) -> tuple[Field, ...]:
    """Pick the loop order with the lowest model cost over all permutations.

    Ties keep the first permutation in ``expr.fields()`` order. Exponential in the
    number of indices; intended as an oracle for small nests and benchmarks.
    """
    if cost_model is None:
        cost_model = LOOP_COST_MODEL
    machine = StatsMachine(stats_factory, dict(stats_bindings))
    match expr:
        case Aggregate(arg=arg):
            pass
        case _:
            arg = expr
    fields = (*arg.fields(), *(expr.fields() if output_vars is None else output_vars))
    all_vars = tuple(dict.fromkeys(fields))
    best_order = ()
    best_cost = float("inf")
    for order in itertools.permutations(all_vars):
        if (
            output_vars is not None
            and tuple(idx for idx in order if idx in output_vars) != output_vars
        ):
            continue
        cost = sum(
            cost_model.predict_component_cost(
                Prefix(order[:n], expr, output_vars), machine
            )
            for n in range(len(order) + 1)
        )
        if not best_order or cost < best_cost:
            best_order, best_cost = order, cost
    return best_order


class BFSLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        k: int | None = None,
        cost_model: CostModel[Prefix] | None = None,
    ):
        super().__init__(ctx)
        self.cost_model = cost_model if cost_model is not None else LOOP_COST_MODEL
        self.k = k

    def set_loop_orders(
        self,
        prgm: Plan,
        stats: MutableMapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> Plan:
        stats_bindings = dict(stats)

        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case QueryInto(_, _, Literal()):
                    new_queries.append(query)
                case Query(Table(_, out_idxs), Aggregate() as arg) | QueryInto(
                    Table(_, out_idxs), _, arg
                ):
                    match query:
                        case QueryInto():
                            arg = query.as_query().rhs
                    idxs_2 = loop_order_bfs(
                        arg,
                        stats_factory,
                        stats_bindings,
                        out_idxs,
                        k=self.k,
                        cost_model=self.cost_model,
                    )
                    new_queries.append(with_loop_order(query, idxs_2))
                case Query(_, Table(Alias(), _)) as q:
                    new_queries.append(q)
                case _:
                    raise Exception(f"Invalid node: {query} in BFSLoopOrderer")

            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache={}
            )

        return Plan(tuple(new_queries + [prgm.bodies[-1]]))


class DFSLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        cost_model: CostModel[Prefix] | None = None,
    ):
        super().__init__(ctx)
        self.cost_model = cost_model if cost_model is not None else LOOP_COST_MODEL

    def set_loop_orders(
        self,
        prgm: Plan,
        stats: MutableMapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> Plan:
        stats_bindings = dict(stats)

        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case QueryInto(_, _, Literal()):
                    new_queries.append(query)
                case Query(Table(_, out_idxs), Aggregate() as arg) | QueryInto(
                    Table(_, out_idxs), _, arg
                ):
                    match query:
                        case QueryInto():
                            arg = query.as_query().rhs
                    idxs_2 = loop_order_dfs(
                        arg,
                        stats_factory,
                        stats_bindings,
                        out_idxs,
                        cost_model=self.cost_model,
                    )
                    new_queries.append(with_loop_order(query, idxs_2))
                case Query(_, Table(Alias(), _)) as q:
                    new_queries.append(q)
                case _:
                    raise Exception(f"Invalid node: {query} in DFSLoopOrderer")

            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache={}
            )

        return Plan(tuple(new_queries + [prgm.bodies[-1]]))


class BruteForceLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        cost_model: CostModel[Prefix] | None = None,
    ):
        super().__init__(ctx)
        self.cost_model = cost_model if cost_model is not None else LOOP_COST_MODEL

    def set_loop_orders(
        self,
        prgm: Plan,
        stats: MutableMapping[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> Plan:
        stats_bindings = dict(stats)

        new_queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case QueryInto(_, _, Literal()):
                    new_queries.append(query)
                case Query(Table(_, out_idxs), Aggregate() as arg) | QueryInto(
                    Table(_, out_idxs), _, arg
                ):
                    match query:
                        case QueryInto():
                            arg = query.as_query().rhs
                    idxs_2 = loop_order_brute_force(
                        arg,
                        stats_factory,
                        stats_bindings,
                        out_idxs,
                        cost_model=self.cost_model,
                    )
                    new_queries.append(with_loop_order(query, idxs_2))
                case Query(_, Table(Alias(), _)) as q:
                    new_queries.append(q)
                case _:
                    raise Exception(f"Invalid node: {query} in BruteForceLoopOrderer")

            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache={}
            )

        return Plan(tuple(new_queries + [prgm.bodies[-1]]))
