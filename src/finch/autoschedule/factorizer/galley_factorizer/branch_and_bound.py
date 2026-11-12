"""
Branch-and-bound optimizer for Galley query planning.
Converted from Julia.
Removed general cache. Both alias_hash and cost_cache
"""

import heapq
from collections import OrderedDict
from typing import Literal

from finch.finch_logic import Query

from .annotated_query import AnnotatedQuery

GalleyOptimizer = Literal["greedy", "bfs", "dfs", "best-first"]

# A component of this many indices or more takes the greedy order rather than
# an exact search, whose cost grows exponentially with it, as Galley.jl's
# pruned optimizer does.
GREEDY_COMPONENT_SIZE = 10

# The relative slack on the bounds max_subquery_costs gives. The exact search
# can reach the greedy order's cost by adding the same costs in another order,
# which rounds differently; without slack such a sum can exceed its own bound by
# an ulp and leave no complete order.
BOUND_TOLERANCE = 1e-9


def branch_and_bound(
    input_aq: AnnotatedQuery,
    component: list,
    k: float,
    max_subquery_costs: OrderedDict,
) -> tuple:
    """
    Branch-and-bound search for optimal reduction order within a component.

    Returns ``((order, queries, aq, cost), optimal_subquery_costs)``.

    Raises
    ------
    RuntimeError
        If no complete elimination order exists for ``component``.
    """
    # --- Initialize ---
    # optimal_orders[vars_key] = (order, queries, aq, cost) for each state.
    # vars_key = frozenset of indices already reduced. Start with empty set.
    optimal_orders: OrderedDict[frozenset, tuple] = OrderedDict(
        [(frozenset(), ([], [], input_aq, 0.0))]
    )
    prev_new_optimal_orders = optimal_orders

    # --- Main loop: one iteration per reduction step ---
    for _ in range(len(component)):
        # best_idx_ext[new_vars] = (aq, idx, old_order, old_queries, total_cost)
        # Tracks best way to reach each new state in this step.
        best_idx_ext: OrderedDict[frozenset, tuple] = OrderedDict()

        # --- Extend each current state by trying every possible next reduction ---
        for vars_key, pc in prev_new_optimal_orders.items():
            old_order, old_queries, aq, prev_cost = pc
            reducible_in_comp = aq.get_reducible_idxs_for_component(component)
            for idx in reducible_in_comp:
                # Cost of this reduction and which vars it eliminates
                _, _, _, reduced_vars = aq.get_reduce_query(idx)
                cost = aq.get_cost_of_reduce_idx(idx)
                total_cost = cost + prev_cost
                new_vars = vars_key | frozenset(reduced_vars)

                # --- Pruning bound: if we have a cheaper cost for a superset
                # (vars2 >= new_vars), that gives an upper bound; we can prune
                # if our total_cost exceeds it.
                bound = float("inf")
                for vars2 in max_subquery_costs:
                    if vars2 >= new_vars:
                        bound = min(bound, max_subquery_costs[vars2])
                bound *= 1 + BOUND_TOLERANCE

                # cheapest_cost = best known cost to reach new_vars (from any path)
                best_val = best_idx_ext.get(
                    new_vars, (None, None, None, None, float("inf"))
                )
                cheapest_cost = min(
                    best_val[4],
                    optimal_orders.get(new_vars, (None, None, None, float("inf")))[3],
                    bound,
                )
                # Keep this candidate only if it is at least as good as cheapest
                if total_cost <= cheapest_cost:
                    best_idx_ext[new_vars] = (
                        aq,
                        idx,
                        old_order,
                        old_queries,
                        total_cost,
                    )

        if len(best_idx_ext) == 0:
            break

        # --- Keep only top-k by cost (greedy k=1 keeps single best per state) ---
        if k == float("inf"):
            num_to_keep = len(best_idx_ext)
        else:
            num_to_keep = int(min(k, len(best_idx_ext)))

        sorted_items = sorted(best_idx_ext.items(), key=lambda x: x[1][4])
        top_k_idx_ext = OrderedDict(sorted_items[:num_to_keep])

        # --- Apply reductions and build next round of states ---
        new_optimal_orders = OrderedDict()
        for new_vars, idx_ext_info in top_k_idx_ext.items():
            aq, idx, old_order, old_queries, cost = idx_ext_info
            new_aq = aq.copy()
            reduce_query = new_aq.reduce_idx(idx)
            new_queries = list(old_queries) + [reduce_query]
            new_order = list(old_order) + [idx]
            new_optimal_orders[new_vars] = (new_order, new_queries, new_aq, cost)

        optimal_orders.update(new_optimal_orders)
        prev_new_optimal_orders = new_optimal_orders

    # --- Build subquery cost map for pruning (only when k=1 greedy) ---
    optimal_subquery_costs = OrderedDict()
    if k == 1:
        for vars_key in optimal_orders:
            optimal_subquery_costs[vars_key] = optimal_orders[vars_key][3]

    # --- Return result if we found a full order for the component ---
    component_set = frozenset(component)
    if component_set not in optimal_orders:
        raise RuntimeError(
            "branch_and_bound: no complete reduction order for component "
            f"{component!r}; reducible idxs on input: "
            f"{input_aq.get_reducible_idxs()!r}"
        )
    return (optimal_orders[component_set], optimal_subquery_costs)


def branch_and_bound_best_first(
    input_aq: AnnotatedQuery,
    component: list,
) -> tuple:
    """
    Exact branch-and-bound, expanding the cheapest reached state first.

    States are sets of reduced indices. Step costs are non-negative, so the
    first time a set is popped it was reached by its cheapest route: each set
    is expanded at most once, and the search stops when the full component
    pops. A child's query is only copied and reduced when it is popped.

    Returns ``((order, queries, aq, cost), optimal_subquery_costs)`` where the
    second element is an empty ``OrderedDict``.

    Raises
    ------
    RuntimeError
        If no complete elimination order exists for ``component``.
    """
    component_set = frozenset(component)
    best_cost: dict[frozenset, float] = {frozenset(): 0.0}
    expanded: set[frozenset] = set()
    tie = 0
    heap: list[tuple] = [(0.0, tie, frozenset(), [], [], input_aq, None)]

    while heap:
        cost, _, vars_key, order, queries, aq, idx = heapq.heappop(heap)
        if vars_key in expanded or cost > best_cost[vars_key]:
            continue
        if idx is not None:
            aq = aq.copy()
            queries = [*queries, aq.reduce_idx(idx)]
            order = [*order, idx]
        expanded.add(vars_key)

        if vars_key == component_set:
            return ((order, queries, aq, cost), OrderedDict())

        for next_idx in aq.get_reducible_idxs_for_component(component):
            _, _, _, reduced_vars = aq.get_reduce_query(next_idx)
            new_vars = vars_key | frozenset(reduced_vars)
            if new_vars in expanded:
                continue
            new_cost = cost + aq.get_cost_of_reduce_idx(next_idx)

            if not new_cost < best_cost.get(new_vars, float("inf")):
                continue
            best_cost[new_vars] = new_cost
            tie += 1
            heapq.heappush(
                heap, (new_cost, tie, new_vars, order, queries, aq, next_idx)
            )

    raise RuntimeError(
        "branch_and_bound_best_first: no complete reduction order for component "
        f"{component!r}; reducible idxs on input: "
        f"{input_aq.get_reducible_idxs()!r}"
    )


def branch_and_bound_dfs(
    input_aq: AnnotatedQuery,
    component: list,
) -> tuple:
    """
    Raises
    ------
    RuntimeError
        If no complete elimination order exists for ``component``.
    """
    component_set = frozenset(component)
    best_cost: dict[frozenset, float] = {frozenset(): 0.0}
    best_complete = float("inf")
    best_state: tuple[list, list, AnnotatedQuery, float] | None = None
    tie = 0
    heap: list[tuple] = [(0, 0.0, tie, frozenset(), [], [], input_aq, None)]

    while heap:
        _, cost, _, vars_key, order, queries, aq, idx = heapq.heappop(heap)
        if cost > best_cost[vars_key] or not cost < best_complete:
            continue
        if idx is not None:
            aq = aq.copy()
            queries = [*queries, aq.reduce_idx(idx)]
            order = [*order, idx]

        if vars_key == component_set:
            best_complete = cost
            best_state = (order, queries, aq, cost)
            continue

        for next_idx in aq.get_reducible_idxs_for_component(component):
            _, _, _, reduced_vars = aq.get_reduce_query(next_idx)
            new_vars = vars_key | frozenset(reduced_vars)
            new_cost = cost + aq.get_cost_of_reduce_idx(next_idx)
            limit = min(best_cost.get(new_vars, float("inf")), best_complete)
            if not new_cost < limit:
                continue
            best_cost[new_vars] = new_cost
            tie += 1
            heapq.heappush(
                heap,
                (-len(new_vars), new_cost, tie, new_vars, order, queries, aq, next_idx),
            )

    if best_state is None:
        raise RuntimeError(
            "branch_and_bound_dfs: no complete reduction order for component "
            f"{component!r}; reducible idxs on input: "
            f"{input_aq.get_reducible_idxs()!r}"
        )
    return (best_state, OrderedDict())


def pruned_query_to_plan(
    input_aq: AnnotatedQuery,
    use_components: bool = True,
    *,
    optimizer: GalleyOptimizer = "dfs",
) -> tuple[list[Query], float]:
    """
    Pruned Galley optimizer.

    * ``"greedy"`` — ``k=1`` (also used automatically for a component of at
      least ``GREEDY_COMPONENT_SIZE`` indices).
    * ``"bfs"`` — greedy ``k=1`` for bounds, then BFS BnB.
    * ``"dfs"`` — depth-first BnB (deepest state first, cheapest among equals).
    * ``"best-first"`` — best-first (uniform-cost) BnB.

    Returns ``(queries, total_cost)``.
    """
    total_cost = 0.0
    elimination_order: list = []
    queries: list[Query] = []
    cur_aq = input_aq.copy()

    # --- Process components until no more reducible indices ---
    # Do this instead of for comp in components because components
    # are recomputed in reduce_idx same as julia code
    while cur_aq.get_reducible_idxs():
        component = cur_aq.connected_components[0]

        if not use_components:
            component = list(
                set().union(*(set(c) for c in cur_aq.connected_components))
            )

        if len(component) >= GREEDY_COMPONENT_SIZE or optimizer == "greedy":
            (
                (greedy_order, _, _, greedy_cost),
                _,
            ) = branch_and_bound(cur_aq, component, 1, OrderedDict())
            elimination_order.extend(greedy_order)
            for idx in greedy_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += greedy_cost
            continue

        if optimizer == "dfs":
            (exact_order, _, _, exact_cost), _ = branch_and_bound_dfs(cur_aq, component)
        elif optimizer == "best-first":
            (exact_order, _, _, exact_cost), _ = branch_and_bound_best_first(
                cur_aq, component
            )
        else:
            (
                (greedy_order, _, _, _greedy_cost),
                greedy_subquery_costs,
            ) = branch_and_bound(cur_aq, component, 1, OrderedDict())
            (exact_order, _, _, exact_cost), _ = branch_and_bound(
                cur_aq, component, float("inf"), greedy_subquery_costs
            )
        elimination_order.extend(exact_order)
        for idx in exact_order:
            reduce_query = cur_aq.reduce_idx(idx)
            queries.append(reduce_query)
        total_cost += exact_cost

    remaining_q = cur_aq.get_remaining_query()
    queries.append(remaining_q)
    return queries, total_cost
