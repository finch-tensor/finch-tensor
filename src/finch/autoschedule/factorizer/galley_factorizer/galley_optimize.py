"""
Galley logical optimizer: applies greedy query rewriting to logical plans, with
an optional exact branch-and-bound path for query bodies.
"""

from __future__ import annotations

import logging

from finch.algebra.tensor import TensorFType
from finch.autoschedule.factorizer.optimize import with_unique_lhs
from finch.autoschedule.stages import LogicFactorizer
from finch.autoschedule.tensor_stats.logic_to_stats import (
    insert_statistics,
)
from finch.autoschedule.util import desugar_query_into, flatten_plans
from finch.finch_logic import (
    Alias,
    LogicLoader,
    LogicStatement,
    Plan,
    Query,
    StatsFactory,
    TensorStats,
)
from finch.util.logging import LOG_GALLEY

from .annotated_query import (
    AnnotatedQuery,
)
from .branch_and_bound import (
    GalleyOptimizer,
    pruned_query_to_plan,
)
from .query_normalization import (
    postprocess_plan_after_galley,
    preprocess_plan_for_galley,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_GALLEY)


def optimize_query(
    query,
    stats_factory,
    stats_bindings,
    use_components: bool = True,
    *,
    optimizer: GalleyOptimizer = "dfs",
):
    """Rewrite a single logical Query using ``optimizer``:
    greedy, bfs, or dfs."""
    annotated_query = AnnotatedQuery(stats_factory, query, stats_bindings)
    new_queries, _ = pruned_query_to_plan(
        annotated_query,
        use_components=use_components,
        optimizer=optimizer,
    )
    return new_queries


def optimize_plan(
    plan,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    use_components: bool = True,
    *,
    optimizer: GalleyOptimizer = "greedy",
):
    """
    Optimize a full Plan: run the Galley optimizer on each Query body,
    pass through non-Query bodies (Produces), and update stats bindings.
    """
    plan = preprocess_plan_for_galley(plan)
    optimized_queries = []
    cache_dict: dict[object, TensorStats] = {}
    for body in plan.bodies:
        if isinstance(body, Query):
            new_queries = optimize_query(
                body,
                stats_factory,
                stats_bindings,
                use_components=use_components,
                optimizer=optimizer,
            )
            for new_query in new_queries:
                insert_statistics(
                    stats_factory,
                    new_query,
                    stats_bindings,
                    replace=True,
                    cache=cache_dict,
                )
            optimized_queries.extend(new_queries)
        else:
            optimized_queries.append(body)

    return postprocess_plan_after_galley(Plan(tuple(optimized_queries)))


class GalleyLogicFactorizer(LogicFactorizer):
    """
    LogicLoader stage that runs Galley on each ``Query`` body (see ``optimizer``),
    then forwards the Plan to the downstream loader ``ctx``.

    Default ``optimizer="bfs"`` is exact layered branch-and-bound; ``"dfs"`` uses the
    DFS kernel. Greedy ``k=1`` bounds are used only on the layered exact path, not
    inside ``branch_and_bound_dfs``.
    """

    def __init__(
        self,
        ctx: LogicLoader,
        use_components: bool = True,
        *,
        optimizer: GalleyOptimizer = "bfs",
    ):
        self.ctx = ctx
        self.use_components = use_components
        self.optimizer = optimizer

    def lower(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        if not isinstance(term, Plan):
            raise ValueError(f"Unsupported program type: {type(term)}")
        logger.debug("Optimizing plan: %s", term)

        def transform(prgm, bindings):
            prgm = optimize_plan(
                prgm,
                stats_factory,
                stats,
                use_components=self.use_components,
                optimizer=self.optimizer,
            )
            return prgm, bindings

        # Galley only keeps the queries that compute produced aliases, so each
        # write to a bound tensor, including an in-place update, is renamed and
        # produced, then copied back to the tensor it updates.
        term = desugar_query_into(term)
        term, bindings = with_unique_lhs(transform, term, bindings)
        assert isinstance(term, Plan)
        return self.ctx(flatten_plans(term), bindings, stats, stats_factory)
