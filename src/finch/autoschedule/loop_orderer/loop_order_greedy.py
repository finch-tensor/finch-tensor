from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from finch.autoschedule.cost_models import (
    LoopCostModel,
    MultiLoopCostModel,
    PigeonCostModel,
    TransposeCostModel,
)
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

from .loop_ordering import AbstractLoopOrderer, with_loop_order

NS = TypeVar("NS", bound=NumericStats)


def connected_loop_candidates(
    prefix: tuple[Field, ...],
    remaining: Sequence[Field],
    output_vars: tuple[Field, ...] | None = None,
) -> list[Field]:
    """Preserve candidate order while respecting the output's index order."""
    output_vars = output_vars or ()
    next_output = next((idx for idx in output_vars if idx not in prefix), None)
    return [idx for idx in remaining if idx not in output_vars or idx == next_output]


def greedy_loop_order(
    expr: LogicExpression,
    stats_factory: StatsFactory[NS],
    stats_bindings: Mapping[Alias, NS],
    output_vars: tuple[Field, ...] | None = None,
    *,
    cost_model: LoopCostModel | None = None,
) -> tuple[Field, ...]:
    """Build a loop order by appending the cheapest candidate prefix."""
    if cost_model is None:
        cost_model = MultiLoopCostModel(PigeonCostModel(), TransposeCostModel())
    machine = StatsMachine(stats_factory, dict(stats_bindings))
    match expr:
        case Aggregate(arg=arg):
            pass
        case _:
            arg = expr
    fields = (*arg.fields(), *(expr.fields() if output_vars is None else output_vars))
    remaining = list(dict.fromkeys(fields))
    prefix: list[Field] = []

    def candidate_cost(field: Field) -> float:
        return cost_model.predict_prefix_cost(
            tuple(prefix) + (field,), expr, machine, output_vars
        )

    while remaining:
        candidates = connected_loop_candidates(tuple(prefix), remaining, output_vars)
        best = min(candidates, key=candidate_cost)
        prefix.append(best)
        remaining.remove(best)

    return tuple(prefix)


def set_greedy_loop_order(
    plan: Plan,
    stats_factory: StatsFactory[NS],
    stats: dict[Alias, NS],
    cost_model: LoopCostModel | None = None,
) -> Plan:
    stats_bindings = dict(stats)
    new_queries = []
    for query in plan.bodies[:-1]:
        match query:
            case QueryInto(_, _, Literal()):
                new_queries.append(query)
            case Query(Table(_, out_idxs), Aggregate() as arg) | QueryInto(
                Table(_, out_idxs), _, arg
            ):
                match query:
                    case QueryInto():
                        arg = query.as_query().rhs
                idxs_2 = greedy_loop_order(
                    arg, stats_factory, stats_bindings, out_idxs, cost_model=cost_model
                )
                new_queries.append(with_loop_order(query, idxs_2))
            case Query(_, Table(Alias(), _)) as q:
                new_queries.append(q)
            case _:
                raise Exception(f"Invalid node: {query} in set_greedy_loop_order")

        insert_statistics(stats_factory, query, stats_bindings, replace=False, cache={})

    return Plan(tuple(new_queries + [plan.bodies[-1]]))


class GreedyLoopOrderer(AbstractLoopOrderer, Generic[NS]):
    def __init__(
        self,
        ctx: LogicLoader | None = None,
        *,
        cost_model: LoopCostModel | None = None,
    ):
        super().__init__(ctx)
        self.cost_model = (
            cost_model
            if cost_model is not None
            else MultiLoopCostModel(PigeonCostModel(), TransposeCostModel())
        )

    def set_loop_orders(
        self,
        prgm: Plan,
        stats: dict[Alias, NS],
        stats_factory: StatsFactory[NS],
    ) -> Plan:
        return set_greedy_loop_order(prgm, stats_factory, stats, self.cost_model)
