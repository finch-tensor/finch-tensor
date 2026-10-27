from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import AliasedForm
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Fuse,
    Literal,
    LogicExpression,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Relabel,
    Reorder,
    Table,
)

from .cost_models import NS, Basis, LinearCostModel


@dataclass(frozen=True)
class Statement:
    """A Query or QueryInto of a program."""

    stmt: LogicStatement


class FactorizeBasis(Basis[Statement]):
    """
    A basis where each query statement is a component, in program order.
    get_component_features returns one row for each statement.
    """

    def components(
        self,
        term: LogicStatement,
        bindings: Mapping[Alias, TensorFType],
        machine: StatsMachine[NS],
    ) -> Iterator[Statement]:
        match term:
            case Plan(bodies):
                for body in bodies:
                    yield from self.components(body, bindings, machine)
            case Fuse(_, body):
                yield from self.components(body, bindings, machine)
            case Query() | QueryInto():
                yield Statement(term)
                machine(term)
            case Produces():
                return
            case _:
                raise ValueError(f"Unsupported statement type: {term}")


class FlopsBasis(FactorizeBasis, AliasedForm):
    """
    FlopsBasis counts the non-fill values in the iteration space of each
    query, and in the output it materializes. Each query is a component.
    """

    @property
    def n_features(self) -> int:
        return 2

    def get_component_features(
        self,
        component: Statement,
        machine: StatsMachine[NS],
    ) -> np.ndarray:
        stats_factory = machine.stats_factory

        def visit_expr(expr: LogicExpression) -> tuple[NS, NS]:
            """Return the stats of the output of expr, and of its iteration space."""
            match expr:
                case Aggregate(Literal(op), Literal(init), arg, idxs):
                    iter_stats, _ = visit_expr(arg)
                    out_stats = stats_factory.aggregate(op, init, idxs, iter_stats)
                    return out_stats, iter_stats
                case Aggregate(Literal(op), init, arg, idxs):
                    # A tensor init is folded into the reduction.
                    iter_stats, _ = visit_expr(arg)
                    init_stats, _ = visit_expr(init)
                    out_stats = stats_factory.aggregate(op, None, idxs, iter_stats)
                    out_stats = stats_factory.mapjoin(op, init_stats, out_stats)
                    return out_stats, iter_stats
                case MapJoin(Literal(op), (arg,)):
                    return visit_expr(arg)
                case MapJoin(Literal(op), args):
                    out_stats = stats_factory.mapjoin(
                        op, *(visit_expr(arg)[0] for arg in args)
                    )
                    return out_stats, out_stats
                case Reorder(arg, idxs):
                    out_stats, iter_stats = visit_expr(arg)
                    return stats_factory.reorder(out_stats, idxs), iter_stats
                case Relabel(arg, idxs):
                    out_stats, iter_stats = visit_expr(arg)
                    return stats_factory.relabel(out_stats, idxs), iter_stats
                case Table() | Literal():
                    out_stats = machine(expr)
                    return out_stats, out_stats
                case _:
                    raise ValueError(f"Unsupported expression type: {expr}")

        match component.stmt:
            case Query(Table(Alias()), rhs):
                out_stats, iter_stats = visit_expr(rhs)
                return np.array(
                    [
                        [
                            iter_stats.estimate_non_fill_values(),
                            out_stats.estimate_non_fill_values(),
                        ]
                    ],
                    dtype=float,
                )
            case QueryInto() as stmt:
                return self.get_component_features(Statement(stmt.as_query()), machine)
            case stmt:
                raise ValueError(f"Unsupported statement type: {stmt}")


# Allocations are weighted 10x iterations.
FLOPS_COST_MODEL = LinearCostModel(FlopsBasis(), [1.0, 10.0])
