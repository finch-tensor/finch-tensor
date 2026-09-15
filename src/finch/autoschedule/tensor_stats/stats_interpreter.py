from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Generic, TypeVar, overload

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicExpression,
    LogicInterpreter,
    LogicNode,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    StatsFactory,
    Table,
)
from finch.tensor import Scalar
from finch.util.logging import LOG_LOGIC_PRE_OPT

from .numeric_stats import NumericStats
from .tensor_stats import TensorStats

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_PRE_OPT)

TS = TypeVar("TS", bound=TensorStats)


class StatsInterpreter(Generic[TS]):
    def __init__(
        self,
        stats_factory: StatsFactory[TS],
    ):
        self.stats_factory = stats_factory

    def __call__(
        self, node: LogicNode, bindings: MutableMapping[Alias, TS]
    ) -> TS | tuple[TS, ...]:
        if not isinstance(bindings, OrderedDict):
            bindings = OrderedDict(bindings)
        machine = StatsMachine(
            stats_factory=self.stats_factory,
            bindings=bindings,
        )
        return machine(node)


class StatsMachine(Generic[TS]):
    def __init__(
        self,
        stats_factory: StatsFactory[TS],
        bindings=None,
    ):
        self.stats_factory = stats_factory
        if bindings is None:
            bindings = OrderedDict()
        self.bindings = bindings

    @overload
    def __call__(self, node: LogicExpression) -> TS: ...

    @overload
    def __call__(self, node: Alias) -> TS: ...

    @overload
    def __call__(self, node: LogicStatement) -> tuple[TS, ...]: ...

    @overload
    def __call__(self, node: LogicNode) -> TS | tuple[TS, ...]: ...

    def __call__(self, node) -> TS | tuple[TS, ...]:
        logger.debug("Evaluating: %s", node)
        match node:
            case Plan():
                last_result: TensorStats | tuple[TensorStats, ...] = ()
                for body in node.bodies:
                    last_result = self(body)
                return last_result

            case Query():
                rhs_stats = self(node.rhs)
                self.bindings[node.lhs] = rhs_stats
                return rhs_stats

            case Alias():
                stats = self.bindings.get(node)
                if stats is None:
                    raise ValueError(f"undefined tensor alias {node}")
                return stats

            case Table(tns, idxs):
                if isinstance(tns, Literal):
                    tensor = self.stats_factory(tns.val, idxs)
                elif isinstance(tns, Alias):
                    base = self.bindings.get(tns)
                    if base is None:
                        raise ValueError(f"No TensorStats bound to alias {node.tns}")

                    new_indices = tuple(f for f in node.idxs)
                    tensor = self.stats_factory.relabel(base, new_indices)
                return tensor

            case MapJoin():
                if not isinstance(node.op, Literal):
                    raise TypeError("MapJoin.op must be Literal(...).")
                child_stats_list: list[TS] = []
                for arg in node.args:
                    res = self(arg)
                    child_stats_list.append(res)
                return self.stats_factory.mapjoin(node.op.val, *child_stats_list)

            case Aggregate():
                if not isinstance(node.op, Literal):
                    raise TypeError("Aggregate.op must be Literal(...).")
                op = node.op.val
                init = node.init.val if isinstance(node.init, Literal) else None
                arg2 = self(node.arg)
                reduce_indices = node.idxs
                return self.stats_factory.aggregate(op, init, reduce_indices, arg2)

            case Literal(val):
                return self.stats_factory(Scalar(val), ())

            case Reorder():
                return self.stats_factory.reorder(self(node.arg), node.idxs)

            case Relabel():
                base = self(node.arg)
                new_indices = tuple(f for f in node.idxs)
                return self.stats_factory.relabel(base, new_indices)

            case Produces(args):
                ret = tuple(self(arg) for arg in args)
                assert all(not isinstance(r, tuple) for r in ret)
                return ret

            case _:
                raise TypeError(f"Unhandled node type {type(node)}")


def calculate_estimated_error(
    node: LogicNode,
    stats_factory: StatsFactory,
    logic_bindings: MutableMapping[Alias, TensorFType],
    stats_bindings: MutableMapping[Alias, TensorStats],
) -> tuple[float, ...]:
    if logic_bindings is None:
        logic_bindings = OrderedDict()

    if stats_bindings is None:
        stats_bindings = OrderedDict()

    logic_interpreter = LogicInterpreter()
    actual_result = logic_interpreter(node, logic_bindings)

    if not isinstance(actual_result, tuple):
        actual_result = (actual_result,)

    stats_interpreter = StatsInterpreter(stats_factory=stats_factory)
    stats_result = stats_interpreter(node, stats_bindings)

    if not isinstance(stats_result, tuple):
        stats_result = (stats_result,)

    errors = []
    for actual_tns, stats_obj in zip(actual_result, stats_result, strict=True):
        actual_nnz = float(np.count_nonzero(actual_tns.to_numpy()))
        if isinstance(stats_obj, NumericStats):
            est_nnz = float(stats_obj.estimate_non_fill_values())
        else:
            raise TypeError("Stats Class must be inherit from NumericStats")

        if actual_nnz == 0.0:
            rel_err = 0.0 if est_nnz == 0.0 else float("inf")

        else:
            rel_err = abs(actual_nnz - est_nnz) / actual_nnz

        errors.append(rel_err)
    return tuple(errors)
