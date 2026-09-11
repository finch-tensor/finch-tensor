import pytest

import numpy as np

import finch as fl
from finch.algebra import ffuncs
from finch.finch_logic import (
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.finch_logic.interpreter import LogicInterpreter


def test_table_query_lhs():
    i, j = Field("i"), Field("j")
    A = fl.asarray(np.array([[1, 2], [3, 4]]))
    plan_query_reg = Plan(
        (
            Query(
                HardAlias("B"),
                MapJoin(
                    Literal(ffuncs.mul), (Table(HardAlias("A"), (i, j)), Literal(2))
                ),
            ),
            Produces((HardAlias("B"),)),
        )
    )
    plan_query_table = Plan(
        (
            Query(
                Table(HardAlias("B"), (i, j)),
                MapJoin(
                    Literal(ffuncs.mul), (Table(HardAlias("A"), (i, j)), Literal(2))
                ),
            ),
            Produces((HardAlias("B"),)),
        )
    )
    bindings = {HardAlias("A"): A}
    (r1,) = LogicInterpreter()(plan_query_reg, dict(bindings))
    (r2,) = LogicInterpreter()(plan_query_table, dict(bindings))
    assert (r1.to_numpy() == r2.to_numpy()).all()


def test_table_alias():
    i, j = Field("i"), Field("j")
    a = fl.asarray(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError):
        Query(Table(Literal(a), (i, j)), Table(Literal(a), (i, j)))
