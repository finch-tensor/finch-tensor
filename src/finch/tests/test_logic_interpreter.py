import _operator  # noqa: F401

import pytest

import numpy as np
from numpy import array  # noqa: F401

import finch as ft
import finch.finch_logic as lgc
from finch.algebra import ffuncs
from finch.finch_logic import (
    Aggregate,
    Field,
    HardAlias,
    Literal,
    LogicInterpreter,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
    TableValue,
)

from .conftest import finch_assert_equal


@pytest.mark.parametrize(
    "a, b",
    [
        (
            ft.asarray(np.array([[1, 2], [3, 4]])),
            ft.asarray(np.array([[5, 6], [7, 8]])),
        ),
        (
            ft.asarray(np.array([[2, 0], [1, 3]])),
            ft.asarray(np.array([[4, 1], [2, 2]])),
        ),
    ],
)
def test_matrix_multiplication(a, b):
    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
        (
            Query(HardAlias("A"), Table(Literal(a), (i, k))),
            Query(HardAlias("B"), Table(Literal(b), (k, j))),
            Query(
                HardAlias("AB"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                HardAlias("C"),
                Reorder(
                    Aggregate(
                        Literal(ffuncs.add),
                        Literal(0),
                        Table(HardAlias("AB"), (i, k, j)),
                        (k,),
                    ),
                    (i, j),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    result = LogicInterpreter()(p)[0]

    expected = np.matmul(a.to_numpy(), b.to_numpy())

    assert (result.to_numpy() == expected).all()


def test_plan_repr():
    i = Field("i")
    j = Field("j")
    k = Field("k")
    # To avoid equality issues with numpy arrays, we use string literals here instead
    p = Plan(
        (
            Query(HardAlias("A"), Table(Literal("A"), (i, k))),
            Query(HardAlias("B"), Table(Literal("B"), (k, j))),
            Query(
                HardAlias("AB"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                HardAlias("C"),
                Reorder(
                    Aggregate(
                        Literal(ffuncs.add),
                        Literal(0),
                        Table(HardAlias("AB"), (i, k, j)),
                        (k,),
                    ),
                    (i, j),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    assert p == eval(repr(p), {**vars(lgc), **vars(ffuncs), **globals()})


def test_materialize():
    i = Field("i")
    j = Field("j")

    C = ft.asarray(np.array([[0, 0], [0, 0]]))

    p = Plan(
        (
            Query(
                HardAlias("A"),
                Table(Literal(ft.asarray(np.array([[1, 2], [3, 4]]))), (i, j)),
            ),
            Query(
                HardAlias("B"),
                Table(Literal(ft.asarray(np.array([[1, 1], [1, 1]]))), (i, j)),
            ),
            Query(
                HardAlias("C"),
                MapJoin(
                    Literal(ffuncs.add),
                    (Table(HardAlias("A"), (i, j)), Table(HardAlias("B"), (i, j))),
                ),
            ),
            Query(
                HardAlias("D"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("C"), (i, j)), Table(HardAlias("A"), (i, j))),
                ),
            ),
            Query(HardAlias("C"), Table(HardAlias("B"), (i, j))),
            Produces((HardAlias("D"), HardAlias("C"))),
        )
    )

    result = LogicInterpreter()(p, {HardAlias("C"): C})[0]

    expected = ft.asarray(
        np.array([[((1 + 1) * 1), ((2 + 1) * 2)], [((3 + 1) * 3), ((4 + 1) * 4)]])
    )

    assert (result.to_numpy() == expected.to_numpy()).all()
    finch_assert_equal(C, ft.asarray(np.array([[1, 1], [1, 1]])))


@pytest.mark.parametrize(
    "node",
    [
        Literal(6.0),
        Reorder(Literal(6.0), ()),
        Relabel(Literal(6.0), ()),
        Aggregate(Literal(ffuncs.add), Literal(0.0), Literal(6.0), ()),
        MapJoin(Literal(ffuncs.add), (Literal(2.0), Literal(4.0))),
    ],
    ids=["literal", "reorder", "relabel", "aggregate", "mapjoin"],
)
def test_bare_literal_is_zero_dimensional(node):
    """A bare Literal evaluates to a rank-0 TableValue, so every node type can
    consume it without special-casing raw scalars."""
    result = LogicInterpreter()(node)
    assert isinstance(result, TableValue)
    assert result.idxs == ()
    assert float(np.asarray(result.tns)) == 6.0
