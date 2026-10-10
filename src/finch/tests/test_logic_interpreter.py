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
            Query(Table(HardAlias("A"), (i, k)), Table(Literal(a), (i, k))),
            Query(Table(HardAlias("B"), (k, j)), Table(Literal(b), (k, j))),
            Query(
                Table(HardAlias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
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
            Query(Table(HardAlias("A"), (i, k)), Table(Literal("A"), (i, k))),
            Query(Table(HardAlias("B"), (k, j)), Table(Literal("B"), (k, j))),
            Query(
                Table(HardAlias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
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
                Table(HardAlias("A"), (i, j)),
                Table(Literal(ft.asarray(np.array([[1, 2], [3, 4]]))), (i, j)),
            ),
            Query(
                Table(HardAlias("B"), (i, j)),
                Table(Literal(ft.asarray(np.array([[1, 1], [1, 1]]))), (i, j)),
            ),
            Query(
                Table(HardAlias("C"), (i, j)),
                MapJoin(
                    Literal(ffuncs.add),
                    (Table(HardAlias("A"), (i, j)), Table(HardAlias("B"), (i, j))),
                ),
            ),
            Query(
                Table(HardAlias("D"), (i, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("C"), (i, j)), Table(HardAlias("A"), (i, j))),
                ),
            ),
            Query(Table(HardAlias("C"), (i, j)), Table(HardAlias("B"), (i, j))),
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


@pytest.mark.parametrize(
    "a_shape,b_shape,expected",
    [((0,), (0,), (0,)), ((0,), (1,), None), ((1,), (0,), None), ((1,), (1,), (1,))],
)
def test_infer_shape_zero_extent(a_shape, b_shape, expected):
    # A zero extent is a real extent, not the unit extent of a broadcast.
    i = Field("i")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    args = (Table(a, (i,)), Table(b, (i,)))
    query = Query(Table(c, (i,)), MapJoin(Literal(ffuncs.add), args))
    if expected is None:
        with pytest.raises(ValueError, match="Dimension mismatch"):
            query.infer_shape({a: a_shape, b: b_shape})
    else:
        assert query.infer_shape({a: a_shape, b: b_shape})[c] == expected


def test_infer_shape_squeezes_only_unit_extents():
    i, j = Field("i"), Field("j")
    a = HardAlias("a")
    squeeze = Reorder(Table(a, (i, j)), (i,))
    assert squeeze.shape({a: (2, 1)}) == (2,)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        squeeze.shape({a: (2, 0)})
