import pytest

import numpy as np

import finch
import finch.finch_logic as lgc
from finch import ffuncs
from finch.algebra import DynamicFill
from finch.autoschedule.util import fuse_loop_order

i, j, k = lgc.Field("i"), lgc.Field("j"), lgc.Field("k")
A, B, C, w = (lgc.HardAlias(name) for name in "ABCw")
add, mul = lgc.Literal(ffuncs.add), lgc.Literal(ffuncs.mul)


def query(lhs, idxs, rhs):
    return lgc.Query(lgc.Table(lhs, idxs), rhs)


copy = query(B, (i, j), lgc.Table(A, (i, j)))
matmul = query(
    C,
    (i, j),
    lgc.Aggregate(
        add,
        lgc.Literal(np.float64(0.0)),
        lgc.MapJoin(mul, (lgc.Table(A, (i, k)), lgc.Table(B, (k, j)))),
        (k,),
    ),
)


@pytest.mark.parametrize(
    "prgm",
    [
        lgc.Plan(),
        copy,
        lgc.Plan((copy,)),
        lgc.Plan((copy, matmul, lgc.Produces((C,)))),
        # Nested plans and plans of one statement are printed explicitly.
        lgc.Plan((lgc.Plan((copy, matmul)), lgc.Plan(), lgc.Produces((C,)))),
        lgc.Plan((lgc.Plan((copy,)), lgc.Plan((lgc.Plan((matmul,)),)))),
        lgc.Fuse(i, lgc.Plan()),
        lgc.Fuse(i, lgc.Plan((copy,))),
        fuse_loop_order(lgc.Plan((copy, matmul)), (i, j)),
        lgc.Plan(
            (
                lgc.Fuse(
                    i,
                    lgc.Plan(
                        (
                            query(lgc.FusedAlias(w, 1), (i, j), lgc.Table(A, (i, j))),
                            lgc.Fuse(
                                j,
                                lgc.QueryInto(
                                    lgc.Table(B, (i, j)),
                                    add,
                                    lgc.Table(lgc.FusedAlias(w, 1), (i, j)),
                                ),
                            ),
                        )
                    ),
                ),
                lgc.Produces((B, lgc.FusedAlias(w, 1))),
            )
        ),
        query(lgc.FusedAlias(w, 0), (i,), lgc.Table(A, (i,))),
        query(lgc.FusedAlias(w, 2), (i, j), lgc.Table(A, (i, j))),
        query(
            C,
            (),
            lgc.MapJoin(
                mul,
                (
                    lgc.FillTable(A),
                    lgc.FillTable(lgc.FusedAlias(w, 1)),
                    lgc.Reorder(lgc.Relabel(lgc.Table(A, (i, j)), (j, i)), (i, j)),
                    lgc.Table(lgc.Literal(np.float32(2.0)), ()),
                ),
            ),
        ),
        lgc.Produces(()),
        lgc.Table(lgc.FusedAlias(w, 1), (i, j)),
        lgc.MapJoin(add, (lgc.Table(A, (i,)),)),
    ],
)
def test_parse_logic_round_trips(prgm):
    text = str(prgm)
    parsed = lgc.parse_logic(text)
    assert parsed == prgm
    assert str(parsed) == text


@pytest.mark.parametrize(
    "val",
    [
        ffuncs.add,
        ffuncs.floordiv,
        ffuncs.choose(np.float64(0.0)),
        ffuncs.choose(DynamicFill(np.int64(0), finch.int64)),
        ffuncs.astype(finch.float32),
        ffuncs.init_write(np.float64(1.0)),
        ffuncs.scaled_negative_power(-2.0),
        ffuncs.scaled_power(2.0),
        0,
        -3,
        1.5e-300,
        -0.0,
        float("inf"),
        -float("inf"),
        1 + 2j,
        True,
        None,
        "a 'quoted'\nstring",
        np.float64(1.0),
        np.float32(-1.5),
        np.int16(7),
        np.uint8(0),
        np.True_,
        np.complex128(1 - 2j),
        finch.int64,
        DynamicFill(np.int64(0), finch.int64),
        (),
        (np.float64(0.0),),
        (1, (ffuncs.add, "s")),
    ],
)
def test_parse_logic_round_trips_literals(val):
    prgm = query(C, (), lgc.MapJoin(add, (lgc.Literal(val),)))
    assert lgc.parse_logic(str(prgm)) == prgm


def test_nested_plans_print_explicitly():
    prgm = lgc.Plan(
        (lgc.Plan((copy, lgc.Produces((B,)))), lgc.Plan((copy,)), lgc.Plan())
    )
    assert str(prgm) == (
        "plan:\n"
        "    B[i, j] = A[i, j]\n"
        "    return B\n"
        "plan:\n"
        "    B[i, j] = A[i, j]\n"
        "plan:"
    )


def test_parse_logic_names_values_from_namespace():
    tns = np.ones((2, 2))
    prgm = lgc.parse_logic("A[i, j] = Table(ones, i, j)", {"ones": tns})
    assert prgm == query(A, (i, j), lgc.Table(lgc.Literal(tns), (i, j)))


def test_parse_logic_rejects_unknown_literals():
    with pytest.raises(ValueError, match="Unknown literal 'ones'"):
        lgc.parse_logic("A[i, j] = Table(ones, i, j)")
