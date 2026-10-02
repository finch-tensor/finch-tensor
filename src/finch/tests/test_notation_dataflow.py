import pytest

import numpy as np

import finch
import finch.finch_notation as ntn
from finch import ffuncs
from finch.finch_notation import NotationCFGBuilder, notation_copy_propagation

from .scripts.nodes import create_ntn_simple_node

T = finch.ftype(finch.asarray(np.zeros((1,))))

x = ntn.Variable("x", finch.int64)
y = ntn.Variable("y", finch.int64)
z = ntn.Variable("z", finch.int64)
i = ntn.Variable("i", finch.int64)
n = ntn.Variable("n", finch.int64)
C = ntn.Variable("C", T)
D = ntn.Variable("D", T)
C_ = ntn.Slot("C_", T)


def lit(v):
    return ntn.Literal(np.int64(v))


def add(a, b):
    return ntn.Call(ntn.Literal(ffuncs.add), (a, b))


def extent(hi):
    return ntn.Call(ntn.Literal(finch.compile.make_extent), (lit(0), hi))


def func(args, *body):
    return ntn.Module(
        (ntn.Function(ntn.Variable("f", finch.int64), args, ntn.Block(body)),)
    )


def test_ntn_cfg_printer_simple(file_regression):
    prgm = create_ntn_simple_node()
    cfg = NotationCFGBuilder().build(prgm)
    file_regression.check(str(cfg), extension=".txt")


@pytest.mark.parametrize(
    "prgm, expected",
    [
        # y = x; return y  ->  return x
        (
            func((x,), ntn.Assign(y, x), ntn.Return(y)),
            func((x,), ntn.Assign(y, x), ntn.Return(x)),
        ),
        # the copy is propagated into nested expressions
        (
            func((x,), ntn.Assign(y, x), ntn.Assign(z, add(y, y)), ntn.Return(z)),
            func((x,), ntn.Assign(y, x), ntn.Assign(z, add(x, x)), ntn.Return(z)),
        ),
        (
            func((x,), ntn.Assign(y, x), ntn.Assign(z, y), ntn.Return(z)),
            func((x,), ntn.Assign(y, x), ntn.Assign(z, x), ntn.Return(x)),
        ),
    ],
)
def test_copy_propagation_straight_line(prgm, expected):
    assert notation_copy_propagation(prgm) == expected


@pytest.mark.parametrize(
    "prgm",
    [
        # redefining the source kills `y -> x`
        func((x,), ntn.Assign(y, x), ntn.Assign(x, lit(1)), ntn.Return(y)),
        # redefining the destination kills `y -> x`
        func((x,), ntn.Assign(y, x), ntn.Assign(y, lit(2)), ntn.Return(y)),
        # self-assignment is not a copy
        func((x,), ntn.Assign(x, x), ntn.Return(x)),
        # a non-variable right-hand side is not a copy
        func((x,), ntn.Assign(y, add(x, lit(1))), ntn.Return(y)),
    ],
)
def test_copy_propagation_no_change(prgm):
    assert notation_copy_propagation(prgm) == prgm


def test_copy_propagation_ifelse_both_branches():
    def prgm(ret):
        return func(
            (x, n),
            ntn.IfElse(n, ntn.Assign(y, x), ntn.Assign(y, x)),
            ntn.Return(ret),
        )

    assert notation_copy_propagation(prgm(y)) == prgm(x)


def test_copy_propagation_ifelse_one_branch():
    # only one branch establishes `y -> x`, so it does not hold after the join
    prgm = func(
        (x, n),
        ntn.IfElse(n, ntn.Assign(y, x), ntn.Assign(y, lit(0))),
        ntn.Return(y),
    )
    assert notation_copy_propagation(prgm) == prgm


def test_copy_propagation_if_kills_copy():
    # `If` has an implicit empty else branch, so `y -> x` survives only on one path
    prgm = func(
        (x, n),
        ntn.Assign(y, x),
        ntn.If(n, ntn.Assign(x, lit(0))),
        ntn.Return(y),
    )
    assert notation_copy_propagation(prgm) == prgm


def test_copy_propagation_into_loop():
    # a copy made before a loop and untouched inside it holds in the loop body
    def prgm(use):
        return func(
            (x, n),
            ntn.Assign(y, x),
            ntn.Loop(i, extent(n), ntn.Assign(z, add(use, i))),
            ntn.Return(use),
        )

    assert notation_copy_propagation(prgm(y)) == prgm(x)


def test_copy_propagation_loop_back_edge_kills_copy():
    # `y` is redefined later in the body, so on the second iteration the use
    # of `y` no longer refers to `x`
    prgm = func(
        (x, n),
        ntn.Assign(y, x),
        ntn.Loop(
            i,
            extent(n),
            ntn.Block((ntn.Assign(z, add(y, i)), ntn.Assign(y, lit(0)))),
        ),
        ntn.Return(z),
    )
    assert notation_copy_propagation(prgm) == prgm


def test_copy_propagation_loop_index_kills_copy():
    # `Loop(i, ...)` redefines `i`, so inside the loop `y` no longer equals `i`
    prgm = func(
        (i, n),
        ntn.Assign(y, i),
        ntn.Loop(i, extent(n), ntn.Assign(z, y)),
        ntn.Return(z),
    )
    assert notation_copy_propagation(prgm) == prgm


def test_copy_propagation_repack_kills_copy():
    # `Repack(C_, C)` redefines `C`, so `D` is no longer a copy of it
    prgm = func(
        (C,),
        ntn.Assign(D, C),
        ntn.Unpack(C_, D),
        ntn.Repack(C_, C),
        ntn.Return(D),
    )
    expected = func(
        (C,),
        ntn.Assign(D, C),
        ntn.Unpack(C_, C),
        ntn.Repack(C_, C),
        ntn.Return(D),
    )
    assert notation_copy_propagation(prgm) == expected


def test_copy_propagation_transitive():
    prgm = func(
        (x,),
        ntn.Assign(y, x),
        ntn.Assign(z, y),
        ntn.Return(z),
    )
    expected = func(
        (x,),
        ntn.Assign(y, x),
        ntn.Assign(z, x),
        ntn.Return(x),
    )
    assert notation_copy_propagation(prgm) == expected
