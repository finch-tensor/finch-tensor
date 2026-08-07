import pytest

from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import ffuncs, int64
from finch.symbolic import simplify

x = ntn.Variable("x", int64)
y = ntn.Variable("y", int64)


def call(op, *args):
    return ntn.Call(ntn.Literal(op), args)


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        # constant folding
        (call(ffuncs.add, ntn.Literal(1), ntn.Literal(2)), ntn.Literal(3)),
        (
            call(
                ffuncs.mul,
                call(ffuncs.add, ntn.Literal(1), ntn.Literal(2)),
                ntn.Literal(3),
            ),
            ntn.Literal(9),
        ),
        # identities, from either side
        (call(ffuncs.add, x, ntn.Literal(0)), x),
        (call(ffuncs.add, ntn.Literal(0), x), x),
        (call(ffuncs.mul, x, ntn.Literal(1)), x),
        (call(ffuncs.or_, x, ntn.Literal(False)), x),
        (call(ffuncs.and_, x, ntn.Literal(True)), x),
        # annihilators
        (call(ffuncs.mul, x, ntn.Literal(0)), ntn.Literal(0)),
        (call(ffuncs.mul, ntn.Literal(0), x), ntn.Literal(0)),
        (call(ffuncs.and_, x, ntn.Literal(False)), ntn.Literal(False)),
        (call(ffuncs.or_, x, ntn.Literal(True)), ntn.Literal(True)),
        # flattening and folding literals across an associative call
        (
            call(ffuncs.add, call(ffuncs.add, x, ntn.Literal(1)), ntn.Literal(2)),
            call(ffuncs.add, x, ntn.Literal(3)),
        ),
        # non-adjacent literals meet once they are hoisted
        (
            call(ffuncs.add, ntn.Literal(1), x, ntn.Literal(2)),
            call(ffuncs.add, x, ntn.Literal(3)),
        ),
        # folding can expose an identity, which can expose a singleton
        (call(ffuncs.add, ntn.Literal(1), x, ntn.Literal(-1)), x),
        # idempotence
        (call(ffuncs.min, x, y, x), call(ffuncs.min, x, y)),
        (call(ffuncs.max, x, x), x),
        # a one-argument reduction is its argument
        (call(ffuncs.add, x), x),
        # everything at once
        (
            call(
                ffuncs.mul,
                call(ffuncs.add, call(ffuncs.mul, x, ntn.Literal(1)), ntn.Literal(0)),
                call(ffuncs.mul, ntn.Literal(2), ntn.Literal(3)),
            ),
            call(ffuncs.mul, x, ntn.Literal(6)),
        ),
    ],
)
def test_simplify_notation(term, expected):
    assert simplify(term) == expected


@pytest.mark.parametrize(
    "term",
    [
        # sub is not associative, so 0 is not known to be droppable
        call(ffuncs.sub, x, ntn.Literal(0)),
        # truediv reports 1 as an identity, but only as the divisor
        call(ffuncs.truediv, ntn.Literal(1), x),
        # nothing to do
        call(ffuncs.add, x, y),
        x,
    ],
)
def test_simplify_leaves_alone(term):
    assert simplify(term) == term


def test_simplify_recurses_into_statements():
    block = ntn.Block(
        (
            ntn.Assign(
                y, call(ffuncs.add, call(ffuncs.mul, x, ntn.Literal(1)), ntn.Literal(0))
            ),
            ntn.Assign(x, call(ffuncs.mul, y, ntn.Literal(0))),
        )
    )
    assert simplify(block) == ntn.Block(
        (ntn.Assign(y, x), ntn.Assign(x, ntn.Literal(0)))
    )


def test_simplify_assembly():
    v = asm.Variable("x", int64)
    term = asm.Call(
        asm.Literal(ffuncs.mul),
        (
            asm.Call(asm.Literal(ffuncs.add), (v, asm.Literal(0))),
            asm.Literal(1),
        ),
    )
    assert simplify(term) == v


def test_simplify_logic_mapjoin():
    a = lgc.Alias("A")
    term = lgc.MapJoin(
        lgc.Literal(ffuncs.mul),
        (lgc.MapJoin(lgc.Literal(ffuncs.mul), (a, lgc.Literal(2))), lgc.Literal(3)),
    )
    assert simplify(term) == lgc.MapJoin(lgc.Literal(ffuncs.mul), (a, lgc.Literal(6)))


def test_simplify_tolerates_non_scalar_literals():
    """A literal the operator cannot be run on is left untouched, not an error."""
    import numpy as np

    term = call(ffuncs.mul, x, ntn.Literal(np.zeros(3)))
    assert simplify(term) == term
