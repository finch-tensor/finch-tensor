import pytest

from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import bool_, ffuncs, float64, int64
from finch.symbolic import simplify

x = ntn.Variable("x", int64)
y = ntn.Variable("y", int64)
a = ntn.Variable("a", bool_)
b = ntn.Variable("b", bool_)
c = ntn.Variable("c", bool_)


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
        # a whole-call fold is sound even where the operator is not associative
        (call(ffuncs.sub, ntn.Literal(1), ntn.Literal(2)), ntn.Literal(-1)),
        (call(ffuncs.add, ntn.Literal(7)), ntn.Literal(7)),
        # a run of adjacent literals folds in one step
        (
            call(ffuncs.add, x, ntn.Literal(1), ntn.Literal(2), ntn.Literal(3)),
            call(ffuncs.add, x, ntn.Literal(6)),
        ),
        (
            call(ffuncs.min, x, ntn.Literal(5), ntn.Literal(3)),
            call(ffuncs.min, x, ntn.Literal(3)),
        ),
        # commutativity gathers literals from wherever they sit
        (
            call(ffuncs.add, ntn.Literal(1), x, ntn.Literal(2), y, ntn.Literal(3)),
            call(ffuncs.add, x, y, ntn.Literal(6)),
        ),
        (
            call(ffuncs.min, ntn.Literal(5), x, ntn.Literal(3)),
            call(ffuncs.min, x, ntn.Literal(3)),
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


def test_simplify_does_not_flatten_a_fixed_arity_operator():
    """
    Associativity permits regrouping, not a wider call. `logical_and` is
    associative but takes exactly two arguments, so flattening a nest of them
    would build a term that cannot be typed or evaluated.
    """
    term = call(ffuncs.logical_and, call(ffuncs.logical_and, a, b), c)
    result = simplify(term)
    assert result == term
    assert result.result_type == term.result_type


def test_simplify_flattens_a_variadic_operator():
    """The same shape does flatten when the operator accepts any arity."""
    term = call(ffuncs.and_, call(ffuncs.and_, a, b), c)
    assert simplify(term) == call(ffuncs.and_, a, b, c)


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        # an identity buried one level down is lifted out, then dropped
        (
            call(ffuncs.logical_and, call(ffuncs.logical_and, a, ntn.Literal(True)), b),
            call(ffuncs.logical_and, a, b),
        ),
        # ... from either side of the nest
        (
            call(ffuncs.logical_and, a, call(ffuncs.logical_and, b, ntn.Literal(True))),
            call(ffuncs.logical_and, a, b),
        ),
        # a buried annihilator collapses the whole nest
        (
            call(
                ffuncs.logical_and, call(ffuncs.logical_and, a, ntn.Literal(False)), b
            ),
            ntn.Literal(False),
        ),
        # two literals on different levels are combined
        (
            call(
                ffuncs.logical_and,
                call(ffuncs.logical_and, a, ntn.Literal(True)),
                ntn.Literal(False),
            ),
            ntn.Literal(False),
        ),
        # lifting reaches past more than one level
        (
            call(
                ffuncs.logical_and,
                call(
                    ffuncs.logical_and,
                    call(ffuncs.logical_and, a, ntn.Literal(True)),
                    b,
                ),
                c,
            ),
            call(ffuncs.logical_and, call(ffuncs.logical_and, a, b), c),
        ),
        # a nest of nothing but literals folds away entirely
        (
            call(
                ffuncs.logical_and,
                call(ffuncs.logical_and, ntn.Literal(True), ntn.Literal(True)),
                ntn.Literal(True),
            ),
            ntn.Literal(True),
        ),
    ],
)
def test_simplify_lifts_literals_out_of_a_fixed_arity_nest(term, expected):
    assert simplify(term) == expected


def test_simplify_lifts_and_folds_numeric_literals_across_levels():
    x_f = ntn.Variable("x", float64)
    term = call(
        ffuncs.logaddexp,
        call(ffuncs.logaddexp, x_f, ntn.Literal(2.0)),
        ntn.Literal(3.0),
    )
    assert simplify(term) == call(
        ffuncs.logaddexp, x_f, ntn.Literal(ffuncs.logaddexp(2.0, 3.0))
    )


def test_simplify_does_not_reassociate_a_nest_without_literals():
    """Lifting must not churn the argument order when it buys nothing."""
    term = call(ffuncs.logical_and, call(ffuncs.logical_and, a, b), c)
    assert simplify(term) == term


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


def test_simplify_surfaces_a_malformed_call():
    """
    A literal the operator cannot be run on is a malformed term, so the error
    reaches the caller rather than being quietly skipped over.
    """
    term = call(ffuncs.add, ntn.Literal(1), ntn.Literal("s"))
    with pytest.raises(TypeError):
        simplify(term)
