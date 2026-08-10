import pytest

import numpy as np

import finch
from finch import ConstantScalar
from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import bool_, ffuncs, float64, ftype, int64, is_commutative
from finch.autoschedule import COMPILE_NUMBA
from finch.symbolic import simplify
from finch.symbolic.term import CallTerm

x = ntn.Variable("x", int64)
y = ntn.Variable("y", int64)
a = ntn.Variable("a", bool_)
b = ntn.Variable("b", bool_)
c = ntn.Variable("c", bool_)


def call(op, *args):
    return ntn.Call(ntn.Literal(op), args)


def assert_simplifies_to(term, expected):
    """
    Assert `simplify(term)` matches `expected`, ignoring the argument order of
    commutative calls.

    Which end the literals are gathered at is a canonicalization choice that no
    caller can observe, so pinning it here would make every test brittle against
    a change to that choice.
    """
    _assert_same(simplify(term), expected, ())


def _assert_same(actual, expected, path):
    where = f" at args{list(path)}" if path else ""
    match (actual, expected):
        case (CallTerm(op=a_op, args=a_args), CallTerm(op=e_op, args=e_args)) if (
            a_op == e_op and len(a_args) == len(e_args)
        ):
            if is_commutative(a_op.val):
                remaining = list(e_args)
                for arg in a_args:
                    found = next((e for e in remaining if _same(arg, e)), None)
                    assert found is not None, (
                        f"no counterpart for {arg}{where} among {list(e_args)}"
                    )
                    remaining.remove(found)
            else:
                for i, (a, e) in enumerate(zip(a_args, e_args, strict=True)):
                    _assert_same(a, e, (*path, i))
        case _:
            assert actual == expected, f"{actual} != {expected}{where}"


def _same(actual, expected) -> bool:
    try:
        _assert_same(actual, expected, ())
    except AssertionError:
        return False
    return True


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
    assert_simplifies_to(term, expected)


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
    assert_simplifies_to(term, call(ffuncs.and_, a, b, c))


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
    assert_simplifies_to(term, expected)


def test_simplify_lifts_and_folds_numeric_literals_across_levels():
    x_f = ntn.Variable("x", float64)
    term = call(
        ffuncs.logaddexp,
        call(ffuncs.logaddexp, x_f, ntn.Literal(2.0)),
        ntn.Literal(3.0),
    )
    assert_simplifies_to(
        term, call(ffuncs.logaddexp, x_f, ntn.Literal(ffuncs.logaddexp(2.0, 3.0)))
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
    assert_simplifies_to(
        term, lgc.MapJoin(lgc.Literal(ffuncs.mul), (a, lgc.Literal(6)))
    )


def test_simplify_surfaces_a_malformed_call():
    """
    A literal the operator cannot be run on is a malformed term, so the error
    reaches the caller rather than being quietly skipped over.
    """
    term = call(ffuncs.add, ntn.Literal(1), ntn.Literal("s"))
    with pytest.raises(TypeError):
        simplify(term)


@pytest.mark.parametrize(
    "dtype,folds",
    [(np.int64, True), (np.bool_, True), (np.float64, False)],
    ids=["int64", "bool", "float64"],
)
def test_annihilator_folds_only_where_it_absorbs(dtype, folds):
    """
    `x * 0` may discard `x` over the integers and booleans, but not over the
    floats, where `nan * 0` and `inf * 0` are `nan`. pydata/sparse keeps IEEE
    semantics here (it computes even a `nan` fill as `nan * 0 == nan`), so we
    do too.
    """
    op = ffuncs.logical_and if dtype is np.bool_ else ffuncs.mul
    zero = ntn.Literal(dtype(False) if dtype is np.bool_ else dtype(0))
    term = ntn.Call(ntn.Literal(op), (ntn.Variable("x", ftype(dtype)), zero))

    if folds:
        assert_simplifies_to(term, zero)
    else:
        assert_simplifies_to(term, term)


def test_float_annihilator_preserves_nan_end_to_end():
    """The guard is what keeps the compiled backend agreeing with NumPy."""
    arr = np.array([[1.0, np.nan], [np.inf, 4.0]])
    x = finch.asarray(arr)
    out = finch.compute(finch.lazy(x) * ConstantScalar(0.0), ctx=COMPILE_NUMBA)
    np.testing.assert_array_equal(out.to_numpy(), arr * 0.0)
