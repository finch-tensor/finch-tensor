import pytest

import numpy as np

import finch
from finch import ConstantScalar
from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import bool_, ffuncs, float64, ftype, int64, is_commutative
from finch.autoschedule import (
    DefaultLogicFormatter,
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    LogicCapture,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)
from finch.compile import NotationCompiler
from finch.finch_assembly import (
    AssemblyInterpreter,
    AssemblyLoader,
    AssemblySimplify,
    LowerPackedStructSlots,
)
from finch.finch_logic import LogicSimplify
from finch.finch_logic.simplification import simplify_logic, unwrap_literal
from finch.finch_notation.interpreter import NotationInterpreter
from finch.symbolic import UnvalidatedForm
from finch.symbolic.rewriters import Chain, Fixpoint, PostWalk, Rewrite
from finch.symbolic.simplification import simplify_rules
from finch.symbolic.term import CallTerm

x = ntn.Variable("x", int64)
y = ntn.Variable("y", int64)
a = ntn.Variable("a", bool_)
b = ntn.Variable("b", bool_)
c = ntn.Variable("c", bool_)


def simplify(term):
    return Rewrite(Fixpoint(PostWalk(Chain(simplify_rules()))))(term)


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
    a = lgc.HardAlias("A")
    term = lgc.MapJoin(
        lgc.Literal(ffuncs.mul),
        (lgc.MapJoin(lgc.Literal(ffuncs.mul), (a, lgc.Literal(2))), lgc.Literal(3)),
    )
    assert_simplifies_to(
        term, lgc.MapJoin(lgc.Literal(ffuncs.mul), (a, lgc.Literal(6)))
    )


def test_annihilator_folds():
    term = ntn.Call(
        ntn.Literal(ffuncs.mul), (ntn.Variable("x", ftype(0)), ntn.Literal(0))
    )
    assert_simplifies_to(term, ntn.Literal(0))


def _capturing_scheduler():
    """A scheduler that records the program `LogicSimplify` hands downstream."""
    capture = LogicCapture(
        DefaultLoopOrderer(DefaultLogicFormatter(LogicCompiler(NotationInterpreter())))
    )
    executor = LogicExecutor(DefaultLogicOptimizer(LogicSimplify(capture)))
    return capture, LogicNormalizer(executor)


def _mapjoins(node):
    if isinstance(node, lgc.MapJoin):
        return [node, *(m for c in node.children for m in _mapjoins(c))]
    if isinstance(node, lgc.LogicTree):
        return [m for c in node.children for m in _mapjoins(c)]
    return []


def test_logic_simplify_unwraps_literals():
    """
    `optimize` leaves a literal operand inside a `Reorder`, which hides it from
    every rule keyed on `LiteralTerm`.
    """
    lit = lgc.Literal(2)
    assert unwrap_literal(lgc.Reorder(lit, ())) == lit
    assert unwrap_literal(lgc.Relabel(lit, ())) == lit
    # A reorder that does something is left alone.
    idxs = (lgc.Field("i"), lgc.Field("j"))
    table = lgc.Reorder(lgc.Table(lgc.HardAlias("A"), idxs), idxs[::-1])
    assert unwrap_literal(table) is None


def test_logic_simplify_folds_wrapped_literals():
    """Once unwrapped, an all-literal MapJoin evaluates at compile time."""
    two = lgc.Reorder(lgc.Literal(2), ())
    three = lgc.Reorder(lgc.Literal(3), ())
    expr = lgc.MapJoin(lgc.Literal(ffuncs.add), (two, three))
    assert simplify_logic(expr) == lgc.Literal(5)


@pytest.mark.parametrize(
    "build,ids",
    [
        (lambda x: x * ConstantScalar(1), "drops_identity"),
        (lambda x: x + ConstantScalar(0), "drops_zero_addend"),
    ],
    ids=["drops_identity", "drops_zero_addend"],
)
def test_logic_simplify_drops_identity_operands(build, ids):
    """An identity operand disappears together with the MapJoin holding it."""
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    x = finch.asarray(arr)
    capture, ctx = _capturing_scheduler()
    out = finch.compute(build(finch.defer(x)), ctx=ctx)

    np.testing.assert_array_equal(out.to_numpy(), arr)
    assert not _mapjoins(capture.last_prgm)


def test_logic_simplify_leaves_annihilators_to_notation():
    """
    Dropping an annihilated operand would drop the fields that give the result
    its extent, so the MapJoin has to survive: `A * 0` must still be 2x3, not
    the 1x1 that annihilating here produces.
    """
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    x = finch.asarray(arr)
    capture, ctx = _capturing_scheduler()
    out = finch.compute(finch.defer(x) * ConstantScalar(0), ctx=ctx)

    assert out.to_numpy().shape == arr.shape
    np.testing.assert_array_equal(out.to_numpy(), arr * 0)
    assert _mapjoins(capture.last_prgm)


class _CaptureAssembly(UnvalidatedForm, AssemblyLoader):
    """Records the assembly a kernel is built from, after all transforms."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.last: asm.Module

    def lower(self, prgm: asm.Module):
        self.last = prgm
        return self.ctx(prgm)


def _assembly_for(build):
    capture = _CaptureAssembly(AssemblyInterpreter())
    ctx = LogicNormalizer(
        LogicExecutor(
            DefaultLogicOptimizer(
                LogicSimplify(
                    DefaultLoopOrderer(
                        DefaultLogicFormatter(
                            LogicCompiler(
                                NotationCompiler(
                                    capture,
                                    ctx_transforms=(
                                        LowerPackedStructSlots(),
                                        AssemblySimplify(),
                                    ),
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    out = finch.compute(build(finch.defer(finch.asarray(arr)), arr), ctx=ctx)
    return str(capture.last), out.to_numpy(), arr


def test_annihilator_empties_the_loop_body():
    """
    The payoff of a compile-time constant: `A * 0` keeps its loop nest, but the
    body loses the multiply and the load of `A`, so the pass over `A` is gone.
    A runtime scalar of the same value cannot be simplified away.
    """
    # The remaining `mul`s are stride arithmetic; `mul(load(` is the one that
    # multiplies an element of `A`.
    code, out, arr = _assembly_for(lambda x, arr: x * ConstantScalar(0))
    np.testing.assert_array_equal(out, arr * 0)
    assert "mul(load(" not in code

    runtime_code, runtime_out, _ = _assembly_for(lambda x, arr: x * 0)
    np.testing.assert_array_equal(runtime_out, arr * 0)
    assert "mul(load(" in runtime_code
