import pytest

import numpy as np

import finch
import finch.finch_notation as ntn
from finch.algebra import ffuncs

a, b, x, y, z, n, i = (
    ntn.Variable(name, finch.int64) for name in ("a", "b", "x", "y", "z", "n", "i")
)


def lit(value):
    return ntn.Literal(np.int64(value))


def call(op, *args):
    return ntn.Call(ntn.Literal(op), args)


def add(left, right):
    return call(ffuncs.add, left, right)


def program(args, *body, name="f", result=finch.int64):
    return ntn.Module(
        (
            ntn.Function(
                ntn.Variable(name, result),
                args,
                ntn.Block(body),
            ),
        )
    )


def optimize(prgm):
    return ntn.notation_global_value_numbering(
        prgm,
        assume_pure_arithmetic=True,
        assume_value_semantics=True,
    )


def statements(prgm):
    return prgm.funcs[0].body.bodies


def compile_notation(prgm, transforms=None):
    import finch.finch_logic as lgc
    from finch.autoschedule.compiler import LogicCompiler
    from finch.autoschedule.stages import LogicNotationLowerer

    class FixedLowerer(LogicNotationLowerer):
        def lower(self, term, bindings, stats, stats_factory):
            return prgm

    class RecordingInterpreter(ntn.NotationInterpreter):
        def lower(self, term):
            self.seen = term
            return super().lower(term)

    loader = RecordingInterpreter()
    compiler = LogicCompiler(
        ctx_load=loader, ctx_lower=FixedLowerer(), ctx_transforms=transforms
    )
    library, *_ = compiler(lgc.Plan((lgc.Produces(()),)), {}, {}, None)
    return loader.seen, library


def check_execution(prgm, optimized, args, expected):
    for term in (prgm, optimized):
        assert (
            ntn.NotationInterpreter()(term).f(*(np.int64(v) for v in args)) == expected
        )


def test_gvn_repeated_arithmetic_and_operand_congruence():
    prgm = program(
        (a, b),
        ntn.Assign(x, a),
        ntn.Assign(y, add(a, b)),
        ntn.Assign(z, add(x, b)),
        ntn.Return(z),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2] == ntn.Assign(z, y)
    check_execution(prgm, optimized, (4, 7), 11)


def test_gvn_structural_matching_does_not_inspect_custom_scalar_types():
    from finch.algebra import FType

    class ScalarType(FType):
        def __hash__(self):
            raise AssertionError("GVN must not hash type annotations")

        def __eq__(self, other):
            raise AssertionError("GVN must not compare type annotations")

        def __call__(self, value):
            return int(value)

        def fisinstance(self, value):
            return type(value) is int

    scalar = ScalarType()
    p, q, r, s = (ntn.Variable(name, scalar) for name in ("p", "q", "r", "s"))
    prgm = program(
        (p, q),
        ntn.Assign(r, add(p, q)),
        ntn.Assign(s, add(p, q)),
        ntn.Return(s),
        result=scalar,
    )
    optimized = ntn.notation_global_value_numbering(
        prgm, assume_pure_arithmetic=True, assume_value_semantics=True
    )
    assert statements(optimized)[1] == ntn.Assign(s, r)
    for term in (prgm, optimized):
        assert ntn.NotationInterpreter()(term).f(1000, 2000) == 3000


def test_gvn_value_semantics_alone_does_not_grant_purity():
    p, q = (ntn.Variable(name, finch.int_) for name in ("p", "q"))
    rhs = add(ntn.Literal(1000), ntn.Literal(2000))
    prgm = program(
        (), ntn.Assign(p, rhs), ntn.Assign(q, rhs), ntn.Return(q), result=finch.int_
    )
    optimized = ntn.notation_global_value_numbering(prgm, assume_value_semantics=True)
    assert statements(optimized)[1].rhs is rhs
    assert ntn.NotationInterpreter()(optimized).f() == 3000


@pytest.mark.parametrize(
    "rhs", [call(ffuncs.sub, a, b), add(b, a), call(ffuncs.mul, a, b)]
)
def test_gvn_preserves_operator_and_operand_order(rhs):
    prgm = program((a, b), ntn.Assign(x, add(a, b)), ntn.Assign(y, rhs), ntn.Return(y))
    assert statements(optimize(prgm))[1].rhs == rhs


@pytest.mark.parametrize("target", [a, x])
def test_gvn_kills_overwritten_operand_or_representative(target):
    prgm = program(
        (a, b),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(target, lit(99)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2].rhs == add(a, b)
    check_execution(prgm, optimized, (4, 7), 106 if target == a else 11)


def test_gvn_rhs_before_self_update():
    prgm = program(
        (x,),
        ntn.Assign(y, add(x, lit(1))),
        ntn.Assign(x, add(x, lit(1))),
        ntn.Return(x),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[1] == ntn.Assign(x, y)
    check_execution(prgm, optimized, (4,), 5)


@pytest.mark.parametrize("condition", [0, 1])
def test_gvn_diamond_preserves_relationship_not_constant(condition):
    prgm = program(
        (n,),
        ntn.Assign(x, lit(0)),
        ntn.Assign(y, lit(0)),
        ntn.IfElse(
            n,
            ntn.Block((ntn.Assign(x, lit(1)), ntn.Assign(y, lit(1)))),
            ntn.Block((ntn.Assign(x, lit(2)), ntn.Assign(y, lit(2)))),
        ),
        ntn.Assign(a, add(x, lit(3))),
        ntn.Assign(b, add(y, lit(3))),
        ntn.Return(b),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[4] == ntn.Assign(b, a)
    check_execution(prgm, optimized, (condition,), 4 if condition else 5)


@pytest.mark.parametrize("condition", [0, 1])
def test_gvn_diamond_common_available_expression(condition):
    prgm = program(
        (a, b, n),
        ntn.Assign(x, lit(0)),
        ntn.IfElse(n, ntn.Assign(x, add(a, b)), ntn.Assign(x, add(a, b))),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2] == ntn.Assign(y, x)
    check_execution(prgm, optimized, (4, 7, condition), 11)


@pytest.mark.parametrize("condition", [0, 1])
def test_gvn_one_arm_write_drops_fact(condition):
    prgm = program(
        (a, b, n),
        ntn.Assign(x, add(a, b)),
        ntn.If(n, ntn.Assign(a, lit(9))),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2].rhs == add(a, b)
    check_execution(prgm, optimized, (4, 7, condition), 16 if condition else 11)


def extent(hi):
    return call(finch.compile.make_extent, lit(0), hi)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_gvn_loop_invariant_and_changing_operand(count):
    prgm = program(
        (a, b, n),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, lit(0)),
        ntn.Loop(
            i,
            extent(n),
            ntn.Block((ntn.Assign(y, add(a, b)), ntn.Assign(a, add(a, lit(1))))),
        ),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert isinstance(statements(optimized)[2].body.bodies[0].rhs, ntn.Call)
    check_execution(prgm, optimized, (4, 7, count), 10 + count if count else 0)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_gvn_loop_invariant_reuse(count):
    prgm = program(
        (a, b, n),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, lit(0)),
        ntn.Loop(i, extent(n), ntn.Assign(y, add(a, b))),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2].body == ntn.Assign(y, x)
    check_execution(prgm, optimized, (4, 7, count), 11 if count else 0)


def test_gvn_branch_local_does_not_escape():
    prgm = program(
        (a, b, n),
        ntn.IfElse(n, ntn.Assign(x, add(a, b)), ntn.Assign(x, add(a, b))),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[1].rhs == add(a, b)
    check_execution(prgm, optimized, (4, 7, 1), 11)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_gvn_index_not_a_self_copy(count):
    prgm = program(
        (n,),
        ntn.Assign(i, lit(10)),
        ntn.Assign(x, add(i, lit(1))),
        ntn.Assign(y, lit(0)),
        ntn.Loop(i, extent(n), ntn.Assign(y, add(i, lit(1)))),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    check_execution(prgm, optimized, (count,), count)


def test_gvn_unknown_calls_preserve_effects_and_break_facts():
    seen = []

    def record(value):
        seen.append(value)
        return value

    op = record
    prgm = program(
        (a, b),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(z, call(op, a)),
        ntn.Assign(z, call(op, a)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[3].rhs == add(a, b)
    check_execution(prgm, optimized, (4, 7), 11)
    assert seen == [4, 4, 4, 4]


def test_gvn_signed_zero_is_not_merged():
    p, q = (ntn.Variable(name, finch.float64) for name in ("p", "q"))
    r, s = (ntn.Variable(name, finch.float64) for name in ("r", "s"))
    one = ntn.Literal(np.float64(1.0))
    prgm = program(
        (),
        ntn.Assign(p, ntn.Literal(np.float64(0.0))),
        ntn.Assign(q, ntn.Literal(np.float64(-0.0))),
        ntn.Assign(r, call(ffuncs.mul, p, one)),
        ntn.Assign(s, call(ffuncs.mul, q, one)),
        ntn.Return(s),
        result=finch.float64,
    )
    optimized = optimize(prgm)
    assert isinstance(statements(optimized)[3].rhs, ntn.Call)
    assert np.signbit(ntn.NotationInterpreter()(optimized).f())


def test_gvn_opaque_extent_can_mutate_between_iterations():
    from finch.compile.lower import Extent

    class ChangingExtent(Extent):
        def loop_interpreter(self, ctx, idx, body):
            ctx.bindings["a"] = np.int64(9)
            super().loop_interpreter(ctx, idx, body)

    prgm = program(
        (a, b),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, lit(0)),
        ntn.Loop(i, ntn.Literal(ChangingExtent(0, 1)), ntn.Assign(y, add(a, b))),
        ntn.Return(y),
    )
    check_execution(prgm, optimize(prgm), (4, 7), 16)


def test_gvn_transform_keeps_redundant_assignments():
    prgm = program(
        (a, b), ntn.Assign(x, add(a, b)), ntn.Assign(y, add(a, b)), ntn.Return(x)
    )
    optimized = ntn.NotationGlobalValueNumberingTransform(
        assume_pure_arithmetic=True,
        assume_value_semantics=True,
    )(prgm)
    assert statements(optimized) == (
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, x),
        ntn.Return(x),
    )


def test_gvn_functions_are_independent():
    first = program((a, b), ntn.Assign(x, add(a, b)), ntn.Return(x))
    second = program((a, b), ntn.Assign(y, add(a, b)), ntn.Return(y), name="g")
    optimized = optimize(ntn.Module(first.funcs + second.funcs))
    assert optimized.funcs[1].body.bodies[0].rhs == add(a, b)


def test_gvn_nested_expression_congruence():
    prgm = program(
        (a, b),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, call(ffuncs.mul, x, b)),
        ntn.Assign(z, call(ffuncs.mul, add(a, b), b)),
        ntn.Return(z),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2] == ntn.Assign(z, y)
    check_execution(prgm, optimized, (4, 7), 77)


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_gvn_loop_swap_preserves_saved_values(count):
    prgm = program(
        (a, b, n),
        ntn.Assign(x, a),
        ntn.Assign(y, b),
        ntn.Assign(z, lit(0)),
        ntn.Loop(
            i,
            extent(n),
            ntn.Block(
                (
                    ntn.Assign(z, x),
                    ntn.Assign(x, y),
                    ntn.Assign(y, z),
                )
            ),
        ),
        ntn.Assign(z, call(ffuncs.sub, x, y)),
        ntn.Return(z),
    )
    check_execution(prgm, optimize(prgm), (4, 7, count), 3 if count % 2 else -3)


@pytest.mark.parametrize("condition", [0, 1])
@pytest.mark.parametrize("entrypoint", ["function", "analysis"])
def test_gvn_preserves_interpreter_execution_after_nested_return(condition, entrypoint):
    from finch.finch_notation.global_value_numbering import NotationGlobalValueNumbering

    prgm = program(
        (n,),
        ntn.Assign(a, lit(4)),
        ntn.Assign(b, lit(7)),
        ntn.Assign(x, add(a, b)),
        ntn.If(n, ntn.Block((ntn.Return(x), ntn.Assign(a, lit(99))))),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    if entrypoint == "function":
        optimized = ntn.notation_global_value_numbering(
            prgm,
            assume_pure_arithmetic=True,
            assume_value_semantics=True,
        )
    else:
        analysis = NotationGlobalValueNumbering(
            prgm.funcs[0],
            assume_pure_arithmetic=True,
            assume_value_semantics=True,
        )
        analysis.analyze()
        optimized = analysis.rewrite()
    assert optimized == prgm
    check_execution(prgm, optimized, (condition,), 106 if condition else 11)


def test_gvn_preserves_interpreter_execution_after_nonterminal_return():
    prgm = program(
        (),
        ntn.Assign(a, lit(4)),
        ntn.Assign(b, lit(7)),
        ntn.Assign(x, add(a, b)),
        ntn.Return(x),
        ntn.Assign(a, lit(99)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert optimized == prgm
    check_execution(prgm, optimized, (), 106)


def test_gvn_join_and_worklist_order_do_not_change_loop_facts():
    from finch.finch_notation.global_value_numbering import NotationGlobalValueNumbering

    prgm = program(
        (a, b, n),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, lit(0)),
        ntn.Loop(i, extent(n), ntn.Assign(y, add(a, b))),
        ntn.Return(y),
    )
    first = NotationGlobalValueNumbering(
        prgm.funcs[0],
        assume_pure_arithmetic=True,
        assume_value_semantics=True,
    )
    second = NotationGlobalValueNumbering(
        prgm.funcs[0],
        assume_pure_arithmetic=True,
        assume_value_semantics=True,
    )
    second.cfg.blocks = dict(reversed(tuple(second.cfg.blocks.items())))
    for block in second.cfg.blocks.values():
        block.predecessors.reverse()
        block.successors.reverse()
    first.analyze()
    second.analyze()
    assert first.input_states == second.input_states
    assert first.output_states == second.output_states
    assert first.rewrite() == second.rewrite()
    assert statements(first.rewrite())[2].body == ntn.Assign(y, x)


def test_gvn_meet_preserves_only_common_equalities():
    from finch.finch_notation.global_value_numbering import NotationGlobalValueNumbering

    analysis = NotationGlobalValueNumbering(
        program((a, b, x, y), ntn.Assign(z, add(x, y)), ntn.Return(z)).funcs[0],
    )
    left = analysis.transfer((ntn.Assign(x, a), ntn.Assign(y, a)), analysis.boundary)
    right = analysis.transfer((ntn.Assign(x, b), ntn.Assign(y, b)), analysis.boundary)
    joined = analysis.join(left, right)
    ids = analysis.universe.variables
    classes = joined["classes"]
    assert classes[ids["x"][0]] == classes[ids["y"][0]]
    assert classes[ids["a"][0]] != classes[ids["x"][0]]
    assert classes[ids["b"][0]] != classes[ids["x"][0]]
    assert analysis.join(left, {}) == left
    assert analysis.join(left, left) == left
    assert joined == analysis.join(right, left)
    assert analysis.join(joined, analysis.boundary) == analysis.join(
        left, analysis.join(right, analysis.boundary)
    )
    assert analysis.transfer((ntn.Assign(x, lit(9)),), joined) != joined
    assert analysis.join(left, right) == joined


def test_gvn_literal_types_are_distinct():
    from finch.finch_notation.global_value_numbering import NotationGlobalValueNumbering

    p = ntn.Variable("p", finch.int_)
    q = ntn.Variable("q", finch.bool_)
    prgm = program(
        (),
        ntn.Assign(p, ntn.Literal(1)),
        ntn.Assign(q, ntn.Literal(True)),
        ntn.Assign(x, add(lit(1), lit(2))),
        ntn.Return(x),
    )
    analysis = NotationGlobalValueNumbering(prgm.funcs[0])
    state = analysis.transfer(statements(prgm), analysis.boundary)
    p_id = analysis.universe.variables["p"][0]
    q_id = analysis.universe.variables["q"][0]
    assert state["classes"][p_id] != state["classes"][q_id]


def test_gvn_tensor_read_and_write_are_not_reused():
    data = finch.asarray(np.array([2], dtype=np.int64))
    tensor = ntn.Variable("tensor", finch.ftype(data))
    slot = ntn.Slot("tensor_slot", finch.ftype(data))
    read = ntn.Unwrap(ntn.Access(slot, ntn.Read(), (lit(0),)))
    update = ntn.Access(slot, ntn.Update(ntn.Literal(ffuncs.overwrite)), (lit(0),))
    prgm = program(
        (tensor,),
        ntn.Unpack(slot, tensor),
        ntn.Assign(x, read),
        ntn.Increment(update, lit(9)),
        ntn.Assign(y, read),
        ntn.Assign(z, add(y, lit(1))),
        ntn.Return(z),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[1].rhs == read
    assert statements(optimized)[3].rhs == read
    for term in (prgm, optimized):
        tensor_data = finch.asarray(np.array([2], dtype=np.int64))
        assert ntn.NotationInterpreter()(term).f(tensor_data) == 10


def test_gvn_unknown_condition_is_a_barrier():
    def op(value):
        return value

    prgm = program(
        (a, b, n),
        ntn.Assign(x, add(a, b)),
        ntn.If(call(op, n), ntn.Block(())),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = optimize(prgm)
    assert statements(optimized)[2].rhs == add(a, b)
    check_execution(prgm, optimized, (4, 7, 1), 11)


@pytest.mark.parametrize("origin", ["input", "literal", "call"])
def test_gvn_preserves_overloaded_scalar_arithmetic(origin):
    seen = []

    class EffectfulInt(np.int64):
        def __add__(self, other):
            seen.append(1)
            return np.int64(int(self) + int(other) + len(seen))

    value = EffectfulInt(4)
    prefix = ()
    if origin == "literal":
        prefix = (ntn.Assign(a, ntn.Literal(value)),)
    elif origin == "call":

        def op():
            return value

        prefix = (ntn.Assign(a, call(op)),)
    prgm = program(
        (a, b),
        *prefix,
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
    )
    optimized = ntn.notation_global_value_numbering(prgm, assume_value_semantics=True)
    for term in (prgm, optimized):
        seen.clear()
        assert ntn.NotationInterpreter()(term).f(value, np.int64(7)) == 13
        assert len(seen) == 2


def test_gvn_does_not_compare_opaque_literal_metaclasses():
    class OpaqueMeta(type):
        def __eq__(cls, other):
            raise RuntimeError("Opaque metaclass comparison")

        __hash__ = type.__hash__

    class OpaqueCallable(metaclass=OpaqueMeta):
        def __call__(self):
            return np.int64(7)

    prgm = program((), ntn.Assign(x, call(OpaqueCallable())), ntn.Return(x))
    assert ntn.NotationInterpreter()(prgm).f() == 7
    optimized = ntn.notation_global_value_numbering(
        prgm, assume_pure_arithmetic=True, assume_value_semantics=True
    )
    assert ntn.NotationInterpreter()(optimized).f() == 7


@pytest.mark.parametrize(
    "entrypoint", ["function", "transform", "analysis", "compiler"]
)
@pytest.mark.parametrize("trust_arithmetic", [False, True])
@pytest.mark.parametrize("escape", [False, True])
def test_gvn_preserves_scalar_identity_without_value_semantics(
    entrypoint, trust_arithmetic, escape
):
    from finch.finch_notation.global_value_numbering import NotationGlobalValueNumbering

    a, b, x, y = (ntn.Variable(name, finch.int_) for name in ("a", "b", "x", "y"))

    def observe(left, right):
        return (left, right) if escape else left is right

    prgm = program(
        (),
        ntn.Assign(a, ntn.Literal(1000)),
        ntn.Assign(b, ntn.Literal(2000)),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(call(observe, x, y)),
        result=finch.ftype((1, 2)) if escape else finch.bool_,
    )
    options = {
        "assume_pure_arithmetic": trust_arithmetic,
    }
    if entrypoint == "function":
        optimized = ntn.notation_global_value_numbering(prgm, **options)
    elif entrypoint == "transform":
        optimized = ntn.NotationGlobalValueNumberingTransform(**options)(prgm)
    elif entrypoint == "analysis":
        analysis = NotationGlobalValueNumbering(prgm.funcs[0], **options)
        analysis.analyze()
        optimized = analysis.rewrite()
    else:
        optimized, library = compile_notation(prgm)
    if entrypoint != "compiler":
        library = ntn.NotationInterpreter()(optimized)
    for executable in (ntn.NotationInterpreter()(prgm), library):
        result = executable.f()
        if escape:
            assert result == (3000, 3000)
            assert result[0] is not result[1]
        else:
            assert result is False


@pytest.mark.parametrize("entrypoint", ["function", "transform", "compiler"])
def test_gvn_explicit_contract_rewrites_literal_computations(entrypoint):
    a, b, x, y = (ntn.Variable(name, finch.int_) for name in ("a", "b", "x", "y"))
    prgm = program(
        (),
        ntn.Assign(a, ntn.Literal(4)),
        ntn.Assign(b, ntn.Literal(7)),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
        result=finch.int_,
    )
    if entrypoint == "function":
        optimized = ntn.notation_global_value_numbering(
            prgm,
            assume_pure_arithmetic=True,
            assume_value_semantics=True,
        )
    else:
        transform = ntn.NotationGlobalValueNumberingTransform(
            assume_pure_arithmetic=True, assume_value_semantics=True
        )
        if entrypoint == "compiler":
            optimized, library = compile_notation(prgm, (transform,))
            assert library.f() == 11
        else:
            optimized = transform(prgm)
    assert statements(optimized)[3] == ntn.Assign(y, x)
    check_execution(prgm, optimized, (), 11)


def test_gvn_preserves_numpy_error_callback_exception():
    a, b, x, y = (ntn.Variable(name, finch.float64) for name in ("a", "b", "x", "y"))
    prgm = program(
        (),
        ntn.Assign(a, ntn.Literal(np.float64(1e308))),
        ntn.Assign(b, ntn.Literal(np.float64(1e308))),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
        result=finch.float64,
    )
    optimized = ntn.notation_global_value_numbering(prgm, assume_value_semantics=True)
    previous = np.geterrcall()
    try:

        def callback(*args):
            np.seterr(over="raise")

        np.seterrcall(callback)
        for term in (prgm, optimized):
            with np.errstate(over="call"), pytest.raises(FloatingPointError):
                ntn.NotationInterpreter()(term).f()
    finally:
        np.seterrcall(previous)


def test_gvn_custom_loop_index_conversion_can_mutate_bindings():
    from finch.algebra import FType
    from finch.symbolic import ScopedDict

    bindings = ScopedDict({"a": 0})

    class EffectfulIndex(FType):
        def __eq__(self, other):
            return self is other

        def __hash__(self):
            return id(self)

        def __call__(self, value):
            bindings["a"] = 99
            return int(value)

    a, b, x, y = (ntn.Variable(name, finch.int_) for name in ("a", "b", "x", "y"))
    idx = ntn.Variable("idx", EffectfulIndex())
    prgm = program(
        (),
        ntn.Assign(a, ntn.Literal(4)),
        ntn.Assign(b, ntn.Literal(7)),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, ntn.Literal(0)),
        ntn.Loop(
            idx,
            call(finch.compile.make_extent, ntn.Literal(0), ntn.Literal(1)),
            ntn.Assign(y, add(a, b)),
        ),
        ntn.Return(y),
        result=finch.int_,
    )
    optimized = ntn.notation_global_value_numbering(prgm, assume_value_semantics=True)
    for term in (prgm, optimized):
        bindings["a"] = 0
        assert ntn.NotationInterpreter(bindings=bindings)(term).f() == 106


def test_gvn_numpy_loop_index_conversion_can_invoke_error_callback():
    from finch.symbolic import ScopedDict

    bindings = ScopedDict({"a": 0})
    a, b, x, y = (ntn.Variable(name, finch.int_) for name in ("a", "b", "x", "y"))
    idx = ntn.Variable("idx", finch.float16)
    prgm = program(
        (),
        ntn.Assign(a, ntn.Literal(4)),
        ntn.Assign(b, ntn.Literal(7)),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(y, ntn.Literal(0)),
        ntn.Loop(
            idx,
            call(finch.compile.make_extent, ntn.Literal(65520), ntn.Literal(65521)),
            ntn.Assign(y, add(a, b)),
        ),
        ntn.Return(y),
        result=finch.int_,
    )
    optimized = ntn.notation_global_value_numbering(prgm, assume_value_semantics=True)
    previous = np.geterrcall()
    try:

        def callback(*args):
            bindings["a"] = 99

        np.seterrcall(callback)
        for term in (prgm, optimized):
            bindings["a"] = 0
            with np.errstate(over="call"):
                assert ntn.NotationInterpreter(bindings=bindings)(term).f() == 106
    finally:
        np.seterrcall(previous)


def test_gvn_requires_purity_for_custom_assignment_validation():
    from finch.algebra import FType
    from finch.symbolic import ScopedDict

    bindings = ScopedDict({"a": 0})

    class EffectfulType(FType):
        def __eq__(self, other):
            return self is other

        def __hash__(self):
            return id(self)

        def __call__(self, value):
            return value

        def fisinstance(self, value):
            bindings["a"] = 99
            return True

    a, b, x, y = (ntn.Variable(name, finch.int_) for name in ("a", "b", "x", "y"))
    custom = ntn.Variable("custom", EffectfulType())
    prgm = program(
        (),
        ntn.Assign(a, ntn.Literal(4)),
        ntn.Assign(b, ntn.Literal(7)),
        ntn.Assign(x, add(a, b)),
        ntn.Assign(custom, ntn.Literal(0)),
        ntn.Assign(y, add(a, b)),
        ntn.Return(y),
        result=finch.int_,
    )
    optimized = ntn.notation_global_value_numbering(prgm, assume_value_semantics=True)
    for term in (prgm, optimized):
        bindings["a"] = 0
        assert ntn.NotationInterpreter(bindings=bindings)(term).f() == 106


def test_gvn_does_not_query_unsupported_tensor_expression_type():
    opaque = ntn.Unwrap(a)
    prgm = program((a,), ntn.Assign(x, add(opaque, lit(1))), ntn.Return(x))
    assert (
        ntn.notation_global_value_numbering(
            prgm, assume_pure_arithmetic=True, assume_value_semantics=True
        )
        == prgm
    )


def test_gvn_preserves_copy_chain_and_unused_literal_assignment():
    prgm = program(
        (a,), ntn.Assign(x, a), ntn.Assign(y, x), ntn.Assign(z, lit(99)), ntn.Return(y)
    )
    assert (
        ntn.NotationGlobalValueNumberingTransform(
            assume_pure_arithmetic=True, assume_value_semantics=True
        )(prgm)
        == prgm
    )


def test_compiler_requires_explicit_value_numbering_transform():
    from finch.autoschedule.compiler import LogicCompiler

    assert LogicCompiler().ctx_transforms == ()
    assert LogicCompiler(ctx_transforms=()).ctx_transforms == ()
    stages = (
        ntn.NotationGlobalValueNumberingTransform(
            assume_pure_arithmetic=True, assume_value_semantics=True
        ),
    )
    assert LogicCompiler(ctx_transforms=stages).ctx_transforms is stages
