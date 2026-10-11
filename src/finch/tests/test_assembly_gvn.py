import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.algebra import FinchOperator
from finch.autoschedule import (
    CompilerFormLowerer,
    DefaultLogicFactorizer,
    DefaultLogicFormatter,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)
from finch.codegen import NumpyBuffer
from finch.compile import NotationCompiler
from finch.finch_assembly import (
    AssemblyGlobalValueNumbering,
    AssemblyGlobalValueNumberingTransform,
    AssemblyInterpreter,
    AssemblyLoader,
    AssemblySimplify,
    LowerPackedStructSlots,
    assembly_check_types,
    assembly_dataflow_analyze,
    assembly_global_value_numbering,
)
from finch.finch_logic import LogicSimplify
from finch.symbolic import UnvalidatedForm

from .scripts.nodes import (
    create_asm_comprehensive_node,
    create_asm_gvn_node,
    create_asm_if_node,
)


def call(op, *args):
    return asm.Call(asm.Literal(op), args)


def add(x, y):
    return call(ffuncs.add, x, y)


def mul(x, y):
    return call(ffuncs.mul, x, y)


def lt(x, y):
    return call(ffuncs.lt, x, y)


def i64(val):
    return asm.Literal(np.int64(val))


def variables(names, type_=finch.int64):
    return [asm.Variable(name, type_) for name in names.split()]


def module(args, *body, ret=finch.int64):
    arg_types = tuple(arg.result_type for arg in args)
    # fixed definition, so modules built from the same parts compare equal
    func_t = asm.AssemblyKernelFType("f", arg_types, ret, definition="f")
    func = asm.Variable("f", func_t)
    return asm.Module((asm.Function(func, tuple(args), asm.Block(body)),))


def analyze(prgm):
    """run gvn and map each original statement (by `id`) to the state before it"""
    ctx, _ = assembly_dataflow_analyze(prgm, AssemblyGlobalValueNumbering)
    return ctx, {id(stmt.stmt): state for stmt, state in ctx.statement_states()}


def vn(state, name):
    return state.get(("var", name))


def value(ctx, state, expr):
    """vn of `expr` in `state`, or None"""
    return ctx.eval(expr, dict(state), None, None)


def kind(ctx, number):
    return next(key[0] for key, other in ctx.table.items() if other == number)


def test_gvn_straight_line():
    a, b, c, t, u, v, w = variables("a b c t u v w")
    s_t = asm.Assign(t, add(a, b))
    ret = asm.Return(w)
    ctx, states = analyze(
        module(
            (a, b),
            s_t,
            asm.Assign(u, add(a, b)),
            asm.Assign(c, a),
            asm.Assign(v, add(c, b)),
            asm.Assign(a, mul(a, b)),
            asm.Assign(w, add(a, b)),
            ret,
        )
    )
    final = states[id(ret)]
    assert vn(final, "t") == vn(final, "u") == vn(final, "v")
    assert vn(final, "c") == vn(states[id(s_t)], "a")
    assert vn(final, "a") != vn(final, "c")
    assert vn(final, "w") not in (None, vn(final, "t"))


def test_gvn_branches():
    a, b, c, x, y, r = variables("a b c x y r")
    s_r = asm.Assign(r, add(x, y))
    ctx, states = analyze(
        module(
            (a, b, c),
            asm.Assign(x, i64(0)),
            asm.IfElse(
                lt(c, i64(0)),
                asm.Block((asm.Assign(x, add(a, b)), asm.Assign(y, mul(a, b)))),
                asm.Block(
                    (asm.Assign(x, add(a, b)), asm.Assign(y, call(ffuncs.sub, a, b)))
                ),
            ),
            s_r,
            asm.Return(r),
        )
    )
    joined = states[id(s_r)]
    assert vn(joined, "x") == value(ctx, joined, add(a, b))
    assert kind(ctx, vn(joined, "y")) == "phi"


def test_gvn_branches_share_merged_values():
    """a value merged at a join has one name in both branches of the next if"""
    a, c, x, y, z = variables("a c x y z")
    s_z = asm.Assign(z, x)
    ctx, states = analyze(
        module(
            (a, c),
            asm.Assign(x, i64(0)),
            asm.If(lt(c, i64(0)), asm.Assign(x, a)),
            asm.IfElse(
                lt(x, a),
                asm.Assign(y, add(x, i64(1))),
                asm.Assign(y, add(x, i64(1))),
            ),
            s_z,
            asm.Return(z),
        )
    )
    after = states[id(s_z)]
    assert vn(after, "y") == value(ctx, after, add(x, i64(1)))


def test_gvn_loop_invariant_and_variant():
    a, b, n, t, u, i, acc = variables("a b n t u i acc")
    s_u = asm.Assign(u, add(a, b))
    ret = asm.Return(acc)
    ctx, states = analyze(
        module(
            (a, b, n),
            asm.Assign(t, add(a, b)),
            asm.Assign(i, i64(0)),
            asm.Assign(acc, i64(0)),
            asm.WhileLoop(
                lt(i, n),
                asm.Block(
                    (
                        s_u,
                        asm.Assign(acc, add(acc, u)),
                        asm.Assign(i, add(i, i64(1))),
                    )
                ),
            ),
            ret,
        )
    )
    body = states[id(s_u)]
    assert vn(body, "t") == value(ctx, body, add(a, b))
    assert kind(ctx, vn(body, "i")) == "phi"
    assert vn(states[id(ret)], "t") == vn(body, "t")


def test_gvn_loop_carried_values_are_not_reused():
    n, i, prev, cur = variables("n i prev cur")
    probe = asm.Assert(lt(prev, cur))
    ctx, states = analyze(
        module(
            (n,),
            asm.Assign(prev, i64(0)),
            asm.Assign(cur, i64(0)),
            asm.ForLoop(
                i,
                i64(0),
                n,
                asm.Block(
                    (asm.Assign(prev, cur), asm.Assign(cur, mul(i, i64(2))), probe)
                ),
            ),
            asm.Return(prev),
        )
    )
    state = states[id(probe)]
    assert vn(state, "cur") == value(ctx, state, mul(i, i64(2)))
    assert vn(state, "prev") != vn(state, "cur")


def test_gvn_nested_loops_with_break():
    a, b, n, t, u, i, j = variables("a b n t u i j")
    s_u = asm.Assign(u, add(a, b))
    ctx, states = analyze(
        module(
            (a, b, n),
            asm.Assign(t, add(a, b)),
            asm.Assign(i, i64(0)),
            asm.WhileLoop(
                lt(i, n),
                asm.Block(
                    (
                        asm.Assign(j, i64(0)),
                        asm.WhileLoop(
                            lt(j, n),
                            asm.Block(
                                (
                                    asm.If(
                                        call(ffuncs.eq, j, i64(3)),
                                        asm.Block((asm.Break(),)),
                                    ),
                                    asm.Assign(j, add(j, i64(1))),
                                )
                            ),
                        ),
                        asm.Assign(i, add(i, i64(1))),
                    )
                ),
            ),
            s_u,
            asm.Return(u),
        )
    )
    after = states[id(s_u)]
    assert vn(after, "t") == value(ctx, after, add(a, b))
    assert kind(ctx, vn(after, "i")) == "phi"


def test_gvn_early_return_keeps_facts():
    a, b, t, u = variables("a b t u")
    s_u = asm.Assign(u, add(a, b))
    ctx, states = analyze(
        module(
            (a, b),
            asm.Assign(t, add(a, b)),
            asm.If(lt(a, i64(0)), asm.Block((asm.Return(i64(0)),))),
            s_u,
            asm.Return(u),
        )
    )
    state = states[id(s_u)]
    assert vn(state, "t") == value(ctx, state, add(a, b))


class _Bump(FinchOperator):
    """an operator outside `ffuncs`, so it may have effects"""

    @property
    def ftype(self):
        return ffuncs.add.ftype

    def __call__(self, *args):
        return ffuncs.add(*args)


def test_gvn_impure_calls_are_not_numbered():
    a, b, x, y = variables("a b x y")
    ret = asm.Return(add(x, y))
    bump = _Bump()
    ctx, states = analyze(
        module(
            (a, b),
            asm.Assign(x, call(bump, a, b)),
            asm.Assign(y, call(bump, a, b)),
            ret,
        )
    )
    final = states[id(ret)]
    assert vn(final, "x") != vn(final, "y")
    assert kind(ctx, vn(final, "x")) == "def"


def test_gvn_operator_and_literal_keys():
    (i,) = variables("i")
    (a,) = variables("a", finch.float64)
    exprs = {
        "pos": add(a, asm.Literal(np.float64(0.0))),
        "neg": add(a, asm.Literal(np.float64(-0.0))),
        "i32": add(i, asm.Literal(np.int32(1))),
        "i64": add(i, i64(1)),
        "choose_pos": call(ffuncs.choose(0.0), a, a),
        "choose_neg": call(ffuncs.choose(-0.0), a, a),
        "cast_1": call(ffuncs.astype(finch.int64), a),
        "cast_2": call(ffuncs.astype(finch.int64), a),
    }
    names = {key: asm.Variable(key, expr.result_type) for key, expr in exprs.items()}
    ret = asm.Return(i)
    ctx, states = analyze(
        module(
            (i, a),
            *(asm.Assign(names[key], expr) for key, expr in exprs.items()),
            ret,
        )
    )
    final = states[id(ret)]
    assert vn(final, "pos") != vn(final, "neg")
    assert vn(final, "i32") != vn(final, "i64")
    assert vn(final, "choose_pos") != vn(final, "choose_neg")
    assert vn(final, "cast_1") == vn(final, "cast_2")


def test_gvn_nested_loops_terminate():
    n, x, y, t = variables("n x y t")
    body = asm.Block(
        (asm.Assign(t, x), asm.Assign(x, y), asm.Assign(y, add(t, i64(1))))
    )
    for depth in range(5):
        index = asm.Variable(f"i{depth}", finch.int64)
        body = asm.Block(
            (asm.Assign(t, y), asm.ForLoop(index, i64(0), n, body), asm.Assign(y, t))
        )
    ctx, _ = analyze(
        module((n,), asm.Assign(x, i64(0)), asm.Assign(y, i64(1)), body, asm.Return(x))
    )
    assert len(ctx.labels) < 200


def test_asm_gvn_debug_if(file_regression):
    ctx, _ = analyze(create_asm_if_node())
    file_regression.check(str(ctx), extension=".txt")


def test_asm_gvn_debug_redundant(file_regression):
    ctx, _ = analyze(create_asm_gvn_node())
    file_regression.check(str(ctx), extension=".txt")


def check_rewrite(prgm, expected, *inputs):
    """gvn turns `prgm` into `expected`, which type checks and computes the same"""
    result = assembly_global_value_numbering(prgm)
    assert result == expected
    assembly_check_types(result)
    assert assembly_global_value_numbering(result) == result
    for args in inputs:
        assert AssemblyInterpreter()(result).f(*args) == (
            AssemblyInterpreter()(prgm).f(*args)
        )


def test_gvn_rewrite_reuses_values():
    a, b, t, u, v = variables("a b t u v")

    def make(redundant, expr):
        return module(
            (a, b),
            asm.Assign(t, add(a, b)),
            asm.Assign(u, redundant),
            asm.Assign(v, mul(expr, a)),
            asm.Assert(lt(expr, i64(100))),
            asm.Return(add(expr, v)),
        )

    check_rewrite(
        make(add(a, b), add(a, b)),
        make(t, t),
        (np.int64(1), np.int64(2)),
        (np.int64(-4), np.int64(9)),
    )


def test_gvn_rewrite_store():
    p, q, x = variables("p q x", finch.float64)
    buf_t = NumpyBuffer(np.zeros(2)).ftype
    buf = asm.Variable("buf", buf_t)
    buf_slt = asm.Slot("buf_", buf_t)

    def make(value):
        return module(
            (p, q, buf),
            asm.Assign(x, mul(p, q)),
            asm.Unpack(buf_slt, buf),
            asm.Store(buf_slt, asm.Literal(np.intp(1)), value),
            asm.Repack(buf_slt),
            asm.Return(x),
            ret=finch.float64,
        )

    prgm, expected = make(mul(p, q)), make(x)
    result = assembly_global_value_numbering(prgm)
    assert result == expected
    for module_ in (prgm, result):
        out = NumpyBuffer(np.zeros(2))
        AssemblyInterpreter()(module_).f(np.float64(3.0), np.float64(4.0), out)
        np.testing.assert_array_equal(out.arr, [0.0, 12.0])


def test_gvn_rewrite_respects_scopes():
    """
    `t` (declared in both branches) and `w` (in a nested block) aren't visible
    after them, `x` (declared before the if) is.
    """
    a, b, c, t, u, w, x = variables("a b c t u w x")

    def make(branch_x, product, total):
        return module(
            (a, b, c),
            asm.Assign(x, i64(0)),
            asm.IfElse(
                lt(c, i64(0)),
                asm.Block((asm.Assign(t, add(a, b)), asm.Assign(x, branch_x))),
                asm.Block((asm.Assign(t, add(a, b)), asm.Assign(x, branch_x))),
            ),
            asm.Block((asm.Assign(w, mul(a, b)),)),
            asm.Assign(u, add(product, total)),
            asm.Return(u),
        )

    check_rewrite(
        make(add(a, b), mul(a, b), add(a, b)),
        make(t, mul(a, b), x),
        (np.int64(2), np.int64(3), np.int64(-1)),
        (np.int64(2), np.int64(3), np.int64(1)),
    )


def test_gvn_rewrite_loop_invariant():
    a, b, n, t, e, u, i, acc = variables("a b n t e u i acc")

    def make(inner, after):
        return module(
            (a, b, n),
            asm.Assign(t, add(a, b)),
            asm.Assign(acc, i64(0)),
            asm.ForLoop(
                i,
                i64(0),
                n,
                asm.Block((asm.Assign(e, inner), asm.Assign(acc, add(acc, mul(e, i))))),
            ),
            asm.Assign(u, after),
            asm.Return(add(acc, u)),
        )

    check_rewrite(
        make(add(a, b), add(a, b)),
        make(t, t),
        (np.int64(1), np.int64(2), np.int64(4)),
        (np.int64(1), np.int64(2), np.int64(0)),
    )


def test_gvn_rewrite_loop_variables():
    """
    neither the loop index `i` (named like an outer `i`) nor the loop-local `e`
    can stand in for `add(a, b)`.
    """
    a, b, n, e, u, i, acc = variables("a b n e u i acc")
    prgm = module(
        (a, b, n),
        asm.Assign(i, add(a, b)),
        asm.Assign(acc, i64(0)),
        asm.ForLoop(
            i,
            i64(0),
            n,
            asm.Block((asm.Assign(e, add(a, b)), asm.Assign(acc, add(acc, e)))),
        ),
        asm.Assign(u, add(a, b)),
        asm.Return(add(acc, u)),
    )
    assert assembly_global_value_numbering(prgm) == prgm


def test_gvn_rewrite_leaves_dead_code():
    a, b, t, u = variables("a b t u")
    prgm = module(
        (a, b),
        asm.Assign(t, add(a, b)),
        asm.Return(t),
        asm.Assign(u, add(a, b)),
    )
    assert assembly_global_value_numbering(prgm) == prgm


def test_asm_gvn_if(file_regression):
    prgm = assembly_global_value_numbering(create_asm_if_node())
    file_regression.check(str(prgm), extension=".txt")


def test_asm_gvn_comprehensive(file_regression):
    prgm = assembly_global_value_numbering(create_asm_comprehensive_node())
    file_regression.check(str(prgm), extension=".txt")


def test_asm_gvn_redundant(file_regression):
    prgm = create_asm_gvn_node()
    result = assembly_global_value_numbering(prgm)
    assert assembly_global_value_numbering(result) == result
    for args in [(1, 2, 3), (5, -2, 0)]:
        args = tuple(np.int64(arg) for arg in args)
        assert AssemblyInterpreter()(result).gvn(*args) == (
            AssemblyInterpreter()(prgm).gvn(*args)
        )
    file_regression.check(str(result), extension=".txt")


class _CaptureAssembly(UnvalidatedForm, AssemblyLoader):
    """records the assembly a kernel is built from, after all transforms"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.last: asm.Module

    def lower(self, term: asm.Module, *, mode=None):
        self.last = term
        return self.ctx(term, mode=mode)


def test_gvn_pipeline_dense():
    capture = _CaptureAssembly(AssemblyInterpreter())
    ctx = LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(
                LogicSimplify(
                    DefaultLoopOrderer(
                        DefaultLogicFormatter(
                            CompilerFormLowerer(
                                LogicCompiler(
                                    NotationCompiler(
                                        capture,
                                        ctx_transforms=(
                                            LowerPackedStructSlots(),
                                            AssemblySimplify(),
                                            AssemblyGlobalValueNumberingTransform(),
                                        ),
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    y = np.arange(12, dtype=np.float64).reshape(3, 4)
    a, b = finch.defer(finch.asarray(x)), finch.defer(finch.asarray(y))

    out = finch.compute(a @ b, ctx=ctx)
    np.testing.assert_allclose(out.to_numpy(), x @ y)
    assembly_check_types(capture.last)

    out = finch.compute(a * 2 + a, ctx=ctx)
    np.testing.assert_allclose(out.to_numpy(), x * 2 + x)
    assembly_check_types(capture.last)
