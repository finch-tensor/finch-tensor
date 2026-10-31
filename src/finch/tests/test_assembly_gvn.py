import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
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
    AssemblySimplify,
    LowerPackedStructSlots,
    assembly_check_types,
    assembly_copy_propagation,
    assembly_dataflow_analyze,
    assembly_dead_code_elimination,
    assembly_global_value_numbering,
)
from finch.finch_logic import LogicSimplify

from .scripts.nodes import (
    create_asm_comprehensive_node,
    create_asm_gvn_node,
    create_asm_if_node,
)
from .test_simplification import _CaptureAssembly
from .utils import bump


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
    """run gvn, and return it with `vn(stmt, expr)`, the vn of `expr` before `stmt`"""
    ctx, _ = assembly_dataflow_analyze(prgm, AssemblyGlobalValueNumbering)
    states = {id(stmt.stmt): state for stmt, state in ctx.statement_states()}

    def vn(stmt, expr):
        return ctx.value_number(states[id(stmt)]["classes"], expr)

    return ctx, vn


def test_gvn_straight_line():
    a, b, c, t, u, v, w = variables("a b c t u v w")
    s_v = asm.Assign(v, add(c, b))
    ret = asm.Return(w)
    _, vn = analyze(
        module(
            (a, b),
            asm.Assign(t, add(a, b)),
            asm.Assign(u, add(a, b)),
            asm.Assign(c, a),
            s_v,
            asm.Assign(a, mul(a, b)),
            asm.Assign(w, add(a, b)),
            ret,
        )
    )
    assert vn(s_v, c) == vn(s_v, a)
    assert vn(ret, t) == vn(ret, u) == vn(ret, v)
    assert vn(ret, a) != vn(ret, c)
    assert vn(ret, w) not in (None, vn(ret, t))


def test_gvn_branches():
    a, b, c, x, y, r = variables("a b c x y r")
    s_r = asm.Assign(r, add(x, y))
    _, vn = analyze(
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
    assert vn(s_r, x) == vn(s_r, add(a, b))
    assert vn(s_r, y) not in (vn(s_r, mul(a, b)), vn(s_r, call(ffuncs.sub, a, b)))


def test_gvn_branches_share_merged_values():
    """a value merged at a join is the same in both branches of the next if"""
    a, c, x, y, z = variables("a c x y z")
    s_z = asm.Assign(z, x)
    _, vn = analyze(
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
    assert vn(s_z, y) == vn(s_z, add(x, i64(1)))


def test_gvn_branches_relate_merged_values():
    """x and y hold different values on each path, but equal ones"""
    c, x, y, r = variables("c x y r")
    s_r = asm.Assign(r, add(x, i64(1)))
    _, vn = analyze(
        module(
            (c,),
            asm.IfElse(
                lt(c, i64(0)),
                asm.Block((asm.Assign(x, i64(1)), asm.Assign(y, i64(1)))),
                asm.Block((asm.Assign(x, i64(2)), asm.Assign(y, i64(2)))),
            ),
            s_r,
            asm.Return(add(r, add(y, i64(1)))),
        )
    )
    assert vn(s_r, x) == vn(s_r, y)
    assert vn(s_r, add(x, i64(1))) == vn(s_r, add(y, i64(1)))
    assert vn(s_r, x) != vn(s_r, i64(1))


def test_gvn_loop_invariant_and_variant():
    a, b, n, t, u, i, acc = variables("a b n t u i acc")
    s_u = asm.Assign(u, add(a, b))
    ret = asm.Return(acc)
    _, vn = analyze(
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
    assert vn(s_u, t) == vn(s_u, add(a, b))
    assert vn(s_u, i) != vn(s_u, i64(0))
    assert vn(ret, t) == vn(ret, add(a, b))


def test_gvn_loop_carried_values_are_not_reused():
    n, i, prev, cur = variables("n i prev cur")
    probe = asm.Assert(lt(prev, cur))
    _, vn = analyze(
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
    assert vn(probe, cur) == vn(probe, mul(i, i64(2)))
    assert vn(probe, prev) != vn(probe, cur)


def test_gvn_nested_loops_with_break():
    a, b, n, t, u, i, j = variables("a b n t u i j")
    s_u = asm.Assign(u, add(a, b))
    _, vn = analyze(
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
    assert vn(s_u, t) == vn(s_u, add(a, b))
    assert vn(s_u, i) != vn(s_u, i64(0))


def test_gvn_early_return_keeps_facts():
    a, b, t, u = variables("a b t u")
    s_u = asm.Assign(u, add(a, b))
    _, vn = analyze(
        module(
            (a, b),
            asm.Assign(t, add(a, b)),
            asm.If(lt(a, i64(0)), asm.Block((asm.Return(i64(0)),))),
            s_u,
            asm.Return(u),
        )
    )
    assert vn(s_u, t) == vn(s_u, add(a, b))


def test_gvn_impure_calls_are_not_numbered():
    a, b, x, y = variables("a b x y")
    ret = asm.Return(add(x, y))
    _, vn = analyze(
        module(
            (a, b),
            asm.Assign(x, call(bump, a, b)),
            asm.Assign(y, call(bump, a, b)),
            ret,
        )
    )
    assert vn(ret, x) != vn(ret, y)
    assert vn(ret, call(bump, a, b)) is None


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
    _, vn = analyze(
        module(
            (i, a),
            *(asm.Assign(names[key], expr) for key, expr in exprs.items()),
            ret,
        )
    )
    assert vn(ret, names["pos"]) != vn(ret, names["neg"])
    assert vn(ret, names["i32"]) != vn(ret, names["i64"])
    assert vn(ret, names["choose_pos"]) != vn(ret, names["choose_neg"])
    assert vn(ret, names["cast_1"]) == vn(ret, names["cast_2"])


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
    ret = asm.Return(x)
    _, vn = analyze(
        module((n,), asm.Assign(x, i64(0)), asm.Assign(y, i64(1)), body, ret)
    )
    assert vn(ret, x) != vn(ret, y)


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


def test_gvn_rewrite_nested_loops():
    """the inner loop reuses a value of the outer body"""
    n, m, i, j, e, u, acc = variables("n m i j e u acc")

    def make(inner):
        return module(
            (n, m),
            asm.Assign(acc, i64(0)),
            asm.ForLoop(
                i,
                i64(0),
                n,
                asm.Block(
                    (
                        asm.Assign(e, mul(i, i64(2))),
                        asm.ForLoop(
                            j,
                            i64(0),
                            m,
                            asm.Block(
                                (asm.Assign(u, inner), asm.Assign(acc, add(acc, u)))
                            ),
                        ),
                    )
                ),
            ),
            asm.Return(acc),
        )

    check_rewrite(
        make(mul(i, i64(2))),
        make(e),
        (np.int64(3), np.int64(2)),
        (np.int64(0), np.int64(4)),
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


def test_asm_dce_after_gvn(file_regression):
    """gvn, then copy propagation, then dce drops the copies gvn left behind"""
    prgm = create_asm_gvn_node()
    result = assembly_dead_code_elimination(
        assembly_copy_propagation(assembly_global_value_numbering(prgm))
    )
    assembly_check_types(result)
    for args in [(1, 2, 3), (5, -2, 0)]:
        args = tuple(np.int64(arg) for arg in args)
        assert AssemblyInterpreter()(result).gvn(*args) == (
            AssemblyInterpreter()(prgm).gvn(*args)
        )
    file_regression.check(str(result), extension=".txt")
