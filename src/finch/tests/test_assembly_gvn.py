import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.algebra import FinchOperator
from finch.finch_assembly import (
    AssemblyGlobalValueNumbering,
    assembly_dataflow_analyze,
)

from .scripts.nodes import create_asm_gvn_node, create_asm_if_node


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
    func = asm.Variable("f", asm.AssemblyKernelFType("f", arg_types, ret))
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
