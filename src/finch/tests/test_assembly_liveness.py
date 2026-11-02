import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.finch_assembly import AssemblyLiveVariables, assembly_dataflow_analyze

from .scripts.nodes import create_asm_comprehensive_node


def call(op, *args):
    return asm.Call(asm.Literal(op), args)


def i64(val):
    return asm.Literal(np.int64(val))


def variables(names):
    return [asm.Variable(name, finch.int64) for name in names.split()]


def module(args, *body):
    arg_types = tuple(arg.result_type for arg in args)
    func = asm.Variable("f", asm.AssemblyKernelFType("f", arg_types, finch.int64))
    return asm.Module((asm.Function(func, tuple(args), asm.Block(body)),))


def live_after(prgm):
    """map each original statement (by `id`) to the names live after it"""
    ctx, _ = assembly_dataflow_analyze(prgm, AssemblyLiveVariables)
    return {id(stmt.stmt): set(state) for stmt, state in ctx.statement_states()}


def test_liveness_straight_line():
    a, b, c = variables("a b c")
    s_a = asm.Assign(a, i64(1))
    s_b = asm.Assign(b, call(ffuncs.add, a, i64(2)))
    s_c = asm.Assign(c, i64(5))
    live = live_after(module((), s_a, s_b, s_c, asm.Return(b)))
    assert live[id(s_a)] == {"a"}
    assert live[id(s_b)] == {"b"}
    assert live[id(s_c)] == {"b"}


def test_liveness_branches_and_conditions():
    a, b, c, x = variables("a b c x")
    s_x = asm.Assign(x, i64(0))
    then_x = asm.Assign(x, a)
    live = live_after(
        module(
            (a, b, c),
            s_x,
            asm.IfElse(call(ffuncs.lt, c, b), then_x, asm.Block(())),
            asm.Return(x),
        )
    )
    # `c` and `b` are read by the condition, `a` by one branch, `x` by the other
    assert live[id(s_x)] == {"a", "b", "c", "x"}
    assert live[id(then_x)] == {"x"}


def test_liveness_loops():
    n, i, s, dead = variables("n i s dead")
    s_inc = asm.Assign(s, call(ffuncs.add, s, i))
    s_dead = asm.Assign(dead, call(ffuncs.mul, i, i64(2)))
    live = live_after(
        module(
            (n,),
            asm.Assign(s, i64(0)),
            asm.ForLoop(i, i64(0), n, asm.Block((s_inc, s_dead))),
            asm.Return(s),
        )
    )
    # `s` is carried around the loop and `n` is re-read by the loop condition
    assert {"s", "n"} <= live[id(s_inc)]
    assert "dead" not in live[id(s_dead)]


def test_asm_liveness_debug_comprehensive(file_regression):
    ctx, _ = assembly_dataflow_analyze(
        create_asm_comprehensive_node(), AssemblyLiveVariables
    )
    file_regression.check(str(ctx), extension=".txt")
