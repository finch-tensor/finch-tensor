from collections import namedtuple

import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.algebra import ftype
from finch.finch_assembly import AssemblyReachingDefinitions, assembly_dataflow_analyze

from .scripts.nodes import create_asm_comprehensive_node, create_asm_if_node


def call(op, *args):
    return asm.Call(asm.Literal(op), args)


def i64(val):
    return asm.Literal(np.int64(val))


def variables(names, type_=finch.int64):
    return [asm.Variable(name, type_) for name in names.split()]


def module(args, *body, ret=finch.int64):
    arg_types = tuple(arg.result_type for arg in args)
    func = asm.Variable("f", asm.AssemblyKernelFType("f", arg_types, ret))
    return asm.Module((asm.Function(func, tuple(args), asm.Block(body)),))


def analyze(prgm):
    """run the analysis and number the original statements by `id`"""
    ctx, _ = assembly_dataflow_analyze(prgm, AssemblyReachingDefinitions)
    sids = {id(stmt.stmt): stmt.sid for stmt, _ in ctx.statement_states()}
    return ctx, ctx.use_def_chains(), lambda stmt: sids[id(stmt)]


def test_reaching_definitions_branches_and_kills():
    a, c, x, y = variables("a c x y")
    x_0 = asm.Assign(x, i64(0))
    x_1 = asm.Assign(x, a)
    s_y = asm.Assign(y, call(ffuncs.add, x, i64(1)))
    x_2 = asm.Assign(x, i64(2))
    ret = asm.Return(call(ffuncs.add, x, y))
    ctx, chains, sid = analyze(
        module(
            (a, c),
            x_0,
            asm.If(call(ffuncs.lt, c, i64(0)), x_1),
            s_y,
            x_2,
            ret,
        )
    )
    assert chains[(sid(s_y), "x")] == {sid(x_0), sid(x_1)}
    assert chains[(sid(ret), "x")] == {sid(x_2)}
    assert chains[(sid(ret), "y")] == {sid(s_y)}
    assert ctx.def_use_chains()[sid(x_0)] == {sid(s_y)}


def test_reaching_definitions_loops():
    n, i = variables("n i")
    i_0 = asm.Assign(i, i64(0))
    i_inc = asm.Assign(i, call(ffuncs.add, i, i64(1)))
    ret = asm.Return(i)
    _, chains, sid = analyze(
        module(
            (n,),
            i_0,
            asm.WhileLoop(call(ffuncs.lt, i, n), i_inc),
            ret,
        )
    )
    assert chains[(sid(i_inc), "i")] == {sid(i_0), sid(i_inc)}
    assert chains[(sid(ret), "i")] == {sid(i_0), sid(i_inc)}


def test_reaching_definitions_synthetic_definitions():
    """arguments and loop variables are defined by statements of the cfg itself"""
    n, i, x = variables("n i x")
    s_x = asm.Assign(x, call(ffuncs.add, i, n))
    _, chains, sid = analyze(
        module(
            (n,),
            asm.Assign(x, i64(0)),
            asm.ForLoop(i, i64(0), n, s_x),
            asm.Return(x),
        )
    )
    (arg_def,) = chains[(sid(s_x), "n")]
    (loop_def,) = chains[(sid(s_x), "i")]
    assert arg_def != loop_def
    assert sid(s_x) not in (arg_def, loop_def)


def test_reaching_definitions_setattr_is_partial():
    point = namedtuple("Point", ["x", "y"])(np.float64(1.0), np.float64(2.0))
    p = asm.Variable("p", ftype(point))
    (b,) = variables("b", finch.float64)
    s_set = asm.SetAttr(p, asm.Literal("x"), asm.Literal(np.float64(5.0)))
    s_b = asm.Assign(b, asm.GetAttr(p, asm.Literal("y")))
    _, chains, sid = analyze(module((p,), s_set, s_b, asm.Return(b), ret=finch.float64))
    defs = chains[(sid(s_b), "p")]
    assert sid(s_set) in defs
    assert len(defs) == 2


def test_asm_reaching_definitions_debug_if(file_regression):
    ctx, _, _ = analyze(create_asm_if_node())
    file_regression.check(str(ctx), extension=".txt")


def test_asm_reaching_definitions_debug_comprehensive(file_regression):
    ctx, _, _ = analyze(create_asm_comprehensive_node())
    file_regression.check(str(ctx), extension=".txt")
