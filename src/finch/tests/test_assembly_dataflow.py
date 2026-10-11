from collections import namedtuple

import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.algebra import ftype
from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyCopyPropagation,
    NumberedStatement,
    assembly_check_types,
    assembly_copy_propagation,
    assembly_dataflow_analyze,
    get_vars_in_expr,
)
from finch.finch_assembly.dataflow import AbstractAssemblyDataflow
from finch.symbolic import Namespace

from .scripts.nodes import (
    create_asm_comprehensive_node,
    create_asm_dot_node,
    create_asm_if_node,
)


def test_asm_cfg_printer_if(file_regression):
    prgm = create_asm_if_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx.cfg), extension=".txt")


def test_asm_copy_propagation_debug_if(file_regression):
    prgm = create_asm_if_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx), extension=".txt")


def test_asm_copy_propagation_if(file_regression):
    prgm = create_asm_if_node()
    prgm = assembly_copy_propagation(prgm)
    file_regression.check(str(prgm), extension=".txt")


def test_asm_cfg_printer_dot(file_regression):
    prgm = create_asm_dot_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx.cfg), extension=".txt")


def test_asm_copy_propagation_debug_dot(file_regression):
    prgm = create_asm_dot_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx), extension=".txt")


def test_asm_copy_propagation_dot(file_regression):
    prgm = create_asm_dot_node()
    prgm = assembly_copy_propagation(prgm)
    file_regression.check(str(prgm), extension=".txt")


def test_asm_cfg_printer_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx.cfg), extension=".txt")


def test_asm_copy_propagation_debug_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx), extension=".txt")


def test_asm_copy_propagation_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    prgm = assembly_copy_propagation(prgm)
    file_regression.check(str(prgm), extension=".txt")


def test_namespace_freshen_trailing_underscore():
    namespace = Namespace()
    assert namespace.freshen("A_") == "A_"
    assert namespace.freshen("A_") == "A__2"
    assert namespace.freshen("x_3") == "x_3"


def test_asm_copy_propagation_generated_shapes():
    """generated kernels have names like `A_`, bare statement bodies and bare calls"""
    a = asm.Variable("A_", finch.float64)
    b = asm.Variable("b", finch.float64)
    c = asm.Variable("c", finch.float64)
    i = asm.Variable("i", finch.intp)
    buf_t = NumpyBuffer(np.zeros(3)).ftype
    buf = asm.Variable("buf", buf_t)
    buf_slt = asm.Slot("buf_", buf_t)
    func = asm.Variable(
        "f", asm.AssemblyKernelFType("f", (finch.float64, buf_t), finch.float64)
    )

    def make_prgm(square, update):
        return asm.Module(
            (
                asm.Function(
                    func,
                    (a, buf),
                    asm.Block(
                        (
                            asm.Assign(b, a),
                            asm.Assign(c, square),
                            asm.If(
                                asm.Call(
                                    asm.Literal(ffuncs.lt),
                                    (c, asm.Literal(np.float64(10.0))),
                                ),
                                asm.Assign(c, update),
                            ),
                            asm.Unpack(buf_slt, buf),
                            asm.ForLoop(
                                i,
                                asm.Literal(np.intp(0)),
                                asm.Length(buf_slt),
                                asm.Store(buf_slt, i, c),
                            ),
                            asm.Call(asm.Literal(ffuncs.add), (b, c)),
                            asm.Repack(buf_slt),
                            asm.Return(c),
                        )
                    ),
                ),
            )
        )

    prgm = make_prgm(
        asm.Call(asm.Literal(ffuncs.mul), (b, b)),
        asm.Call(asm.Literal(ffuncs.add), (c, b)),
    )
    expected = make_prgm(
        asm.Call(asm.Literal(ffuncs.mul), (a, a)),
        asm.Call(asm.Literal(ffuncs.add), (c, a)),
    )
    result = assembly_copy_propagation(prgm)
    assert result == expected

    for module in (prgm, result):
        out = NumpyBuffer(np.zeros(3))
        assert asm.AssemblyInterpreter()(module).f(np.float64(3.0), out) == 12.0
        np.testing.assert_array_equal(out.arr, np.full(3, 12.0))


def test_asm_copy_propagation_skips_dead_blocks():
    """the dead block after an early return must not erase `b = a` at the join"""
    a = asm.Variable("a", finch.int64)
    b = asm.Variable("b", finch.int64)
    func = asm.Variable("f", asm.AssemblyKernelFType("f", (finch.int64,), finch.int64))

    def make_prgm(ret):
        return asm.Module(
            (
                asm.Function(
                    func,
                    (a,),
                    asm.Block(
                        (
                            asm.Assign(b, a),
                            asm.If(
                                asm.Call(
                                    asm.Literal(ffuncs.lt),
                                    (b, asm.Literal(np.int64(0))),
                                ),
                                asm.Block((asm.Return(asm.Literal(np.int64(0))),)),
                            ),
                            asm.Return(ret),
                        )
                    ),
                ),
            )
        )

    ctx, _ = assembly_dataflow_analyze(make_prgm(b), AssemblyCopyPropagation)
    dead = {
        block.id
        for block in ctx.cfg.blocks.values()
        if not block.predecessors and block is not ctx.cfg.entry_block
    }
    assert dead
    assert not dead & ctx.reached
    assert assembly_copy_propagation(make_prgm(b)) == make_prgm(a)


def call(op, *args):
    return asm.Call(asm.Literal(op), args)


def variables(names, type_=finch.int64):
    return [asm.Variable(name, type_) for name in names.split()]


def module(args, *body, ret=finch.int64):
    arg_types = tuple(arg.result_type for arg in args)
    func = asm.Variable("f", asm.AssemblyKernelFType("f", arg_types, ret))
    return asm.Module((asm.Function(func, tuple(args), asm.Block(body)),))


def test_asm_copy_propagation_respects_scopes():
    """
    `t` is declared in a branch, so it can't stand in for `x` after the if,
    even when the other branch returns or breaks and only `x = t` reaches it.
    """
    a, c, n, i, s, t, x = variables("a c n i s t x")
    zero, one = asm.Literal(np.int64(0)), asm.Literal(np.int64(1))

    def branch(cond, exit_):
        return asm.IfElse(
            cond,
            asm.Block((asm.Assign(t, call(ffuncs.add, a, one)), asm.Assign(x, t))),
            asm.Block((exit_,)),
        )

    returns = module(
        (a, c),
        asm.Assign(x, a),
        branch(call(ffuncs.lt, zero, c), asm.Return(zero)),
        asm.Return(call(ffuncs.add, x, one)),
    )
    breaks = module(
        (a, n),
        asm.Assign(x, a),
        asm.Assign(s, zero),
        asm.Assign(i, zero),
        asm.WhileLoop(
            call(ffuncs.lt, i, n),
            asm.Block(
                (
                    asm.Assign(i, call(ffuncs.add, i, one)),
                    branch(call(ffuncs.lt, i, asm.Literal(np.int64(3))), asm.Break()),
                    asm.Assign(s, call(ffuncs.add, s, x)),
                )
            ),
        ),
        asm.Return(s),
    )
    for prgm in (returns, breaks):
        result = assembly_copy_propagation(prgm)
        assert result == prgm
        assembly_check_types(result)


def test_asm_copy_propagation_setattr():
    """mlir rebinds the target of a setattr, so `q` stops being a copy of `p`"""
    point_t = ftype(namedtuple("Point", ["x", "y"])(np.float64(1.0), np.float64(2.0)))
    p, q = asm.Variable("p", point_t), asm.Variable("q", point_t)

    def x_of(obj):
        return asm.GetAttr(obj, asm.Literal("x"))

    prgm = module(
        (p,),
        asm.Assign(q, p),
        asm.SetAttr(q, asm.Literal("x"), asm.Literal(np.float64(5.0))),
        asm.Return(call(ffuncs.add, x_of(p), x_of(q))),
        ret=finch.float64,
    )
    assert assembly_copy_propagation(prgm) == prgm


class _VariablesReadLater(AbstractAssemblyDataflow):
    """toy backward analysis: the variables read after each statement"""

    def direction(self):
        return "backward"

    def transfer(self, stmts, state):
        state = dict(state)
        for stmt in reversed(stmts):
            match stmt:
                case NumberedStatement(asm.Assign(_, rhs), _) | asm.Assign(_, rhs):
                    state.update(dict.fromkeys(get_vars_in_expr(rhs), True))
                case NumberedStatement(stmt, _):
                    state.update(dict.fromkeys(get_vars_in_expr(stmt), True))
        return state

    def join(self, state_1, state_2):
        return state_1 | state_2

    def get_lattice_value(self, state, stmt):
        return [("read_later", sorted(state))] if state else []


def test_backward_dataflow_printing():
    """backward analyses annotate each statement with the state after it"""
    a = asm.Variable("a", finch.int64)
    b = asm.Variable("b", finch.int64)
    c = asm.Variable("c", finch.int64)
    func = asm.Variable("f", asm.AssemblyKernelFType("f", (finch.int64,), finch.int64))
    prgm = asm.Module(
        (
            asm.Function(
                func,
                (a,),
                asm.Block(
                    (
                        asm.Assign(b, a),
                        asm.Assign(c, b),
                        asm.Return(c),
                    )
                ),
            ),
        )
    )
    ctx, _ = assembly_dataflow_analyze(prgm, _VariablesReadLater)
    states = {stmt.sid: sorted(state) for stmt, state in ctx.statement_states()}
    assert states == {3: ["a", "b", "c"], 0: ["b", "c"], 1: ["c"], 2: []}
    lines = [line.strip() for line in str(ctx).splitlines()]
    assert "[0] b: finch.int64 = a \t# read_later = ['b', 'c']" in lines
    assert "[2] return c" in lines
