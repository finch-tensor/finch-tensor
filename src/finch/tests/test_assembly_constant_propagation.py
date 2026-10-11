import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyConstantPropagation,
    AssemblyInterpreter,
    assembly_check_types,
    assembly_constant_propagation,
    assembly_dataflow_analyze,
)

from .scripts.nodes import create_asm_comprehensive_node
from .utils import bump


def call(op, *args):
    return asm.Call(asm.Literal(op), args)


def i64(val):
    return asm.Literal(np.int64(val))


def variables(names, type_=finch.int64):
    return [asm.Variable(name, type_) for name in names.split()]


def module(args, *body, ret=finch.int64):
    arg_types = tuple(arg.result_type for arg in args)
    func_t = asm.AssemblyKernelFType("f", arg_types, ret, definition="f")
    func = asm.Variable("f", func_t)
    return asm.Module((asm.Function(func, tuple(args), asm.Block(body)),))


def check_rewrite(prgm, expected, *inputs):
    """propagation turns `prgm` into `expected`, which type checks and agrees"""
    result = assembly_constant_propagation(prgm)
    assert result == expected
    assembly_check_types(result)
    assert assembly_constant_propagation(result) == result
    for args in inputs:
        assert AssemblyInterpreter()(result).f(*args) == (
            AssemblyInterpreter()(prgm).f(*args)
        )


def test_constant_propagation_folds_and_replaces():
    a, x, y, z = variables("a x y z")

    def make(y_rhs, z_rhs, ret):
        return module(
            (a,),
            asm.Assign(x, i64(2)),
            asm.Assign(y, y_rhs),
            asm.Assign(z, z_rhs),
            asm.Assert(call(ffuncs.lt, z_rhs, i64(100))),
            asm.Return(ret),
        )

    check_rewrite(
        make(
            call(ffuncs.add, x, i64(3)),
            call(ffuncs.mul, y, a),
            call(ffuncs.add, z, x),
        ),
        make(i64(5), call(ffuncs.mul, i64(5), a), call(ffuncs.add, z, i64(2))),
        (np.int64(3),),
    )


def test_constant_propagation_joins():
    """branches that agree keep the constant, others (even 0.0 and -0.0) don't"""
    c, x, y = variables("c x y")
    u, v = variables("u v", finch.float64)

    def make(x_use, u_use, ret):
        return module(
            (c,),
            asm.IfElse(
                call(ffuncs.lt, c, i64(0)),
                asm.Block(
                    (
                        asm.Assign(x, i64(1)),
                        asm.Assign(u, asm.Literal(np.float64(0.0))),
                    )
                ),
                asm.Block(
                    (
                        asm.Assign(x, i64(1)),
                        asm.Assign(u, asm.Literal(np.float64(-0.0))),
                    )
                ),
            ),
            asm.Assign(y, x_use),
            asm.Assign(v, u_use),
            asm.Return(ret),
        )

    prgm = make(call(ffuncs.add, x, i64(1)), call(ffuncs.mul, u, u), y)
    expected = make(i64(2), call(ffuncs.mul, u, u), i64(2))
    assert assembly_constant_propagation(prgm) == expected


def test_constant_propagation_loops():
    n, i, k, s = variables("n i k s")

    def make(k_use):
        return module(
            (n,),
            asm.Assign(k, i64(3)),
            asm.Assign(i, i64(0)),
            asm.Assign(s, i64(0)),
            asm.WhileLoop(
                call(ffuncs.lt, i, n),
                asm.Block(
                    (
                        asm.Assign(s, call(ffuncs.add, s, k_use)),
                        asm.Assign(i, call(ffuncs.add, i, i64(1))),
                    )
                ),
            ),
            asm.Return(s),
        )

    check_rewrite(make(k), make(i64(3)), (np.int64(4),), (np.int64(0),))


def test_constant_propagation_folds_division():
    """division, mod and shifts fold like numpy, which the backends follow"""
    x, q, r, s = variables("x q r s")
    (t,) = variables("t", finch.float64)

    def make(q_rhs, r_rhs, s_rhs, t_rhs, ret):
        return module(
            (),
            asm.Assign(x, i64(-7)),
            asm.Assign(q, q_rhs),
            asm.Assign(r, r_rhs),
            asm.Assign(s, s_rhs),
            asm.Assign(t, t_rhs),
            asm.Return(ret),
        )

    check_rewrite(
        make(
            call(ffuncs.floordiv, x, i64(2)),
            call(ffuncs.mod, x, i64(2)),
            call(ffuncs.rshift, x, i64(1)),
            call(ffuncs.truediv, x, i64(2)),
            call(ffuncs.add, q, call(ffuncs.add, r, s)),
        ),
        make(i64(-4), i64(1), i64(-4), asm.Literal(np.float64(-3.5)), i64(-7)),
        (),
    )


def test_constant_propagation_never_folds_runtime_failures():
    """
    overflow, negative integer powers and division by zero stay at runtime, and
    keep their variable operands so no backend compiler folds them its own way.
    """
    big, seven, zero, x, y, z = variables("big seven zero x y z")

    def make(ret):
        return module(
            (),
            asm.Assign(big, i64(2**63 - 1)),
            asm.Assign(seven, i64(7)),
            asm.Assign(zero, i64(0)),
            asm.Assign(x, call(ffuncs.add, big, i64(1))),
            asm.Assign(y, call(ffuncs.pow, i64(2), i64(-1))),
            asm.Assign(z, call(ffuncs.floordiv, seven, zero)),
            asm.Return(ret),
        )

    assert assembly_constant_propagation(make(big)) == make(i64(2**63 - 1))


def test_constant_propagation_skips_impure_calls():
    """a call is only folded when its operator's ftype says it is pure"""
    x, y = variables("x y")
    prgm = module(
        (),
        asm.Assign(x, call(bump, i64(1), i64(2))),
        asm.Assign(y, call(ffuncs.add, x, i64(1))),
        asm.Return(y),
    )
    assert assembly_constant_propagation(prgm) == prgm
    assert ffuncs.add.ftype.is_pure
    assert not bump.ftype.is_pure


def test_constant_propagation_keeps_types():
    x, y = variables("x y", finch.int32)
    one, two = asm.Literal(np.int32(1)), asm.Literal(np.int32(2))
    prgm = module(
        (),
        asm.Assign(x, one),
        asm.Assign(y, call(ffuncs.add, x, x)),
        asm.Return(y),
        ret=finch.int32,
    )
    result = assembly_constant_propagation(prgm)
    assembly_check_types(result)
    # literals compare by value only, so check the folded type too
    match result:
        case asm.Module((asm.Function(_, _, asm.Block((_, y_def, ret))),)):
            assert y_def == asm.Assign(y, two)
            assert ret == asm.Return(two)
            match y_def:
                case asm.Assign(_, asm.Literal(val)):
                    assert type(val) is np.int32


def test_constant_propagation_buffers():
    """indices and values become literals, buffer operands stay slots"""
    buf_t = NumpyBuffer(np.zeros(2)).ftype
    buf = asm.Variable("buf", buf_t)
    buf_slt = asm.Slot("buf_", buf_t)
    (i,) = variables("i", buf_t.length_type)
    (v,) = variables("v", finch.float64)

    def make(idx, val):
        return module(
            (buf,),
            asm.Assign(i, asm.Literal(np.intp(1))),
            asm.Assign(v, asm.Literal(np.float64(4.0))),
            asm.Unpack(buf_slt, buf),
            asm.Store(buf_slt, idx, val),
            asm.Repack(buf_slt),
            asm.Return(val),
            ret=finch.float64,
        )

    prgm = make(i, v)
    result = assembly_constant_propagation(prgm)
    assert result == make(asm.Literal(np.intp(1)), asm.Literal(np.float64(4.0)))
    for module_ in (prgm, result):
        out = NumpyBuffer(np.zeros(2))
        assert AssemblyInterpreter()(module_).f(out) == 4.0
        np.testing.assert_array_equal(out.arr, [0.0, 4.0])


def test_asm_constant_propagation_debug_comprehensive(file_regression):
    ctx, _ = assembly_dataflow_analyze(
        create_asm_comprehensive_node(), AssemblyConstantPropagation
    )
    file_regression.check(str(ctx), extension=".txt")


def test_asm_constant_propagation_comprehensive(file_regression):
    prgm = assembly_constant_propagation(create_asm_comprehensive_node())
    assembly_check_types(prgm)
    file_regression.check(str(prgm), extension=".txt")
