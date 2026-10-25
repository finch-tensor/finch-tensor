import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyCopyPropagation,
    assembly_copy_propagation,
    assembly_dataflow_analyze,
)
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
