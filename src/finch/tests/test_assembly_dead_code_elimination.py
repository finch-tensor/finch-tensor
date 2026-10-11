import re

import numpy as np

import finch
import finch.finch_assembly as asm
from finch import ffuncs
from finch.algebra import ftype
from finch.autoschedule import (
    CompilerFormLowerer,
    DefaultLogicFactorizer,
    DefaultLogicFormatter,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)
from finch.codegen import NumbaCompiler, NumpyBuffer
from finch.compile import NotationCompiler
from finch.finch_assembly import (
    AssemblyDeadCodeEliminationTransform,
    AssemblyInterpreter,
    AssemblySimplify,
    LowerPackedStructSlots,
    assembly_check_types,
    assembly_dead_code_elimination,
)
from finch.finch_logic import LogicSimplify

from .test_simplification import _CaptureAssembly
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
    return asm.Module(
        (asm.Function(asm.Variable("f", func_t), tuple(args), asm.Block(body)),)
    )


def check_dce(prgm, expected, *inputs):
    result = assembly_dead_code_elimination(prgm)
    assert result == expected
    assembly_check_types(result)
    for args in inputs:
        assert AssemblyInterpreter()(result).f(*args) == (
            AssemblyInterpreter()(prgm).f(*args)
        )


def test_dce_removes_dead_chains():
    a, b, t, u, v = variables("a b t u v")
    check_dce(
        module(
            (a, b),
            asm.Assign(t, call(ffuncs.add, a, b)),
            asm.Assign(u, call(ffuncs.mul, t, i64(2))),
            asm.Assign(v, a),
            asm.Return(a),
        ),
        module((a, b), asm.Return(a)),
        (np.int64(1), np.int64(2)),
    )


def test_dce_keeps_needed_declarations():
    """`x = 0` is dead but declares `x` for after the if, `y = 0` can go"""
    a, c, x, y = variables("a c x y")

    def make(*head):
        return module(
            (a, c),
            asm.Assign(x, i64(0)),
            asm.IfElse(
                call(ffuncs.lt, c, i64(0)),
                asm.Block((asm.Assign(x, a),)),
                asm.Block((asm.Assign(x, i64(1)),)),
            ),
            *head,
            asm.Assign(y, call(ffuncs.add, x, i64(1))),
            asm.Return(y),
        )

    check_dce(
        make(asm.Assign(y, i64(0))),
        make(),
        (np.int64(5), np.int64(-1)),
        (np.int64(5), np.int64(1)),
    )


def test_dce_keeps_effects_and_loop_carried_values():
    n, s, k, p = variables("n s k p")
    buf_t = NumpyBuffer(np.zeros(1, dtype=np.int64)).ftype
    buf = asm.Variable("buf", buf_t)
    buf_slt = asm.Slot("buf_", buf_t)
    i = asm.Variable("i", finch.int64)

    def make(*dead):
        return module(
            (n, buf),
            asm.Assign(s, i64(0)),
            asm.Assign(k, call(bump, n, n)),
            asm.ForLoop(
                i,
                i64(0),
                n,
                asm.Block((asm.Assign(s, call(ffuncs.add, s, i)), *dead)),
            ),
            asm.Unpack(buf_slt, buf),
            asm.Store(buf_slt, asm.Literal(np.intp(0)), s),
            asm.Repack(buf_slt),
            asm.Return(s),
        )

    prgm = make(asm.Assign(p, call(ffuncs.mul, i, i64(2))))
    assert assembly_dead_code_elimination(prgm) == make()


def test_dce_removes_pure_bare_expressions():
    a, b = variables("a b")
    effect = call(bump, a, b)
    prgm = module((a, b), call(ffuncs.add, a, b), effect, asm.Return(a))
    assert assembly_dead_code_elimination(prgm) == module((a, b), effect, asm.Return(a))


def test_dce_pipeline_sparse():
    """the sparse elementwise kernel computes positions it never uses"""
    from finch import DenseLevel, ElementLevel, FiberTensor, SparseListLevel, element

    def make():
        ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
        idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
        data = NumpyBuffer(np.array([1.0, 1.0, 2.0, 1.0]))
        elem = element(
            np.float64(0), ftype(np.float64), ftype(np.intp), type(data.ftype)
        )
        return FiberTensor(
            DenseLevel(
                SparseListLevel(ElementLevel(elem, data), np.intp(3), ptr, idx),
                np.intp(3),
            )
        )

    def compile_with(*transforms):
        capture = _CaptureAssembly(NumbaCompiler())
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
                                                *transforms,
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
        out = finch.compute(finch.defer(make()) * finch.defer(make()), ctx=ctx)
        np.testing.assert_array_equal(
            out.to_numpy(),
            np.array([[1, 0, 0], [1, 4, 0], [0, 0, 1]], dtype=np.float64),
        )
        assembly_check_types(capture.last)
        return str(capture.last)

    before = compile_with()
    after = compile_with(AssemblyDeadCodeEliminationTransform(), AssemblySimplify())
    loops = re.compile(r"^\s*for ", re.MULTILINE)
    assert len(loops.findall(after)) < len(loops.findall(before))
