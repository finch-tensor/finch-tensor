from dataclasses import dataclass

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import (
    CallableFType,
    DynamicFill,
    DynamicFillError,
    FTyped,
    ImmutableStructFType,
    StaticFill,
    TupleFType,
    ffuncs,
    ftype,
)
from finch.codegen import (
    CCompiler,
    CGenerator,
    MLIRCompiler,
    MLIRGenerator,
    NumbaCompiler,
    NumbaGenerator,
)
from finch.codegen.c_codegen.c import COperator
from finch.codegen.mlir_codegen.mlir import MLIROperator, mlir_binary_function_call
from finch.codegen.numba_codegen.numba import NumbaOperator
from finch.compile import NotationCompiler
from finch.finch_assembly.type_checker import AssemblyTypeChecker


@pytest.fixture
def custom_callable_program():
    dtype = ftype(np.int64)

    class ShiftFType(
        ImmutableStructFType, CallableFType, COperator, NumbaOperator, MLIROperator
    ):
        struct_name = "Shift"
        struct_fields = [("offset", dtype)]
        c_symbol = "+"

        def from_fields(self, offset):
            return Shift(offset)

        def return_type(self, *args):
            return args[0]

        def c_function_call(self, op, ctx, *args):
            offset = ctx(asm.GetAttr(op, asm.Literal("offset")))
            return f"({ctx(args[0])} + {offset})"

        def numba_name(self):
            return "+"

        def numba_function_call(self, op, ctx, *args):
            offset = ctx(asm.GetAttr(op, asm.Literal("offset")))
            return f"({ctx(args[0])} + {offset})"

        def mlir_name(self):
            return "arith.addi"

        def mlir_function_call(self, op, ctx, *args):
            return mlir_binary_function_call(
                self.mlir_name(), ctx, args[0], asm.GetAttr(op, asm.Literal("offset"))
            )

    @dataclass
    class Shift(FTyped):
        offset: np.int64

        @property
        def ftype(self):
            return ShiftFType()

        def __call__(self, value):
            return value + self.offset

    op, x = asm.Variable("op", ShiftFType()), asm.Variable("x", dtype)
    callee = asm.Call(asm.Literal(ffuncs.identity), (op,))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "apply",
                    asm.AssemblyKernelFType(
                        "apply",
                        (op.result_type, x.result_type),
                        dtype,
                    ),
                ),
                (op, x),
                asm.Block((asm.Return(asm.Call(callee, (x,))),)),
            ),
        )
    )
    AssemblyTypeChecker()(program)
    return program, Shift


@pytest.mark.parametrize(
    "compiler",
    [
        asm.AssemblyInterpreter(),
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
def test_custom_callable_lowering(compiler, custom_callable_program):
    program, shift = custom_callable_program
    module = compiler(program)
    for offset in (1, 7):
        assert module.apply(shift(np.int64(offset)), np.int64(3)) == 3 + offset


@pytest.mark.parametrize(
    "compiler",
    [
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.float64(np.nan), np.float64(np.nan), True),
        (np.float64(np.nan), np.float64(2), False),
        (np.float64(2), np.float64(np.nan), False),
        (np.float64(np.inf), np.float64(np.inf), True),
        (np.int64(1), np.int64(2), False),
        ((), (), True),
        ((np.int64(1),), (), False),
        (
            (np.float64(np.nan), (np.int64(1),)),
            (np.float64(np.nan), (np.int64(1),)),
            True,
        ),
        (
            (np.float64(np.nan), (np.int64(1),)),
            (np.float64(np.nan), (np.int64(2),)),
            False,
        ),
    ],
)
def test_same_backend_lowering(compiler, x, y, expected):
    a, b = asm.Variable("a", ftype(x)), asm.Variable("b", ftype(y))
    call = asm.Call(asm.Literal(ffuncs.same), (a, b))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "compare",
                    asm.AssemblyKernelFType(
                        "compare",
                        (a.result_type, b.result_type),
                        call.result_type,
                    ),
                ),
                (a, b),
                asm.Block((asm.Return(call),)),
            ),
        )
    )
    assert compiler(program).compare(x, y) == expected


@pytest.mark.parametrize(
    "compiler",
    [
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
@pytest.mark.parametrize("tuple_fill", [False, True])
@pytest.mark.parametrize("nargs", [0, 1, 3])
def test_choose_backend_lowering(compiler, tuple_fill, nargs):
    fill, value = np.float64(np.nan), np.float64(2)
    if tuple_fill:
        fill, value = (fill, np.float64(0)), (value, np.float64(0))
    operator = ffuncs.choose(DynamicFill(fill))
    op = asm.Variable("op", ftype(operator))
    values = (fill, value, fill)[:nargs]
    args = tuple(asm.Variable(f"arg{i}", ftype(arg)) for i, arg in enumerate(values))
    call = asm.Call(op, args)
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "choose",
                    asm.AssemblyKernelFType(
                        "choose",
                        (op.result_type, *(arg.result_type for arg in args)),
                        call.result_type,
                    ),
                ),
                (op, *args),
                asm.Block((asm.Return(call),)),
            ),
        )
    )
    np.testing.assert_equal(
        compiler(program).choose(operator, *values), operator(*values)
    )


@pytest.mark.parametrize(
    "compiler",
    [
        asm.AssemblyInterpreter(),
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("fill_type", [StaticFill, DynamicFill])
def test_callable_expression(compiler, field, fill_type):
    operator = ffuncs.init_write(fill_type(np.int64(0)))
    op_type = ftype(operator)
    arg_type = TupleFType.from_tuple((op_type,)) if field else op_type
    op = asm.Variable("op", arg_type)
    x, y = asm.Variable("x", ftype(np.int64)), asm.Variable("y", ftype(np.int64))
    callee = asm.GetAttr(op, asm.Literal("element_0")) if field else op
    call = asm.Call(callee, (x, y))
    assert call.result_type == ftype(np.int64)
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "write",
                    asm.AssemblyKernelFType(
                        "write",
                        (op.result_type, x.result_type, y.result_type),
                        ftype(np.int64),
                    ),
                ),
                (op, x, y),
                asm.Block((asm.Return(call),)),
            ),
        )
    )
    AssemblyTypeChecker()(program)
    module = compiler(program)
    fills = (0, 3) if fill_type is DynamicFill else (0,)
    for fill in fills:
        value = ffuncs.init_write(fill_type(np.int64(fill)))
        assert ftype(value) == op_type
        arg = (value,) if field else value
        assert module.write(arg, np.int64(9), np.int64(fill)) == 9
        assert module.write(arg, np.int64(9), np.int64(7)) == 7


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter(), NotationCompiler()])
def test_notation_callable_expression(compiler):
    operator = ffuncs.init_write(DynamicFill(np.int64(0)))
    op = ntn.Variable("op", ftype(operator))
    x, y = ntn.Variable("x", ftype(np.int64)), ntn.Variable("y", ftype(np.int64))
    callee = ntn.Call(ntn.Literal(ffuncs.identity), (op,))
    call = ntn.Call(callee, (x, y))
    assert call.result_type == ftype(np.int64)
    program = ntn.Module(
        (
            ntn.Function(
                ntn.Variable(
                    "write",
                    asm.AssemblyKernelFType(
                        "write",
                        (op.result_type, x.result_type, y.result_type),
                        ftype(np.int64),
                    ),
                ),
                (op, x, y),
                ntn.Block((ntn.Return(call),)),
            ),
        )
    )
    module = compiler(program)
    for fill in (0, 3):
        operator = ffuncs.init_write(DynamicFill(np.int64(fill)))
        assert module.write(operator, np.int64(9), np.int64(fill)) == 9
        assert module.write(operator, np.int64(9), np.int64(7)) == 7


@pytest.mark.parametrize("generator", [NumbaGenerator(), CGenerator(), MLIRGenerator()])
def test_dynamic_operator_literal_cannot_specialize(generator):
    op = ffuncs.init_write(DynamicFill(np.int64(3)))
    expression = asm.Call(
        asm.Literal(op), (asm.Literal(np.int64(9)), asm.Literal(np.int64(3)))
    )
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "write", asm.AssemblyKernelFType("write", (), ftype(np.int64))
                ),
                (),
                asm.Block((asm.Return(expression),)),
            ),
        )
    )
    with pytest.raises(DynamicFillError):
        generator(program)


@pytest.mark.parametrize(
    "compiler",
    [
        asm.AssemblyInterpreter(),
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
    ],
)
def test_builtin_operator_argument(compiler):
    op = asm.Variable("op", ftype(ffuncs.add))
    x, y = asm.Variable("x", ftype(np.int64)), asm.Variable("y", ftype(np.int64))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "apply",
                    asm.AssemblyKernelFType(
                        "apply",
                        (op.result_type, x.result_type, y.result_type),
                        ftype(np.int64),
                    ),
                ),
                (op, x, y),
                asm.Block((asm.Return(asm.Call(op, (x, y))),)),
            ),
        )
    )
    AssemblyTypeChecker()(program)
    assert compiler(program).apply(ffuncs.add, np.int64(2), np.int64(3)) == 5


@pytest.mark.parametrize(
    "compiler",
    [
        asm.AssemblyInterpreter(),
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
@pytest.mark.parametrize("factory", [ffuncs.init_write, ffuncs.choose])
def test_call_selects_runtime_operator(compiler, factory):
    dtype = ftype(np.int64)
    op_type = ftype(factory(DynamicFill(np.int64(0))))
    first, second = asm.Variable("first", op_type), asm.Variable("second", op_type)
    which = asm.Variable("which", ftype(np.bool_))
    x, y = asm.Variable("x", dtype), asm.Variable("y", dtype)
    callee = asm.Call(asm.Literal(ffuncs.where), (which, first, second))
    result = asm.Call(callee, (x, y))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "apply",
                    asm.AssemblyKernelFType(
                        "apply",
                        (
                            first.result_type,
                            second.result_type,
                            which.result_type,
                            x.result_type,
                            y.result_type,
                        ),
                        dtype,
                    ),
                ),
                (first, second, which, x, y),
                asm.Block((asm.Return(result),)),
            ),
        )
    )
    AssemblyTypeChecker()(program)
    module = compiler(program)
    for first_fill, second_fill in [(0, 3), (7, 0)]:
        first_value = factory(DynamicFill(np.int64(first_fill)))
        second_value = factory(DynamicFill(np.int64(second_fill)))
        for choice in (True, False):
            selected = first_value if choice else second_value
            for x_value, y_value in [(0, 3), (7, 0), (3, 9)]:
                x_value, y_value = np.int64(x_value), np.int64(y_value)
                assert module.apply(
                    first_value, second_value, np.bool_(choice), x_value, y_value
                ) == selected(x_value, y_value)


@pytest.mark.parametrize("ir", [asm, ntn])
def test_simplification_uses_callable_expression_type(ir):
    from finch.symbolic import Chain, Fixpoint, PostWalk, Rewrite, simplify_rules

    simplify = Rewrite(Fixpoint(PostWalk(Chain(simplify_rules()))))
    dtype = ftype(np.int64)
    add = ir.Variable("add", ftype(ffuncs.add))
    mul = ir.Variable("mul", ftype(ffuncs.mul))
    x = ir.Variable("x", dtype)
    zero, two, three = (ir.Literal(np.int64(i)) for i in (0, 2, 3))
    assert simplify(ir.Call(add, (x, zero))) == x
    assert simplify(ir.Call(mul, (x, zero))) == zero
    # The type proves algebraic rules, but does not supply the runtime callee.
    assert simplify(ir.Call(add, (two, three))) == ir.Call(add, (two, three))
    nested = ir.Call(ir.Literal(ffuncs.identity), (add,))
    assert simplify(ir.Call(nested, (x, zero))) == x
    dynamic = ir.Variable("dynamic", ftype(ffuncs.init_write(DynamicFill(np.int64(0)))))
    assert simplify(ir.Call(dynamic, (x, zero))) == ir.Call(dynamic, (x, zero))


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter(), NotationCompiler()])
@pytest.mark.parametrize("fill", [0, 3])
def test_logic_lowering_preserves_literal_callee(compiler, fill):
    from finch.autoschedule import NotationGenerator
    from finch.tensor import BufferizedNDArray

    op = lgc.Literal(ffuncs.init_write(StaticFill(np.int64(fill))))
    source, output, index = lgc.Alias("source"), lgc.Alias("output"), lgc.Field("i")
    expression = lgc.MapJoin(
        op, (lgc.Literal(np.int64(9)), lgc.Table(source, (index,)))
    )
    plan = lgc.Plan(
        (
            lgc.Query(
                output,
                lgc.Reorder(
                    lgc.Aggregate(
                        lgc.Literal(ffuncs.overwrite),
                        lgc.Literal(np.int64(0)),
                        lgc.Reorder(expression, (index,)),
                        (),
                    ),
                    (index,),
                ),
            ),
            lgc.Produces((output,)),
        )
    )
    values = np.array([0, 3, 7], dtype=np.int64)
    src = BufferizedNDArray.from_numpy(values)
    out = BufferizedNDArray.from_numpy(np.zeros_like(values))
    bindings = {source: ftype(src), output: ftype(out)}
    program = NotationGenerator()(plan, bindings, {}, None)
    module = compiler(program)
    module.main(src, out)
    np.testing.assert_array_equal(out.to_numpy(), np.where(values == fill, 9, values))


@pytest.mark.parametrize("ir", [asm, ntn])
def test_callable_without_algebraic_properties(ir):
    from finch.algebra import CallableFType, FinchOperatorFType, FTyped, return_type
    from finch.symbolic import Chain, Fixpoint, PostWalk, Rewrite, simplify_rules

    class OpaqueCallableFType(CallableFType):
        def __eq__(self, other):
            return type(self) is type(other)

        def __hash__(self):
            return hash(type(self))

        def __call__(self, value):
            return value

        def return_type(self, *args):
            return args[0]

    class OpaqueCallable(FTyped):
        @property
        def ftype(self):
            return OpaqueCallableFType()

        def __call__(self, value):
            return value + 1

    op_type = OpaqueCallableFType()
    dtype = ftype(np.int64)
    assert not isinstance(op_type, FinchOperatorFType)
    assert return_type(op_type, dtype) == dtype
    op, x = ir.Variable("op", op_type), ir.Variable("x", dtype)
    zero = ir.Literal(np.int64(0))
    simplify = Rewrite(Fixpoint(PostWalk(Chain(simplify_rules()))))
    for args in [(x,), (x, x), (x, zero), (zero, zero)]:
        call = ir.Call(op, args)
        assert call.result_type == dtype
        assert simplify(call) == call

    if ir is asm:
        program = asm.Module(
            (
                asm.Function(
                    asm.Variable(
                        "apply",
                        asm.AssemblyKernelFType(
                            "apply",
                            (op.result_type, x.result_type),
                            dtype,
                        ),
                    ),
                    (op, x),
                    asm.Block((asm.Return(asm.Call(op, (x,))),)),
                ),
            )
        )
        AssemblyTypeChecker()(program)
        module = asm.AssemblyInterpreter()(program)
        assert module.apply(OpaqueCallable(), np.int64(3)) == 4
