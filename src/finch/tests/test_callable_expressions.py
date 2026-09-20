import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import (
    DynamicFill,
    DynamicFillError,
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
from finch.compile import NotationCompiler
from finch.finch_assembly.type_checker import AssemblyTypeChecker


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
                asm.Variable("write", ftype(np.int64)),
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
                ntn.Variable("write", ftype(np.int64)),
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
                asm.Variable("write", ftype(np.int64)),
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
                asm.Variable("apply", ftype(np.int64)),
                (op, x, y),
                asm.Block((asm.Return(asm.Call(op, (x, y))),)),
            ),
        )
    )
    AssemblyTypeChecker()(program)
    assert compiler(program).apply(ffuncs.add, np.int64(2), np.int64(3)) == 5
