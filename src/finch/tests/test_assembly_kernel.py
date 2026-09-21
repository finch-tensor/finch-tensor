from types import SimpleNamespace

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import CallableFType, ffuncs, float64, ftype, int64
from finch.codegen.c_codegen import CCompiler
from finch.codegen.mlir_codegen import MLIRGenerator
from finch.codegen.numba_codegen import NumbaCompiler
from finch.compile import NotationCompiler
from finch.symbolic import Reflector


@pytest.fixture(
    params=[
        (asm, asm.AssemblyInterpreter),
        (ntn, ntn.NotationInterpreter),
        (asm, NumbaCompiler),
        (ntn, lambda: NotationCompiler(NumbaCompiler())),
        pytest.param((asm, CCompiler), marks=pytest.mark.c_backend),
        pytest.param(
            (ntn, lambda: NotationCompiler(CCompiler())), marks=pytest.mark.c_backend
        ),
    ],
)
def backend(request):
    ir, compiler = request.param
    return ir, compiler()


def kernel_program(ir, callee_kind):
    x = ir.Variable("x", int64)
    helper = ir.Function(
        ir.Variable(
            "increment", asm.AssemblyKernelFType("increment", (x.result_type,), int64)
        ),
        (x,),
        ir.Block(
            (ir.Return(ir.Call(ir.Literal(ffuncs.add), (x, ir.Literal(np.int64(1))))),),
        ),
    )
    function_type = helper.name.result_type
    factory = ir.Function(
        ir.Variable(
            "get_increment", asm.AssemblyKernelFType("get_increment", (), function_type)
        ),
        (),
        ir.Block((ir.Return(helper.name),)),
    )
    statements = []
    args = (x,)
    match callee_kind:
        case "direct":
            callee = helper.name
        case "alias":
            callee = ir.Variable("f", function_type)
            statements.append(ir.Assign(callee, helper.name))
        case "call":
            callee = ir.Call(factory.name, ())
        case "argument":
            callee = ir.Variable("f", function_type)
            args = (callee, x)
        case _:
            raise ValueError(callee_kind)
    statements.append(ir.Return(ir.Call(callee, (x,))))
    main = ir.Function(
        ir.Variable(
            "apply_increment",
            asm.AssemblyKernelFType(
                "apply_increment", tuple(arg.result_type for arg in args), int64
            ),
        ),
        args,
        ir.Block(tuple(statements)),
    )
    # Callers precede their definitions to exercise module symbol registration.
    return ir.Module((main, factory, helper)), function_type


@pytest.mark.parametrize("callee_kind", ["direct", "alias", "call", "argument"])
def test_kernel_calls(backend, callee_kind):
    ir, compiler = backend
    program, function_type = kernel_program(ir, callee_kind)
    if ir is ntn:
        lowered = NotationCompiler(Reflector())(program)  # ty: ignore[invalid-argument-type]
        assert lowered.funcs[-1].name.result_type == function_type
    else:
        lowered = program
    asm.AssemblyTypeChecker()(lowered)
    library = compiler(program)
    assert ftype(library.increment) == function_type
    assert function_type(library.increment) is library.increment
    args = (
        (library.increment, np.int64(4))
        if callee_kind == "argument"
        else (np.int64(4),)
    )
    assert library.apply_increment(*args) == 5
    returned = library.get_increment()
    assert ftype(returned) == function_type
    assert returned(np.int64(8)) == 9


def test_kernel_module_bindings_are_independent(backend):
    ir, compiler = backend
    program, _ = kernel_program(ir, "direct")
    first = compiler(program)
    helper = program.funcs[-1]
    replacement = ir.Function(
        ir.Variable("increment", asm.AssemblyKernelFType("increment", (int64,), int64)),
        helper.args,
        ir.Block((ir.Return(ir.Literal(np.int64(99))),)),
    )
    second = compiler(ir.Module((replacement,)))
    assert first.apply_increment(np.int64(4)) == 5
    assert second.increment(np.int64(4)) == 99
    assert ftype(first.increment) != ftype(second.increment)


def test_julia_cache_preserves_definition_type(monkeypatch):
    from finch.compile_jl import compiler as jl_compiler

    evaluated = []
    monkeypatch.setattr(jl_compiler, "jl", SimpleNamespace(seval=evaluated.append))
    monkeypatch.setattr(jl_compiler.FinchJLCompiler, "_kernels", {})
    first = ntn.Function(
        ntn.Variable("constant", asm.AssemblyKernelFType("constant", (), int64)),
        (),
        ntn.Block((ntn.Return(ntn.Literal(np.int64(1))),)),
    )
    second = ntn.Function(
        ntn.Variable("constant", asm.AssemblyKernelFType("constant", (), int64)),
        first.args,
        first.body,
    )
    compiler = jl_compiler.FinchJLCompiler()
    a = compiler(ntn.Module((first,))).constant
    b = compiler(ntn.Module((second,))).constant
    assert ftype(a) == first.name.result_type
    assert ftype(b) == second.name.result_type
    assert ftype(a) != ftype(b)
    assert len(evaluated) == 1


@pytest.mark.parametrize("callee_kind", ["direct", "alias", "call", "argument"])
def test_kernel_mlir_generation(callee_kind):
    program, _ = kernel_program(asm, callee_kind)
    code = MLIRGenerator()(program).code
    assert "func.call_indirect" in code
    assert "func.constant @increment" in code


def test_kernel_identity_and_signature():
    a = asm.AssemblyKernelFType("f", (int64,), int64)
    b = asm.AssemblyKernelFType("f", (int64,), int64)
    assert isinstance(a, CallableFType)
    assert a != b
    assert len({a, b}) == 2
    assert a.return_type(int64) == int64
    for args in [(), (float64,), (int64, int64)]:
        with pytest.raises(TypeError, match="expects"):
            a.return_type(*args)


def test_kernel_call_type_errors():
    program, function_type = kernel_program(asm, "direct")
    x = asm.Variable("x", float64)
    bad = asm.Function(
        asm.Variable("bad", asm.AssemblyKernelFType("bad", (x.result_type,), int64)),
        (x,),
        asm.Block((asm.Return(asm.Call(program.funcs[-1].name, (x,))),)),
    )
    with pytest.raises(asm.AssemblyTypeError, match="Cannot call"):
        asm.AssemblyTypeChecker()(asm.Module((*program.funcs, bad)))
    other_type = asm.AssemblyKernelFType(function_type.name, (int64,), int64)
    bad = asm.Function(
        asm.Variable("bad", asm.AssemblyKernelFType("bad", (), int64)),
        (),
        asm.Block(
            (
                asm.Return(
                    asm.Call(
                        asm.Variable("increment", other_type),
                        (asm.Literal(np.int64(1)),),
                    )
                ),
            )
        ),
    )
    with pytest.raises(asm.AssemblyTypeError):
        asm.AssemblyTypeChecker()(asm.Module((*program.funcs, bad)))


def test_kernel_definition_signature_must_match_arguments():
    name = asm.Variable("f", asm.AssemblyKernelFType("f", (int64,), float64))
    x = asm.Variable("x", float64)
    definition = asm.Function(name, (x,), asm.Return(x))
    with pytest.raises(asm.AssemblyTypeError, match="arguments do not match"):
        asm.AssemblyTypeChecker()(definition)
