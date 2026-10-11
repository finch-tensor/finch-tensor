from collections import namedtuple

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch.algebra import FinchOperator, SingletonOperatorFType, ffuncs, ftype
from finch.codegen import CCompiler, NumpyBuffer
from finch.codegen.c_codegen import c as c_backend

pytestmark = pytest.mark.c_backend

EMPTY_VALUES = [
    ffuncs.add,
    (),
    (ffuncs.add, None, (ffuncs.mul,)),
    namedtuple("Empty", ())(),
]


@pytest.mark.parametrize("value", EMPTY_VALUES)
def test_empty_arguments(value):
    empty_type, dtype = ftype(value), ftype(np.int64)
    buffer = NumpyBuffer(np.zeros(1, dtype=np.int64))
    first, last = asm.Variable("first", empty_type), asm.Variable("last", empty_type)
    x, y = asm.Variable("x", dtype), asm.Variable("y", dtype)
    buf, slot = asm.Variable("buf", buffer.ftype), asm.Slot("slot", buffer.ftype)
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "apply",
                    asm.AssemblyKernelFType(
                        "apply",
                        (
                            first.result_type,
                            x.result_type,
                            buf.result_type,
                            y.result_type,
                            last.result_type,
                        ),
                        dtype,
                    ),
                ),
                (first, x, buf, y, last),
                asm.Block(
                    (
                        asm.Unpack(slot, buf),
                        asm.Resize(slot, y),
                        asm.Return(asm.Call(asm.Literal(ffuncs.add), (x, y))),
                    )
                ),
            ),
        )
    )
    kernel = CCompiler()(program).apply
    assert kernel(value, np.int64(2), buffer, np.int64(3), value) == 5
    assert buffer.arr.size == 3
    with pytest.raises(TypeError):
        kernel(np.int64(0), np.int64(2), buffer, np.int64(3), value)
    with pytest.raises(ValueError):
        kernel(np.int64(2), buffer, np.int64(3))


@pytest.mark.parametrize("value", EMPTY_VALUES)
def test_empty_return(value):
    fmt = ftype(value)
    arg, local = asm.Variable("arg", fmt), asm.Variable("local", fmt)
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "identity",
                    asm.AssemblyKernelFType("identity", (arg.result_type,), fmt),
                ),
                (arg,),
                asm.Block(
                    (
                        asm.Assign(
                            local, asm.Call(asm.Literal(ffuncs.identity), (arg,))
                        ),
                        asm.Return(local),
                    )
                ),
            ),
        )
    )
    kernel = CCompiler()(program).identity
    assert kernel(value) == value


@pytest.mark.parametrize("named", [False, True])
def test_structs_with_empty_fields(named):
    value = (ffuncs.add, np.int64(7), ((), ffuncs.mul), None)
    names = ("op", "value", "nested", "nothing")
    if named:
        value = namedtuple("Payload", names)(*value)
    else:
        names = tuple(f"element_{i}" for i in range(len(value)))
    fmt, dtype = ftype(value), ftype(np.int64)
    arg, x = asm.Variable("arg", fmt), asm.Variable("x", dtype)
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "identity",
                    asm.AssemblyKernelFType("identity", (arg.result_type,), fmt),
                ),
                (arg,),
                asm.Block((asm.Return(arg),)),
            ),
            asm.Function(
                asm.Variable(
                    "apply",
                    asm.AssemblyKernelFType(
                        "apply",
                        (arg.result_type, x.result_type),
                        dtype,
                    ),
                ),
                (arg, x),
                asm.Block(
                    (
                        asm.Return(
                            asm.Call(
                                asm.GetAttr(arg, asm.Literal(names[0])),
                                (asm.GetAttr(arg, asm.Literal(names[1])), x),
                            )
                        ),
                    )
                ),
            ),
        )
    )
    module = CCompiler()(program)
    assert module.identity(value) == value
    assert module.apply(value, np.int64(2)) == 9


@pytest.mark.parametrize("use", ["callee", "assign", "return", "conditional"])
def test_empty_results_preserve_effects(use):
    class MakeAdderFType(SingletonOperatorFType, c_backend.COperator):
        c_symbol = "make_adder"

        @property
        def operator(self):
            return make_adder

        def return_type(self, *args):
            return ffuncs.add.ftype

        def c_function_call(self, op, ctx, buffer):
            ctx.add_header("#include <stdint.h>")
            return f"((void)(++((int64_t*)({ctx(buffer)})->data)[0]))"

    class MakeAdder(FinchOperator):
        @property
        def ftype(self):
            return MakeAdderFType()

        def __call__(self, buffer):
            buffer.arr[0] += 1
            return ffuncs.add

    make_adder = MakeAdder()
    counter = NumpyBuffer(np.zeros(1, dtype=np.int64))
    buf = asm.Variable("buf", counter.ftype)
    cond = asm.Variable("cond", ftype(np.bool_))
    callee = asm.Call(asm.Literal(make_adder), (buf,))
    body = []
    if use == "assign":
        local = asm.Variable("local", ffuncs.add.ftype)
        body.append(asm.Assign(local, callee))
        callee = local
    elif use == "conditional":
        callee = asm.Call(
            asm.Literal(ffuncs.where), (cond, callee, asm.Literal(ffuncs.add))
        )
    result = (
        callee
        if use == "return"
        else asm.Call(callee, (asm.Literal(np.int64(2)), asm.Literal(np.int64(3))))
    )
    body.append(asm.Return(result))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "apply",
                    asm.AssemblyKernelFType(
                        "apply",
                        (buf.result_type, cond.result_type),
                        result.result_type,
                    ),
                ),
                (buf, cond),
                asm.Block(tuple(body)),
            ),
        )
    )
    kernel = CCompiler()(program).apply
    assert kernel(counter, np.bool_(True)) == (ffuncs.add if use == "return" else 5)
    assert counter.arr[0] == 1
    if use == "conditional":
        assert kernel(counter, np.bool_(False)) == 5
        assert counter.arr[0] == 1
