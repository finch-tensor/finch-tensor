"""Read one element while varying the size of the tensor crossing into Julia.

Run with ``pixi run -e benchmark-julia pytest --codspeed
benchmarks/test_julia_interop.py``. Tensor allocation and kernel compilation
are excluded. ``cached`` reuses the runtime's input translation; ``fresh``
passes a new Python tensor handle each time, including input conversion.
Both paths call the compiled Julia kernel and retrieve its scalar result.
The sparse formats store every coordinate so their buffers grow with size.
Fresh conversions borrow storage; comparing sizes checks both first-use and
steady-state runtime overhead without relying on the translation cache.
"""

from functools import lru_cache

import pytest

import numpy as np

import finch as ft
import finch.finch_notation as ntn
from finch.compile_jl.compiler import FinchJLKernel
from finch.compile_jl.interop import jl_tensor_to_python
from finch.compile_jl.julia import jl, julia_available
from finch.compile_jl.runtime import DefaultFinchJLRuntime
from finch.finch_assembly import AssemblyKernelFType, Buffer

pytestmark = pytest.mark.skipif(
    not julia_available(), reason="Julia backend (juliacall/juliapkg) not installed"
)

FORMATS = {
    "hash": "SparseHash",
    "list": "SparseList",
    "dense": "Dense",
    "bytemap": "SparseByteMap",
}
SIZES = (1_024, 32_768, 1_048_576)


@lru_cache
def _read_one_julia():
    # Direct fiber lookup isolates interop from the DSL's traversal protocols.
    source = """
        function finch_interop_read_one(input, output)
            output.lvl.val[1] = input(size(input)[1] ÷ 2 + 1)
            return (output,)
        end
    """
    jl.seval(source)
    return source


def _make_tensor(kind, size):
    make_tensor = jl.seval(
        f"n -> Tensor({FORMATS[kind]}(Element(0.0)), collect(1.0:n))"
    )
    source = jl_tensor_to_python(make_tensor(size))
    level = source.lvl
    # Use Python-owned, zero-based buffers, including the hash keys. This
    # exercises conversion rather than returning an existing Julia handle.
    fields = []
    for name, _ in level.ftype.struct_fields:
        value = getattr(level, name)
        if name == "lvl":
            value = ft.ElementLevel(
                value.ftype, ft.NumpyBuffer(value.val.to_numpy(copy=True))
            )
        elif isinstance(value, Buffer):
            data = value.to_numpy(copy=True)
            if kind == "hash" and name == "tbl":
                # Julia leaves unoccupied slots uninitialized.
                data[level.tbl_ctrl.to_numpy() == 0] = 0
            value = ft.NumpyBuffer(data)
        fields.append(value)
    return ft.FiberTensor(level.ftype.from_fields(*fields))


def _make_read_one(source):
    output = ft.asarray(np.array(0.0))
    arg = ntn.Variable("input", source.ftype)
    out = ntn.Variable("output", output.ftype)
    arg_slot = ntn.Slot("input_slot", arg.result_type)
    out_slot = ntn.Slot("output_slot", out.result_type)
    op = ntn.Literal(ft.ffuncs.init_write(0.0))
    result = ntn.Call(ntn.Literal(ft.ffuncs.make_tuple), (out,))
    function = ntn.Function(
        ntn.Variable(
            "read_one",
            AssemblyKernelFType(
                "read_one", (arg.result_type, out.result_type), result.result_type
            ),
        ),
        (arg, out),
        ntn.Block(
            (
                ntn.Unpack(arg_slot, arg),
                ntn.Unpack(out_slot, out),
                ntn.Declare(out_slot, ntn.Literal(0.0), op, ()),
                ntn.Increment(
                    ntn.Access(out_slot, ntn.Update(op), ()),
                    ntn.Access(
                        arg_slot, ntn.Read(), (ntn.Literal(source.shape[0] // 2 + 1),)
                    ),
                ),
                ntn.Freeze(out_slot, op),
                ntn.Repack(out_slot, out),
                ntn.Return(result),
            )
        ),
    )
    runtime = DefaultFinchJLRuntime()
    kernel = FinchJLKernel(
        _read_one_julia(),
        function.name.result_type,
        function,
        runtime,
        "finch_interop_read_one",
    )
    runtime.cache_kernel("read_one", kernel)
    return kernel, output


def _read_call(source, kernel, output, conversion):
    def read_one():
        arg = ft.FiberTensor(source.lvl) if conversion == "fresh" else source
        return kernel(arg, output)[0].item()

    return read_one


@pytest.mark.parametrize("kind", FORMATS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("conversion", ["cached", "fresh"])
def test_julia_single_element_read(kind, size, conversion, benchmark):
    source = _make_tensor(kind, size)
    kernel, output = _make_read_one(source)
    read_one = _read_call(source, kernel, output, conversion)
    try:
        assert read_one() == size // 2 + 1
        benchmark(read_one)
    finally:
        kernel.runtime.close()
