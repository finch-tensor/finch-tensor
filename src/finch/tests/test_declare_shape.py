import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import DynamicFill, ffuncs, ftype
from finch.codegen import CCompiler, MLIRCompiler, NumbaCompiler
from finch.compile import CompilerMode, NotationCompiler, make_extent
from finch.compile.lower import AssemblyGenerator
from finch.tensor import BufferizedNDArray, dense, element, fiber_tensor, sparse_list

from .test_sparse_update import dense_values


@pytest.fixture(
    params=[
        asm.AssemblyInterpreter,
        NumbaCompiler,
        pytest.param(CCompiler, marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler, marks=pytest.mark.mlir_backend),
    ]
)
def loader(request):
    return request.param()


def declare_program(fmt, *, shape_from_tensor=False, init=None):
    tensor = ntn.Variable("tensor", fmt)
    slot = ntn.Slot("tensor_slot", fmt)
    shape = tuple(ntn.Variable(f"dim_{i}", ftype(np.intp)) for i in range(fmt.ndim))
    declared_shape = (
        tuple(
            ntn.Call(
                ntn.Literal(ffuncs.add),
                (ntn.Dimension(slot, ntn.Literal(i)), ntn.Literal(np.intp(1))),
            )
            for i in reversed(range(fmt.ndim))
        )
        if shape_from_tensor
        else shape
    )
    idxs = tuple(ntn.Variable(f"i_{i}", ftype(np.intp)) for i in range(fmt.ndim))
    op = ntn.Literal(ffuncs.overwrite)
    value = (
        ntn.Call(ntn.Literal(ffuncs.add), (ntn.Literal(np.int64(1)), *idxs))
        if idxs
        else ntn.Literal(np.int64(1))
    )
    body = ntn.Increment(ntn.Access(slot, ntn.Update(op), idxs), value)
    if idxs:
        body = ntn.If(
            ntn.Call(ntn.Literal(ffuncs.eq), (idxs[-1], ntn.Literal(np.intp(0)))), body
        )
    for axis, idx in reversed(tuple(enumerate(idxs))):
        body = ntn.Loop(
            idx,
            ntn.Call(
                ntn.Literal(make_extent),
                (
                    ntn.Literal(np.intp(0)),
                    ntn.Call(
                        ntn.Literal(ffuncs.astype(ftype(np.intp))),
                        (ntn.Dimension(slot, ntn.Literal(axis)),),
                    ),
                ),
            ),
            body,
        )
    args = (tensor, *shape)
    return ntn.Module(
        (
            ntn.Function(
                ntn.Variable(
                    "resize",
                    asm.AssemblyKernelFType(
                        "resize", tuple(a.result_type for a in args), fmt
                    ),
                ),
                args,
                ntn.Block(
                    (
                        ntn.Unpack(slot, tensor),
                        ntn.Declare(
                            slot,
                            ntn.Literal(np.int64(0)) if init is None else init,
                            op,
                            declared_shape,
                        ),
                        body,
                        ntn.Freeze(slot, op),
                        ntn.Repack(slot, tensor),
                        ntn.Return(tensor),
                    )
                ),
            ),
        )
    )


def resized_values(shape):
    result = np.zeros(shape, dtype=np.int64)
    for coord in np.ndindex(shape):
        if coord and coord[-1] != 0:
            continue
        result[coord] = 1 + sum(coord)
    return result


@pytest.mark.parametrize("layout", ["array", "ddd", "dsd", "sdd", "sss"])
def test_declare_resizes_and_reuses_kernel(loader, layout):
    if layout == "array":
        tensor = BufferizedNDArray.from_numpy(np.zeros((0, 0, 0), dtype=np.int64))
    else:
        level = element(np.int64(0), position_type=ftype(np.int32))
        for i, kind in reversed(tuple(enumerate(layout))):
            level = (dense if kind == "d" else sparse_list)(
                level, dimension_type=ftype(np.int32 if i % 2 == 0 else np.int64)
            )
        tensor = fiber_tensor(level).construct((0, 0, 0))
    fmt = tensor.ftype
    program = declare_program(fmt)
    asm.AssemblyTypeChecker()(AssemblyGenerator()(program))
    kernel = NotationCompiler(loader, mode=CompilerMode(debug=True))(program).resize
    for shape in ((2, 3, 4), (1, 2, 1), (2, 0, 3), (0, 2, 3), (3, 1, 2)):
        result = kernel(tensor, *(np.intp(dim) for dim in shape))
        for value in (tensor, result):
            assert value.shape == shape
            assert value.ftype == fmt
            actual = value.to_numpy() if layout == "array" else dense_values(value.lvl)
            np.testing.assert_array_equal(actual, resized_values(shape))


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_ndarray_declare_in_notation_interpreter(shape):
    tensor = BufferizedNDArray.from_numpy(
        np.full((1,) * len(shape), 99, dtype=np.int64)
    )
    kernel = ntn.NotationInterpreter()(declare_program(tensor.ftype)).resize
    for dims in (shape, (2,) * len(shape)):
        result = kernel(tensor, *(np.intp(dim) for dim in dims))
        assert tensor.shape == result.shape == dims
        np.testing.assert_array_equal(result.to_numpy(), resized_values(dims))


@pytest.mark.parametrize("array", [False, True])
def test_declare_evaluates_shape_before_resizing(loader, array):
    tensor = (
        BufferizedNDArray.from_numpy(np.zeros((2, 3), dtype=np.int64))
        if array
        else fiber_tensor(dense(dense(element(np.int64(0))))).construct((2, 3))
    )
    program = declare_program(tensor.ftype, shape_from_tensor=True)
    asm.AssemblyTypeChecker()(AssemblyGenerator()(program))
    kernel = NotationCompiler(loader, mode=CompilerMode(debug=True))(program).resize
    for shape in ((4, 3), (4, 5), (6, 5)):
        result = kernel(tensor, np.intp(0), np.intp(0))
        assert tensor.shape == result.shape == shape
        np.testing.assert_array_equal(result.to_numpy(), resized_values(shape))


@pytest.mark.parametrize("array", [False, True])
def test_declare_evaluates_init_before_resizing(loader, array):
    fill = DynamicFill(np.int64(0))
    tensor = (
        BufferizedNDArray.from_numpy(np.zeros((2, 3), dtype=np.int64), fill_value=fill)
        if array
        else fiber_tensor(dense(dense(element(fill.ftype)))).construct(
            (2, 3), fill_value=fill
        )
    )
    init = ntn.Dimension(ntn.Slot("tensor_slot", tensor.ftype), ntn.Literal(0))
    program = declare_program(tensor.ftype, init=init)
    kernel = NotationCompiler(loader, mode=CompilerMode(debug=True))(program).resize
    for shape in ((1, 3), (4, 2)):
        old_dim = tensor.shape[0]
        result = kernel(tensor, *(np.intp(dim) for dim in shape))
        assert result.fill_value == old_dim
        expected = np.full(shape, old_dim, dtype=np.int64)
        expected[:, 0] = 1 + np.arange(shape[0])
        np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parametrize("array", [False, True])
def test_scalar_declare(loader, array):
    tensor = (
        BufferizedNDArray.from_numpy(np.array(99, dtype=np.int64))
        if array
        else fiber_tensor(element(np.int64(0))).construct(())
    )
    program = declare_program(tensor.ftype)
    asm.AssemblyTypeChecker()(AssemblyGenerator()(program))
    result = NotationCompiler(loader)(program).resize(tensor)
    assert result.shape == ()
    np.testing.assert_array_equal(result.to_numpy(), np.int64(1))


def test_dense_stride_is_updated_inside_kernel(loader):
    tensor = fiber_tensor(dense(dense(dense(element(np.int64(0)))))).construct(
        (0, 0, 0)
    )
    func = AssemblyGenerator()(declare_program(tensor.ftype)).funcs[0]
    level = asm.GetAttr(func.args[0], asm.Literal("lvl"))
    strides = []
    for _ in range(3):
        strides.append(asm.GetAttr(level, asm.Literal("stride")))
        level = asm.GetAttr(level, asm.Literal("lvl"))
    result = asm.Call(asm.Literal(ffuncs.make_tuple), tuple(strides))
    program = asm.Module(
        (
            asm.Function(
                asm.Variable(
                    "resize",
                    asm.AssemblyKernelFType(
                        "resize",
                        tuple(a.result_type for a in func.args),
                        result.result_type,
                    ),
                ),
                func.args,
                asm.Block((*func.body.bodies[:-1], asm.Return(result))),
            ),
        )
    )
    asm.AssemblyTypeChecker()(program)
    kernel = loader(asm.LowerPackedStructSlots()(program)).resize
    for shape in ((2, 3, 4), (1, 4, 2), (2, 0, 3)):
        assert kernel(tensor, *(np.intp(dim) for dim in shape)) == (
            shape[1] * shape[2],
            shape[2],
            1,
        )
