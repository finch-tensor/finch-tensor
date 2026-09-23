import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype
from finch.codegen import CCompiler, MLIRGenerator, NumbaCompiler, NumpyBuffer
from finch.codegen.c_codegen import construct_from_c, serialize_to_c
from finch.codegen.numba_codegen import construct_from_numba, serialize_to_numba
from finch.compile import CompilerMode, NotationCompiler
from finch.compile.lower import AssemblyGenerator
from finch.tensor import DenseLevel, ElementLevel, FiberTensor, SparseListLevel, element


@pytest.fixture(
    params=[
        asm.AssemblyInterpreter,
        NumbaCompiler,
        pytest.param(CCompiler, marks=pytest.mark.c_backend),
    ]
)
def loader(request):
    return request.param()


def lifecycle_program(tensor, action):
    fmt = tensor.ftype
    arg = ntn.Variable("tensor", fmt)
    slot = ntn.Slot("tensor_slot", fmt)
    op = ntn.Literal(ffuncs.add)
    body = [ntn.Unpack(slot, arg)]
    if action == "declare":
        body.append(
            ntn.Declare(
                slot,
                ntn.Literal(np.int64(0)),
                op,
                tuple(ntn.Literal(dim) for dim in tensor.shape),
            )
        )
    else:
        body.append(ntn.Thaw(slot, op))
    if action != "thaw":
        body.append(ntn.Freeze(slot, op))
    body.extend((ntn.Repack(slot, arg), ntn.Return(arg)))
    return ntn.Module(
        (
            ntn.Function(
                ntn.Variable(
                    "lifecycle", asm.AssemblyKernelFType("lifecycle", (fmt,), fmt)
                ),
                (arg,),
                ntn.Block(tuple(body)),
            ),
        )
    )


def run_lifecycle(loader, tensor, action):
    program = lifecycle_program(tensor, action)
    asm.AssemblyTypeChecker()(AssemblyGenerator()(program))
    return NotationCompiler(loader, mode=CompilerMode(debug=True))(program).lifecycle(
        tensor
    )


def sparse_tensor(ptr, idx, values, *, position_type=np.intp, frozen=True):
    leaf = ElementLevel(
        element(np.int64(0), position_type=ftype(position_type)),
        NumpyBuffer(np.array(values, dtype=np.int64)),
    )
    sparse = SparseListLevel(
        leaf,
        np.intp(8),
        NumpyBuffer(np.array(ptr, dtype=position_type)),
        NumpyBuffer(np.array(idx, dtype=np.intp)),
        frozen=frozen,
    )
    return FiberTensor(DenseLevel(sparse, np.intp(len(ptr) - 1)))


@pytest.mark.parametrize("position_type", [np.int32, np.int64])
@pytest.mark.parametrize(
    "ptr,idx",
    [
        ([0], []),
        ([0, 0, 0], []),
        ([0, 0, 2, 2, 3, 3], [1, 3, 5]),
    ],
)
def test_sparse_thaw_freeze_across_kernels(loader, position_type, ptr, idx):
    values = list(range(1, len(idx) + 1))
    tensor = sparse_tensor(ptr, idx, values, position_type=position_type)
    thawed = run_lifecycle(loader, tensor, "thaw")
    counts = np.concatenate(([0], np.diff(ptr)))
    for result in (tensor, thawed):
        level = result.lvl.lvl
        np.testing.assert_array_equal(level.ptr.arr, counts)
        assert not level.frozen
        assert level.qos_fill == level.qos_stop == len(idx)
        assert level.prev_pos == max(np.flatnonzero(counts), default=0)
        assert ftype(level.qos_fill) == ftype(position_type)

    # Resuming an already thawed level must retain its counts and cursors.
    resumed = run_lifecycle(loader, thawed, "thaw")
    np.testing.assert_array_equal(resumed.lvl.lvl.ptr.arr, counts)
    frozen = run_lifecycle(loader, resumed, "freeze")
    for result in (resumed, frozen):
        level = result.lvl.lvl
        assert level.frozen
        np.testing.assert_array_equal(level.ptr.arr, ptr)
        np.testing.assert_array_equal(level.idx.arr, idx)
        np.testing.assert_array_equal(level.lvl.val.arr, values)


def test_sparse_freeze_trims_storage(loader):
    tensor = sparse_tensor(
        [0, 2, 0, 1, 0, 99], [1, 3, 5, 99, 99], [10, 20, 30, 99, 99, 99], frozen=False
    )
    tensor.lvl.dimension = np.intp(4)
    result = run_lifecycle(loader, tensor, "freeze")
    for value in (tensor, result):
        level = value.lvl.lvl
        np.testing.assert_array_equal(level.ptr.arr, [0, 2, 2, 3, 3])
        np.testing.assert_array_equal(level.idx.arr, [1, 3, 5])
        np.testing.assert_array_equal(level.val.arr, [10, 20, 30])
        assert level.qos_fill == level.qos_stop == 3


def test_sparse_declare_clears_counts(loader):
    tensor = sparse_tensor([0, 2, 3], [1, 3, 5], [10, 20, 30])
    result = run_lifecycle(loader, tensor, "declare")
    for value in (tensor, result):
        level = value.lvl.lvl
        np.testing.assert_array_equal(level.ptr.arr, [0, 0, 0])
        assert level.idx.length() == level.val.length() == 0
        assert level.qos_fill == level.qos_stop == level.prev_pos == 0
        assert level.frozen


def test_nested_sparse_dense_lifecycle(loader):
    inner = sparse_tensor(
        [0, 1, 0, 2, 0], [1, 2, 4, 99], [10, 20, 30, 99], frozen=False
    ).lvl.lvl
    inner.qos_fill = np.intp(3)
    outer = SparseListLevel(
        DenseLevel(inner, np.intp(2)),
        np.intp(5),
        NumpyBuffer(np.array([0, 2], dtype=np.intp)),
        NumpyBuffer(np.array([0, 3, 99], dtype=np.intp)),
        qos_fill=np.intp(2),
        frozen=False,
    )
    tensor = run_lifecycle(loader, FiberTensor(outer), "freeze")
    outer, inner = tensor.lvl, tensor.lvl.lvl.lvl
    np.testing.assert_array_equal(outer.ptr.arr, [0, 2])
    np.testing.assert_array_equal(outer.idx.arr, [0, 3])
    np.testing.assert_array_equal(inner.ptr.arr, [0, 1, 1, 3, 3])
    np.testing.assert_array_equal(inner.idx.arr, [1, 2, 4])
    np.testing.assert_array_equal(inner.val.arr, [10, 20, 30])
    tensor = run_lifecycle(loader, tensor, "thaw")
    outer, inner = tensor.lvl, tensor.lvl.lvl.lvl
    assert not outer.frozen and not inner.frozen
    assert outer.qos_fill == outer.qos_stop == 2
    assert inner.qos_fill == inner.qos_stop == 3
    assert outer.prev_pos == 1 and inner.prev_pos == 3
    np.testing.assert_array_equal(inner.ptr.arr, [0, 1, 0, 2, 0])


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_dense_element_freeze(loader, shape):
    count = int(np.prod(shape))
    level = ElementLevel(element(np.int64(0)), NumpyBuffer(np.arange(count + 4)))
    for dim in reversed(shape):
        level = DenseLevel(level, np.intp(dim))
    tensor = FiberTensor(level)
    result = run_lifecycle(loader, tensor, "freeze")
    for value in (tensor, result):
        np.testing.assert_array_equal(value.lvl.val.arr, np.arange(count))


def test_mlir_freeze_checks_resize():
    tensor = FiberTensor(
        DenseLevel(
            ElementLevel(element(np.int64(0)), NumpyBuffer(np.arange(3))), np.intp(3)
        )
    )
    program = asm.LowerPackedStructSlots()(
        AssemblyGenerator()(lifecycle_program(tensor, "freeze"))
    )
    code = MLIRGenerator()(program).code
    assert "cf.assert" in code
    assert "memref.realloc" not in code


@pytest.mark.parametrize(
    "serialize,construct",
    [
        (serialize_to_numba, construct_from_numba),
        pytest.param(serialize_to_c, construct_from_c, marks=pytest.mark.c_backend),
    ],
)
@pytest.mark.parametrize("frozen", [False, True])
def test_sparse_serialization(serialize, construct, frozen):
    ptr = [0, 2, 3, 3] if frozen else [0, 2, 1, 0]
    tensor = sparse_tensor(ptr, [1, 3, 5, 99], [10, 20, 30, 99], frozen=frozen)
    level = tensor.lvl.lvl
    level.qos_fill = np.intp(3)
    level.qos_stop = np.intp(4)
    level.prev_pos = np.intp(2)
    result = construct(tensor.ftype, serialize(tensor.ftype, tensor)).lvl.lvl
    assert result.frozen == frozen
    assert result.qos_fill == 3
    assert result.qos_stop == 4
    assert result.prev_pos == 2
    np.testing.assert_array_equal(result.ptr.arr, ptr)
    np.testing.assert_array_equal(result.idx.arr, level.idx.arr)
    np.testing.assert_array_equal(result.val.arr, level.val.arr)
