import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import DynamicFill, ffuncs, ftype, is_dynamic
from finch.codegen import CCompiler, NumbaCompiler
from finch.compile import CompilerMode, NotationCompiler, make_extent
from finch.compile.lower import AssemblyGenerator
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    SparseListLevel,
    dense,
    element,
    fiber_tensor,
    sparse_list,
)


@pytest.fixture(
    params=[
        asm.AssemblyInterpreter,
        NumbaCompiler,
        pytest.param(CCompiler, marks=pytest.mark.c_backend),
    ]
)
def loader(request):
    return request.param()


def copy_program(
    source, target, *, declare=True, freeze=True, twice=False, identity_first=False
):
    src, dst = (
        ntn.Variable("source", source.ftype),
        ntn.Variable("target", target.ftype),
    )
    src_slot, dst_slot = (
        ntn.Slot("source_slot", source.ftype),
        ntn.Slot("target_slot", target.ftype),
    )
    start, stop = (
        ntn.Variable("start", ftype(np.intp)),
        ntn.Variable("stop", ftype(np.intp)),
    )
    op = ntn.Literal(ffuncs.add if twice else ffuncs.overwrite)
    idxs = tuple(ntn.Variable(f"i{d}", ftype(np.intp)) for d in range(source.ndim))
    update = ntn.Increment(
        ntn.Access(dst_slot, ntn.Update(op), idxs),
        ntn.Unwrap(ntn.Access(src_slot, ntn.Read(), idxs)),
    )
    body = ntn.Block((update, update)) if twice else update
    if identity_first:
        body = ntn.Block((ntn.Increment(update.lhs, ntn.Literal(np.int64(0))), update))
    for d in reversed(range(source.ndim)):
        lo = start if d == 0 else ntn.Literal(np.intp(0))
        hi = stop if d == 0 else ntn.Literal(np.intp(source.shape[d]))
        body = ntn.Loop(idxs[d], ntn.Call(ntn.Literal(make_extent), (lo, hi)), body)
    stmts = [ntn.Unpack(src_slot, src), ntn.Unpack(dst_slot, dst)]
    if declare:
        stmts.append(
            ntn.Declare(
                dst_slot,
                ntn.Literal(
                    target.ftype.fill_value
                    if is_dynamic(target.ftype.fill_value)
                    else target.fill_value
                ),
                op,
                tuple(ntn.Literal(dim) for dim in target.shape),
            )
        )
    else:
        stmts.append(ntn.Thaw(dst_slot, op))
    stmts.append(body)
    if freeze:
        stmts.append(ntn.Freeze(dst_slot, op))
    stmts.extend(
        (ntn.Repack(src_slot, src), ntn.Repack(dst_slot, dst), ntn.Return(dst))
    )
    args = (src, dst, start, stop)
    return ntn.Module(
        (
            ntn.Function(
                ntn.Variable(
                    "copy_sparse",
                    asm.AssemblyKernelFType(
                        "copy_sparse",
                        tuple(arg.result_type for arg in args),
                        target.ftype,
                    ),
                ),
                args,
                ntn.Block(tuple(stmts)),
            ),
        )
    )


def run_copy(loader, source, target, *, start=0, stop=None, **options):
    program = copy_program(source, target, **options)
    asm.AssemblyTypeChecker()(AssemblyGenerator()(program))
    kernel = NotationCompiler(loader, mode=CompilerMode(debug=True))(
        program
    ).copy_sparse
    return kernel(
        source,
        target,
        np.intp(start),
        np.intp(source.shape[0] if stop is None else stop),
    )


def dense_values(level, pos=0):
    match level:
        case ElementLevel():
            return level.val.load(pos)
        case DenseLevel():
            result = np.full(level.shape, level.fill_value)
            for i in range(level.dimension):
                result[i] = dense_values(level.lvl, pos * level.dimension + i)
            return result
        case SparseListLevel():
            assert level.frozen
            result = np.full(level.shape, level.fill_value)
            for q in range(level.ptr.load(pos), level.ptr.load(pos + 1)):
                result[level.idx.load(q)] = dense_values(level.lvl, q)
            return result
    raise TypeError(level)


def tensors(format_, data, position_type=np.intp, fill=0):
    src_fmt = element(
        np.int64(fill.value if isinstance(fill, DynamicFill) else fill),
        position_type=ftype(position_type),
    )
    dst_fmt = element(
        fill if is_dynamic(fill) else np.int64(fill), position_type=ftype(position_type)
    )
    for char in reversed(format_):
        src_fmt = dense(src_fmt)
        dst_fmt = (
            dense(dst_fmt) if char == "D" else sparse_list(dst_fmt, ftype(np.intp))
        )
    return fiber_tensor(src_fmt).from_numpy(data), fiber_tensor(dst_fmt).construct(
        shape=data.shape, fill_value=fill
    )


@pytest.mark.parametrize("position_type", [np.int32, np.int64])
@pytest.mark.parametrize(
    "format_,data",
    [
        ("S", [0, 2, 0, 4, 5, 0, 6, 7, 8]),
        ("S", [0, 0, 0]),
        ("S", []),
        ("DS", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
        ("SD", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
        ("SD", [[], [], []]),
        ("SS", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ],
)
def test_sparse_copy(loader, position_type, format_, data):
    data = np.array(data, dtype=np.int64)
    source, target = tensors(format_, data, position_type)
    result = run_copy(loader, source, target)
    for tensor in (target, result):
        np.testing.assert_array_equal(dense_values(tensor.lvl), data)
    if format_ == "S":
        np.testing.assert_array_equal(result.lvl.idx.arr, np.flatnonzero(data))
        np.testing.assert_array_equal(result.lvl.val.arr, data[data != 0])


def test_sparse_updates_across_kernels(loader):
    data = np.array([[1, 2, 3], [0, 0, 0], [4, 0, 5], [6, 7, 8]], dtype=np.int64)
    source, target = tensors("DS", data)
    target = run_copy(loader, source, target, stop=2, freeze=False)
    assert not target.lvl.lvl.frozen
    assert target.lvl.lvl.qos_fill == 3
    assert target.lvl.lvl.qos_stop >= 3
    result = run_copy(loader, source, target, start=2, declare=False)
    np.testing.assert_array_equal(dense_values(result.lvl), data)


@pytest.mark.parametrize("loader", [asm.AssemblyInterpreter(), NumbaCompiler()])
def test_sparse_input_to_sparse_output(loader):
    data = np.array([0, 2, 0, 3, 4, 0], dtype=np.int64)
    source, target = tensors("S", data)
    source = run_copy(loader, source, target)
    result = run_copy(loader, source, source.ftype.construct(shape=data.shape))
    np.testing.assert_array_equal(dense_values(result.lvl), data)
    np.testing.assert_array_equal(result.lvl.idx.arr, [1, 3, 4])


def test_sparse_dynamic_fill(loader):
    data = np.array([[7, 2, 7], [7, 7, 7], [3, 7, 4]], dtype=np.int64)
    source, target = tensors("SS", data, fill=DynamicFill(np.int64(7)))
    result = run_copy(loader, source, target)
    np.testing.assert_array_equal(dense_values(result.lvl), data)
    np.testing.assert_array_equal(result.lvl.idx.arr, [0, 2])
    np.testing.assert_array_equal(result.lvl.lvl.val.arr, [2, 3, 4])


@pytest.mark.parametrize("identity_first", [False, True])
def test_sparse_repeated_increment(loader, identity_first):
    data = np.array([0, 2, 0, 3, 4], dtype=np.int64)
    source, target = tensors("S", data)
    result = run_copy(loader, source, target, twice=True, identity_first=identity_first)
    np.testing.assert_array_equal(
        dense_values(result.lvl), data if identity_first else 2 * data
    )
    np.testing.assert_array_equal(result.lvl.idx.arr, [1, 3, 4])


@pytest.mark.parametrize("loader", [asm.AssemblyInterpreter(), NumbaCompiler()])
def test_safe_sparse_repeated_parent(loader):
    source, target = tensors("DS", np.array([[1, 0], [2, 3]], dtype=np.int64))
    target = run_copy(loader, source, target)
    with pytest.raises(AssertionError):
        run_copy(loader, source, target, declare=False)
