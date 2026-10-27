import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import DynamicFill, ffuncs, ftype, is_dynamic
from finch.codegen import CCompiler, MLIRCompiler, NumbaCompiler
from finch.compile import CompilerMode, NotationCompiler, make_extent
from finch.compile.lower import AssemblyGenerator
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    SparseByteMapLevel,
    SparseHashLevel,
    SparseListLevel,
    dense,
    element,
    fiber_tensor,
    sparse_bytemap,
    sparse_hash,
    sparse_list,
)

LEVELS = {
    "D": dense,
    "S": lambda lvl: sparse_list(lvl, ftype(np.intp)),
    "H": sparse_hash,
    "B": sparse_bytemap,
}


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
                # A static fill can't change, so it is declared with itself.
                ntn.FillValue(dst_slot)
                if is_dynamic(target.ftype.fill_type)
                else ntn.Literal(target.fill_value),
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
        case SparseHashLevel():
            assert level.frozen
            result = np.full(level.shape, level.fill_value)
            for r in range(level.ptr.load(pos), level.ptr.load(pos + 1)):
                q = level.perm.load(r)
                result[level.key.load(q)[1]] = dense_values(level.lvl, q)
            return result
        case SparseByteMapLevel():
            result = np.full(level.shape, level.fill_value)
            # An unoccupied parent's pointer is -1.
            if (start := level.ptr.load(pos)) >= 0:
                for r in range(start, level.ptr.load(pos + 1)):
                    q = level.srt.load(r)
                    assert level.tbl.load(q)
                    result[q - pos * level.dimension] = dense_values(level.lvl, q)
            return result
    raise TypeError(level)


def tensors(format_, data, position_type=np.intp, fill=0):
    src_fmt = element(
        np.int64(fill.value if isinstance(fill, DynamicFill) else fill),
        position_type=ftype(position_type),
    )
    dst_fmt = element(
        fill.ftype if isinstance(fill, DynamicFill) else np.int64(fill),
        position_type=ftype(position_type),
    )
    for char in reversed(format_):
        src_fmt = dense(src_fmt)
        dst_fmt = LEVELS[char](dst_fmt)
    return fiber_tensor(src_fmt).from_numpy(data), fiber_tensor(dst_fmt).construct(
        shape=data.shape, fill_value=fill
    )


@pytest.mark.parametrize("position_type", [np.int32, np.int64])
@pytest.mark.parametrize(
    "loader",
    [
        asm.AssemblyInterpreter(),
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
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


_U64 = (1 << 64) - 1


def _hash_u64(a):
    """Julia's `hash_64_64`."""
    a = (~a + (a << 21)) & _U64
    a ^= a >> 24
    a = (a + (a << 3) + (a << 8)) & _U64
    a ^= a >> 14
    a = (a + (a << 2) + (a << 4)) & _U64
    a ^= a >> 28
    return (a + (a << 31)) & _U64


def hash_probe(level, p, i):
    """The bucket, the first slot, and the control byte where Finch.jl probes
    for the zero-based key `(p, i)`."""
    x = (int(level.pos_multiplier) * (p + 1) + _hash_u64(i + 1)) & _U64
    log_b = int(level.subtables).bit_length() - 1
    bucket = x >> (64 - log_b)
    width = level.tbl_ctrl.length() >> log_b
    y = _hash_u64(x)
    return bucket, bucket * width + ((y >> 7) & (width - 1)), 0x80 | (y & 0x7F)


def hash_slots(level, p, i):
    """The slots of `(p, i)`'s probe, up to the first empty one."""
    bucket, h, _ = hash_probe(level, p, i)
    width = level.tbl_ctrl.length() // int(level.subtables)
    while True:
        yield h
        if level.tbl_ctrl.load(h) == 0:
            return
        h = bucket * width + (h + 1 - bucket * width) % width


def hash_lookup(level, p, i):
    if level.tbl_ctrl.length() == 0:
        return None
    _, _, ctrl = hash_probe(level, p, i)
    for h in hash_slots(level, p, i):
        if level.tbl_ctrl.load(h) == ctrl:
            q = level.tbl.load(h)
            if tuple(level.key.load(q))[:2] == (p, i):
                return q
    return None


def check_hash_table(level):
    """Retained keys are found where Finch.jl probes for them, and nothing else
    occupies the table."""
    keys = [tuple(level.key.load(q)) for q in range(level.key.length())]
    live = {q: (int(p), int(i)) for q, (p, i, state) in enumerate(keys) if state}
    assert all(keys[q][2] == 0x80 for q in live)
    assert np.count_nonzero(level.tbl_ctrl.arr) == len(live)
    totals = np.zeros(level.subtables, dtype=np.intp)
    for q, (p, i) in live.items():
        assert hash_lookup(level, p, i) == q
        totals[hash_probe(level, p, i)[0]] += 1
    np.testing.assert_array_equal(level.tbl_count.arr[: level.subtables], totals)


def sublevels(level):
    while not isinstance(level, ElementLevel):
        yield level
        level = level.lvl


def check_levels(tensor, data):
    np.testing.assert_array_equal(dense_values(tensor.lvl), data)
    for level in sublevels(tensor.lvl):
        match level:
            case SparseHashLevel():
                check_hash_table(level)
            case SparseByteMapLevel():
                assert np.count_nonzero(level.tbl.arr) == level.srt.length()
                np.testing.assert_array_equal(np.sort(level.srt.arr), level.srt.arr)


HASH_BYTEMAP_CASES = [
    ("H", [0, 2, 0, 4, 5, 0, 6, 7, 8]),
    ("B", [0, 2, 0, 4, 5, 0, 6, 7, 8]),
    ("H", [0, 0, 0]),
    ("B", []),
    ("DH", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("DB", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("HD", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("BD", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("HH", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("BB", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("HB", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("BH", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
    ("SH", [[0, 2, 0], [0, 0, 0], [3, 0, 4], [0, 0, 0]]),
]


@pytest.mark.parametrize("position_type", [np.int32, np.int64])
@pytest.mark.parametrize(
    "loader",
    [
        asm.AssemblyInterpreter(),
        NumbaCompiler(),
        pytest.param(CCompiler(), marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler(), marks=pytest.mark.mlir_backend),
    ],
)
@pytest.mark.parametrize("format_,data", HASH_BYTEMAP_CASES)
def test_hash_bytemap_copy(loader, position_type, format_, data):
    data = np.array(data, dtype=np.int64)
    source, target = tensors(format_, data, position_type)
    result = run_copy(loader, source, target)
    for tensor in (target, result):
        check_levels(tensor, data)
    if "S" in format_ and (position_type == np.int32 or isinstance(loader, CCompiler)):
        return  # SparseList reads need intp positions and scansearch.
    # Reading back exercises the walk unfurls.
    _, dense_target = tensors("D" * data.ndim, data, position_type)
    np.testing.assert_array_equal(
        dense_values(run_copy(loader, result, dense_target).lvl), data
    )


@pytest.mark.parametrize("format_", ["H", "B"])
@pytest.mark.parametrize("start", [1, 4, 8])
def test_hash_bytemap_seek(loader, format_, start):
    data = np.array([0, 2, 0, 4, 5, 0, 6, 7, 0], dtype=np.int64)
    source, target = tensors(format_, data)
    source = run_copy(loader, source, target)
    _, dense_target = tensors("D", data)
    result = run_copy(loader, source, dense_target, start=start)
    np.testing.assert_array_equal(
        dense_values(result.lvl), np.where(np.arange(9) < start, 0, data)
    )


@pytest.mark.parametrize("format_", ["H", "B", "DH", "DB"])
@pytest.mark.parametrize("identity_first", [False, True])
def test_hash_bytemap_repeated_increment(loader, format_, identity_first):
    data = np.array([[0, 2, 0, 3, 4], [5, 0, 0, 0, 6]], dtype=np.int64)
    if len(format_) == 1:
        data = data[0]
    source, target = tensors(format_, data)
    result = run_copy(loader, source, target, twice=True, identity_first=identity_first)
    check_levels(result, data if identity_first else 2 * data)


@pytest.mark.parametrize("format_", ["DH", "DB"])
def test_hash_bytemap_updates_across_kernels(loader, format_):
    data = np.array([[1, 2, 3], [0, 0, 0], [4, 0, 5], [6, 7, 8]], dtype=np.int64)
    source, target = tensors(format_, data)
    target = run_copy(loader, source, target, stop=2, freeze=False)
    result = run_copy(loader, source, target, start=2, declare=False)
    check_levels(result, data)


@pytest.mark.parametrize("format_", ["H", "B", "DH", "DB"])
def test_hash_bytemap_redeclare(loader, format_):
    rng = np.random.default_rng(0)
    shape = (6, 40) if len(format_) == 2 else (240,)
    source, target = tensors(format_, np.zeros(shape, dtype=np.int64))
    for density in (0.5, 0.02, 0.9, 0.0, 0.1):
        data = rng.integers(1, 9, shape) * (rng.random(shape) < density)
        source = tensors(format_, data)[0]
        target = run_copy(loader, source, target)
        check_levels(target, data)


@pytest.mark.parametrize("subtables", [1, 4])
def test_hash_rehash_subtables(loader, subtables):
    rng = np.random.default_rng(1)
    data = rng.integers(1, 9, (7, 300)) * (rng.random((7, 300)) < 0.3)
    source, target = tensors("DH", data)
    level = target.lvl.lvl
    target.lvl.lvl = SparseHashLevel(level.lvl, level.dimension, subtables=subtables)
    result = run_copy(loader, source, target)
    check_levels(result, data)
    assert result.lvl.lvl.tbl_count.length() % subtables == 0


def test_hash_declare_clears_sparse_table(loader):
    # Once few entries remain in a large table, declaring clears them one by one.
    data = np.zeros(5000, dtype=np.int64)
    data[::2] = 1
    source, target = tensors("H", data)
    target = run_copy(loader, source, target)
    for index in (3, 4000):
        data = np.zeros(5000, dtype=np.int64)
        data[index] = 7
        target = run_copy(loader, tensors("H", data)[0], target)
        assert target.lvl.tbl_ctrl.length() >= 4096 * target.lvl.perm.length()
        check_levels(target, data)


def test_hash_thaw_reuses_holes(loader):
    data = np.array([0, 2, 0, 4], dtype=np.int64)
    source, target = tensors("H", data)
    level = run_copy(loader, source, target).lvl
    # Free child position 0, as a merge may leave it, and rebuild the table
    # around position 1.
    q_hole = int(hash_lookup(level, 0, 1))
    q_live = 1 - q_hole
    level.key.arr[q_hole] = (0, 0, 0)
    level.lvl.val.arr[q_hole] = 0
    level.tbl_ctrl.arr[:] = 0
    _, h, ctrl = hash_probe(level, 0, 3)
    *_, h = hash_slots(level, 0, 3)
    level.tbl_ctrl.arr[h] = ctrl
    level.tbl.arr[h] = q_live
    level.perm.arr = np.array([q_live], dtype=level.perm.arr.dtype)
    level.ptr.arr = np.array([0, 1], dtype=level.ptr.arr.dtype)
    level.tbl_count.arr = np.array([1, 0], dtype=np.intp)
    check_levels(FiberTensor(level), [0, 0, 0, 4])

    data = np.array([0, 7, 9, 4], dtype=np.int64)
    result = run_copy(loader, tensors("H", data)[0], FiberTensor(level), declare=False)
    check_levels(result, data)
    assert result.lvl.key.length() == 3
    assert hash_lookup(result.lvl, 0, 1) == q_hole


@pytest.mark.parametrize("level", [sparse_hash, sparse_bytemap])
def test_hash_bytemap_narrow_dimension(loader, level):
    data = np.array([[0, 2, 0], [0, 0, 0], [3, 0, 4]], dtype=np.int64)
    leaf = element(np.int64(0), position_type=ftype(np.int32))
    source = fiber_tensor(dense(dense(leaf))).from_numpy(data)
    target = fiber_tensor(dense(level(leaf, ftype(np.int32)))).construct(
        shape=data.shape
    )
    result = run_copy(loader, source, target)
    check_levels(result, data)
    _, dense_target = tensors("DD", data, np.int32)
    np.testing.assert_array_equal(
        dense_values(run_copy(loader, result, dense_target).lvl), data
    )
