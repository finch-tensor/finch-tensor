import pytest

import numpy as np

import finch as ft
import finch.finch_notation as ntn
from finch import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseByteMapLevel,
    SparseCOOLevel,
    SparseListLevel,
    element,
    ffuncs,
    ftype,
)
from finch.algebra import DynamicFill, DynamicFillError, DynamicFillFType
from finch.autoschedule import (
    CompilerFormLowerer,
    DefaultLogicFactorizer,
    DefaultLoopOrderer,
    FDFormatter,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    with_default_scheduler,
)
from finch.autoschedule.tensor_stats import FDStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.interop import jl_tensor_to_python, tensor_to_jl
from finch.compile_jl.julia import jl, julia_available
from finch.finch_assembly import AssemblyKernelFType
from finch.tensor import BufferizedNDArray
from finch.tensor.patterns import (
    ChunkMaskTensor,
    EyeTensor,
    LowerTriangleTensor,
    OddEvenMergeSortLowerMaskTensor,
    OddEvenMergeSortPartnerMaskTensor,
    OneHotMaskTensor,
    PairCarryTensor,
    PairSumTensor,
    ParityMaskTensor,
    RandomMaskTensor,
    RepeatTensor,
    ReshapeMaskTensor,
    ReverseTensor,
    RollTensor,
    SplitMaskTensor,
    UpperTriangleTensor,
)

DTYPE = np.int64
ROWS = np.intp(3)
COLS = np.intp(3)
ROW_PTR = NumpyBuffer(np.array([0, 2, 3, 5], dtype=np.intp))
COL_IDX = NumpyBuffer(np.array([0, 2, 1, 0, 2], dtype=np.intp))
STORED_VALUES = np.array([1, 2, 3, 4, 5], dtype=DTYPE)
EXPECTED_ROW_SUMS = np.array([3, 3, 9], dtype=DTYPE)


def _requires_julia_backend():
    if not julia_available():
        pytest.skip("the julia extra (juliapkg, juliacall) is not installed")


@pytest.mark.parametrize("fill_wrapper", [DynamicFill, DynamicFillFType])
def test_julia_nested_tuple_fill_placeholder(fill_wrapper):
    from finch.compile_jl.types import _julia_literal

    value = (np.float64(3), (np.intp(2), np.bool_(True)))
    assert _julia_literal(fill_wrapper(value)) == "(0.0, (0, false))"


def test_compile_julia_preserves_definition_type():
    _requires_julia_backend()

    tensor = ft.asarray(np.arange(6, dtype=np.int64).reshape(2, 3))
    arg = ntn.Variable("tensor", tensor.ftype)
    result = ntn.Call(ntn.Literal(ffuncs.make_tuple), (arg,))
    definitions = tuple(
        ntn.Function(
            ntn.Variable(
                "identity",
                AssemblyKernelFType("identity", (arg.result_type,), result.result_type),
            ),
            (arg,),
            ntn.Block((ntn.Return(result),)),
        )
        for _ in range(2)
    )
    compiler = FinchJLCompiler()
    first, second = (
        compiler(ntn.Module((definition,))).identity for definition in definitions
    )
    assert ftype(first) == definitions[0].name.result_type
    assert ftype(second) == definitions[1].name.result_type
    assert ftype(first) != ftype(second)
    for kernel in (first, second):
        np.testing.assert_array_equal(kernel(tensor)[0].to_numpy(), tensor.to_numpy())


@pytest.mark.parametrize(
    "fill, value",
    [
        (np.int64(0), 0),
        (np.int64(-1), 7),
        (np.bool_(False), True),
        (np.float64(0), -0.0),
        (np.float64(np.inf), 7),
        (np.float64(np.nan), 7),
    ],
)
def test_compile_julia_init_write(fill, value):
    _requires_julia_backend()
    from finch.compile_jl.compiler import FinchJLGenerator
    from finch.compile_jl.julia import jl

    tensor = BufferizedNDArray.from_numpy(np.array(fill), fill_value=fill)
    arg = ntn.Variable("output", tensor.ftype)
    slot = ntn.Slot("output_slot", tensor.ftype)
    operator = ffuncs.init_write(fill)
    op = ntn.Literal(operator)
    result = ntn.Call(ntn.Literal(ffuncs.make_tuple), (arg,))
    value = type(fill)(value)
    call = ntn.Call(op, (ntn.Literal(fill), ntn.Literal(value)))
    generated = FinchJLGenerator().generate_julia(call)
    np.testing.assert_equal(jl.seval(generated), operator(fill, value))
    program = ntn.Module(
        (
            ntn.Function(
                ntn.Variable(
                    "write",
                    AssemblyKernelFType(
                        "write", (arg.result_type,), result.result_type
                    ),
                ),
                (arg,),
                ntn.Block(
                    (
                        ntn.Unpack(slot, arg),
                        ntn.Declare(slot, ntn.Literal(fill), op, ()),
                        ntn.Increment(
                            ntn.Access(slot, ntn.Update(op), ()), ntn.Literal(value)
                        ),
                        ntn.Freeze(slot, op),
                        ntn.Repack(slot, arg),
                        ntn.Return(result),
                    )
                ),
            ),
        )
    )
    actual = FinchJLCompiler()(program).write(tensor)[0]
    np.testing.assert_array_equal(actual.to_numpy(), value)


def test_compile_julia_init_write_rejects_dynamic_fill():
    from finch.compile_jl.compiler import FinchJLGenerator

    tensor = ft.asarray(np.zeros((), dtype=np.int64))
    op = ntn.Literal(ffuncs.init_write(DynamicFill(np.int64(3))))
    update = ntn.Increment(
        ntn.Access(ntn.Variable("output", tensor.ftype), ntn.Update(op), ()),
        ntn.Literal(np.int64(3)),
    )
    with pytest.raises(DynamicFillError, match="requires a static fill"):
        FinchJLGenerator().generate_julia(update)


@pytest.mark.parametrize(
    "mask",
    [
        *(
            cls((3, 5), k=k)
            for cls in (EyeTensor, UpperTriangleTensor, LowerTriangleTensor)
            for k in (-5, -1, 0, 2, 5)
        ),
        PairSumTensor((3, 5)),
        PairCarryTensor((5, 3)),
        ReverseTensor((3, 5)),
        *(RollTensor((7, 3), k=k) for k in (-4, 0, 5)),
        *(RepeatTensor((3, 7), k=k) for k in (-1, 0, 2)),
        ChunkMaskTensor((4, 10), b=3),
        ChunkMaskTensor((3, 6), b=2, dtype=np.int32),
        ChunkMaskTensor((3, 3), b=1),
        ChunkMaskTensor((1, 2), b=5),
        ChunkMaskTensor((0, 0), b=3),
        SplitMaskTensor((3, 10)),
        SplitMaskTensor((3, 10), dtype=np.intp),
        SplitMaskTensor((3, 6), dtype=np.float64),
        SplitMaskTensor((5, 3)),
        SplitMaskTensor((1, 3)),
        SplitMaskTensor((3, 0)),
        RandomMaskTensor((), 0.4, seed=42),
        RandomMaskTensor((), 0.5, seed=1 << 63),
        RandomMaskTensor((0,), 0.5, seed=42),
        RandomMaskTensor((3, 0), 0.5, seed=42),
        RandomMaskTensor(9, 0.0, seed=42),
        RandomMaskTensor((3, 5), 1.0, seed=42),
        RandomMaskTensor((3, 5), 0.5, seed=42),
        RandomMaskTensor((2, 3, 4), 0.4, seed=(1 << 64) - 1, dtype=np.intp),
        *(
            OddEvenMergeSortPartnerMaskTensor((7, 7), p=p, k=k)
            for p, k in ((1, 1), (2, 1), (4, 2))
        ),
        *(
            OddEvenMergeSortLowerMaskTensor(7, p=p, k=k)
            for p, k in ((1, 1), (2, 1), (4, 2))
        ),
        *(OneHotMaskTensor(5, index=i) for i in (-1, 0, 2, 5)),
        *(ParityMaskTensor(5, parity=p) for p in (-1, 0, 1, 2)),
        ReshapeMaskTensor((2, 3), (3, 2)),
        ReshapeMaskTensor((), (1,)),
        ReshapeMaskTensor((1,), ()),
        ReshapeMaskTensor((), ()),
        ReshapeMaskTensor((0, 3), (0,)),
        RollTensor((3, 0), k=2),
    ],
)
def test_compile_julia_pattern_masks(mask):
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA
    from finch.compile_jl.interop import tensor_to_jl
    from finch.compile_jl.julia import jl
    from finch.compile_jl.types import ftype_to_jl_constructor_str

    jl_mask = tensor_to_jl(mask)
    prototype = jl.seval(ftype_to_jl_constructor_str(mask.ftype))
    assert jl.typeof(jl_mask) == jl.typeof(prototype)

    data = np.full(mask.shape or (1,), 2, dtype=np.int64)
    expected = data + np.array(
        [mask[idx].item() for idx in np.ndindex(mask.shape)]
    ).reshape(mask.shape)
    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(ft.defer(mask) + ft.defer(data))

    np.testing.assert_array_equal(result.to_numpy(), expected)


_MATRIX = np.arange(1.0, 13.0).reshape(3, 4)
_VECTOR = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])


@pytest.mark.parametrize(
    "op, np_op, arg",
    [
        (lambda a: ft.flip(a, axis=1), lambda a: np.flip(a, axis=1), _MATRIX),
        (lambda a: ft.roll(a, 2, axis=1), lambda a: np.roll(a, 2, axis=1), _MATRIX),
        (lambda a: ft.roll(a, -1, axis=0), lambda a: np.roll(a, -1, axis=0), _MATRIX),
        (
            lambda a: ft.repeat(a, 3, axis=1),
            lambda a: np.repeat(a, 3, axis=1),
            _MATRIX,
        ),
        (
            lambda a: ft.cumulative_sum(a, axis=1),
            lambda a: np.cumsum(a, axis=1),
            _MATRIX,
        ),
        (
            lambda a: ft.cumulative_sum(a, include_initial=True),
            lambda a: np.concatenate([[0.0], np.cumsum(a)]),
            _VECTOR,
        ),
        (ft.sort, np.sort, _VECTOR),
        (
            lambda a: ft.argsort(a, stable=True),
            lambda a: np.argsort(a, kind="stable"),
            _VECTOR,
        ),
        (lambda a: ft.triu(a, k=1), lambda a: np.triu(a, k=1), _MATRIX),
        (lambda a: ft.tril(a, k=-1), lambda a: np.tril(a, k=-1), _MATRIX),
    ],
)
def test_compile_julia_pattern_interface(op, np_op, arg):
    # These functions select, reduce, and fill through pattern masks, fill
    # tensors, and choose, which all lower specially on the Julia backend.
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA

    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(op(ft.defer(arg)))

    np.testing.assert_array_equal(result.to_numpy(), np_op(arg))


@pytest.mark.parametrize(
    "arg, dtype",
    [
        (np.array([-2.7, -0.5, 0.0, 1.5, 3.9]), np.int64),
        (np.array([-2.7, -0.5, 0.0, 1.5, 3.9]), np.bool_),
        (np.array([-2.7, -0.5, 0.0, 1.5, 3.9]), np.float32),
        (np.array([0, 1, 2, -3, 300]), np.bool_),
        (np.array([0, 1, 2, -3, 300]), np.uint8),
        (np.array([True, False, True]), np.int64),
    ],
)
def test_compile_julia_astype_matches_numpy(arg, dtype):
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA

    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(ft.astype(ft.defer(arg), dtype))

    expected = arg.astype(dtype)
    assert result.to_numpy().dtype == expected.dtype
    np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parametrize(
    "op",
    [
        ft.sum,
        lambda a: ft.sum(a, axis=1),
        lambda a: ft.sum(a, axis=1, dtype=np.int64),
        lambda a: ft.prod(a, axis=0),
        lambda a: ft.matmul(a, ft.permute_dims(a, (1, 0))),
        lambda a: a + a,
        lambda a: a * a,
    ],
    ids=["sum", "sum_axis", "sum_int", "prod_axis", "matmul", "add", "mul"],
)
def test_compile_julia_bool_arithmetic_matches_interpreter(op):
    # Julia promotes Bool arithmetic to Int, but Finch keeps it Bool.
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA, INTERPRET_NOTATION

    arg = np.array([[True, True, False], [True, True, True]])
    with with_default_scheduler(INTERPRET_NOTATION):
        expected = ft.compute(op(ft.defer(arg))).to_numpy()
    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(op(ft.defer(arg))).to_numpy()

    np.testing.assert_array_equal(result, expected)


def test_compile_julia_numeric_pattern_mask():
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA
    from finch.compile_jl.interop import tensor_to_jl
    from finch.compile_jl.julia import jl

    mask = EyeTensor((3, 5), k=1, dtype=np.int64)
    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(ft.bitwise_invert(ft.defer(mask)))

    actual = jl.Array(tensor_to_jl(result)).to_numpy().T
    expected = np.bitwise_invert(np.eye(3, 5, k=1, dtype=np.int64))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_random_mask_bit_mixing_matches_julia():
    _requires_julia_backend()
    from finch.compile_jl.julia import jl
    from finch.tensor.patterns import _randommask_mix, _randommask_uniform

    mix = jl.seval("seed -> Finch.randommask_mix(UInt64(seed))")
    uniform = jl.seval("seed -> Finch.randommask_uniform(UInt64(seed))")
    rng = np.random.default_rng(42)
    seeds = [0, 1, 1 << 63, (1 << 64) - 1]
    seeds.extend(map(int, rng.integers(0, 1 << 64, size=32, dtype=np.uint64)))
    for seed in seeds:
        assert _randommask_mix(seed) == int(mix(seed))
        assert _randommask_uniform(seed) == uniform(seed)


@pytest.mark.parametrize("shape", [(), (7,), (4, 5), (2, 3, 4)])
def test_random_mask_matches_julia(shape):
    _requires_julia_backend()
    from finch.compile_jl.julia import jl

    materialize = jl.seval("mask -> copyto!(zeros(Bool, size(mask)), mask)")
    for seed in (0, 1, 42, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
        for p in (0.0, np.nextafter(0.0, 1.0), 0.4, 0.5, np.nextafter(1.0, 0.0), 1.0):
            mask = RandomMaskTensor(shape, p, seed=seed)
            # Julia masks have the reversed axes of Python masks.
            native = jl.Finch.randommask(shape[::-1], p, seed=jl.UInt64(seed))
            expected = np.transpose(materialize(native).to_numpy())
            actual = np.array([mask[idx].item() for idx in np.ndindex(shape)])
            np.testing.assert_array_equal(actual.reshape(shape), expected)


def test_random_mask_large_coordinates_match_julia():
    _requires_julia_backend()
    from finch.compile_jl.julia import jl

    shape = (int(np.iinfo(np.intp).max), int(np.iinfo(np.intp).max), 8)
    large_index = (shape[0] + 1) // 2
    for seed in (42, (1 << 64) - 1):
        mask = RandomMaskTensor(shape, 0.5, seed=seed)
        native = jl.Finch.randommask(shape[::-1], 0.5, seed=jl.UInt64(seed))
        for idx in (
            (2, 3, 1),
            tuple(dim - 1 for dim in shape),
            (large_index, large_index, 7),
        ):
            assert mask[idx].item() == jl.getindex(native, *(i + 1 for i in idx[::-1]))


def test_compile_julia_pattern_lowering(file_regression):
    _requires_julia_backend()
    from finch.compile_jl.compiler import (
        FinchJLCompiler,
        FinchJLGenerator,
        handle_fills,
    )

    class RecordingJLCompiler(FinchJLCompiler):
        def __init__(self):
            super().__init__()
            self.sources = []

        def __call__(self, prgm):
            for func in prgm.children:
                func, _ = handle_fills(func)
                self.sources.append(FinchJLGenerator()(func))
            return super().__call__(prgm)

    compiler = RecordingJLCompiler()
    scheduler = _compile_julia_fd(
        FDFormatter(CompilerFormLowerer(LogicCompiler(compiler)))
    )
    for mask in (
        EyeTensor((3, 5), k=1),
        UpperTriangleTensor((3, 5), k=-1),
        PairSumTensor((3, 5)),
        RollTensor((7, 3), k=-4),
        OneHotMaskTensor(5, index=2),
        ReshapeMaskTensor((2, 3), (3, 2)),
        ChunkMaskTensor((4, 10), b=3),
        SplitMaskTensor((3, 10)),
        RandomMaskTensor((), 0.4, seed=42),
        RandomMaskTensor(7, 0.25, seed=42),
        RandomMaskTensor((3, 5), 0.5, seed=(1 << 64) - 1, dtype=np.intp),
    ):
        compiler.sources.append(f"# {type(mask).__name__}")
        data = np.full(mask.shape or (1,), 2, dtype=np.int64)
        with with_default_scheduler(scheduler):
            ft.compute(ft.defer(mask) + ft.defer(data))

    file_regression.check("\n\n".join(compiler.sources), extension=".jl")


def test_compile_julia_sampling_stats_lowering(monkeypatch, file_regression):
    _requires_julia_backend()
    from finch.autoschedule import default_schedulers
    from finch.autoschedule.tensor_stats import SamplingStatsFactory
    from finch.compile_jl.compiler import (
        FinchJLCompiler,
        FinchJLGenerator,
        handle_fills,
    )
    from finch.compile_jl.julia import jl
    from finch.finch_logic import Field, LogicSimplify

    class RecordingJLCompiler(FinchJLCompiler):
        def __init__(self):
            super().__init__()
            self.sources = []

        def __call__(self, prgm):
            for func in prgm.children:
                func, _ = handle_fills(func)
                source = FinchJLGenerator()(func)
                expanded = jl.seval(source.removeprefix("eval(").removesuffix(")"))
                self.sources.append(
                    f"# Finch kernel\n{source}\n\n"
                    f"# Generated Julia\n{jl.string(expanded)}"
                )
            return super().__call__(prgm)

    compiler = RecordingJLCompiler()
    scheduler = LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(
                LogicSimplify(
                    DefaultLoopOrderer(
                        FDFormatter(CompilerFormLowerer(LogicCompiler(compiler)))
                    )
                )
            ),
            stats_factory=FDStatsFactory(),
        )
    )
    monkeypatch.setattr(default_schedulers, "NON_RECURSIVE_SCHEDULER", scheduler)
    i, j = Field("i"), Field("j")
    data = np.arange(35, dtype=DTYPE).reshape(5, 7) % 3
    factory = SamplingStatsFactory()
    factory._rng = np.random.default_rng(42)
    stats = factory(_csr_tensor(data), (i, j), [0.5, 0.5])
    rows, cols = (
        np.array(
            [factory._get_mask(field, size, 0.5)[idx].item() for idx in range(size)]
        )
        for field, size in zip((i, j), data.shape, strict=True)
    )
    expected = np.count_nonzero((data != 0) * rows[:, None] * cols[None, :])
    assert stats.scan(needs_freq=False) == (expected, expected, expected, None)
    file_regression.check("\n\n".join(compiler.sources), extension=".jl")


def test_compile_julia_blocked_uniform_grid_lowering(monkeypatch, file_regression):
    _requires_julia_backend()
    from finch.autoschedule import default_schedulers
    from finch.autoschedule.formatter import DefaultLogicFormatter
    from finch.autoschedule.tensor_stats import BlockedUniformStatsFactory
    from finch.compile_jl.compiler import (
        FinchJLCompiler,
        FinchJLGenerator,
        handle_fills,
    )
    from finch.compile_jl.julia import jl
    from finch.finch_logic import Field, LogicSimplify

    class RecordingJLCompiler(FinchJLCompiler):
        def __init__(self):
            super().__init__()
            self.sources = []

        def __call__(self, prgm):
            for func in prgm.children:
                func, _ = handle_fills(func)
                source = FinchJLGenerator()(func)
                expanded = jl.seval(source.removeprefix("eval(").removesuffix(")"))
                self.sources.append(
                    f"# Finch kernel\n{source}\n\n"
                    f"# Generated Julia\n{jl.string(expanded)}"
                )
            return super().__call__(prgm)

    compiler = RecordingJLCompiler()
    scheduler = LogicNormalizer(
        LogicExecutor(
            LogicSimplify(
                DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler(compiler)))
            ),
            stats_factory=FDStatsFactory(),
        )
    )
    monkeypatch.setattr(
        default_schedulers, "NON_RECURSIVE_STANDARD_SCHEDULER", scheduler
    )
    i, j = Field("i"), Field("j")
    data = np.arange(35, dtype=DTYPE).reshape(5, 7) % 3
    stats = BlockedUniformStatsFactory(blocks_per_dim={i: 2, j: 3})(
        _csr_tensor(data), (i, j)
    )
    expected = np.array(
        [
            [
                np.count_nonzero(data[rows, cols])
                for cols in (slice(0, 2), slice(2, 4), slice(4, 7))
            ]
            for rows in (slice(0, 2), slice(2, 5))
        ]
    )
    np.testing.assert_array_equal(stats.nnz_grid, expected)
    np.testing.assert_array_equal(stats.block_sizes[i], [2, 3])
    np.testing.assert_array_equal(stats.block_sizes[j], [2, 2, 3])
    file_regression.check("\n\n".join(compiler.sources), extension=".jl")


def test_julia_element_ftype_can_customize_vector_lowering():
    _requires_julia_backend()
    from finch.compile_jl import JuliaElementFType
    from finch.compile_jl import types as jl_dtypes
    from finch.compile_jl.julia import jl

    class PairFType(ft.FType, JuliaElementFType):
        def __eq__(self, other):
            return isinstance(other, PairFType)

        def __hash__(self):
            return hash(PairFType)

        def __call__(self, val):
            left, right = val
            return int(left), int(right)

        def julia_type(self):
            return jl.Tuple[(jl.Int64, jl.Int64)]

        def julia_value(self, value, *, offset: int = 0):
            left, right = value
            return int(left) + offset, int(right) + offset

    vec = jl_dtypes.to_jl_vector(PairFType(), [(0, 3), (4, 5)], offset=1)

    assert str(jl.typeof(vec)) == "Vector{Tuple{Int64, Int64}}"
    assert [tuple(entry) for entry in vec] == [(1, 4), (5, 6)]


def _element_level(data) -> ElementLevel:
    elem_ftype = element(DTYPE(0), ftype(DTYPE), ftype(np.intp), NumpyBufferFType)
    return ElementLevel(elem_ftype, NumpyBuffer(np.asarray(data, dtype=DTYPE)))


def _csr_tensor(data: np.ndarray) -> FiberTensor:
    ptr: list[int] = [0]
    idx: list[int] = []
    vals: list[np.integer] = []
    for row in data:
        stored = np.flatnonzero(row)
        idx.extend(stored)
        vals.extend(row[stored])
        ptr.append(len(vals))

    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                _element_level(vals),
                np.intp(data.shape[1]),
                NumpyBuffer(np.asarray(ptr, dtype=np.intp)),
                NumpyBuffer(np.asarray(idx, dtype=np.intp)),
            ),
            np.intp(data.shape[0]),
        )
    )


def _dcsr_tensor(data: np.ndarray) -> FiberTensor:
    row_idx: list[int] = []
    col_ptr: list[int] = [0]
    col_idx: list[int] = []
    vals: list[np.integer] = []
    for row_num, row in enumerate(data):
        stored = np.flatnonzero(row)
        if len(stored) == 0:
            continue
        row_idx.append(row_num)
        col_idx.extend(stored)
        vals.extend(row[stored])
        col_ptr.append(len(vals))

    return FiberTensor(
        SparseListLevel(
            SparseListLevel(
                _element_level(vals),
                np.intp(data.shape[1]),
                NumpyBuffer(np.asarray(col_ptr, dtype=np.intp)),
                NumpyBuffer(np.asarray(col_idx, dtype=np.intp)),
            ),
            np.intp(data.shape[0]),
            NumpyBuffer(np.asarray([0, len(row_idx)], dtype=np.intp)),
            NumpyBuffer(np.asarray(row_idx, dtype=np.intp)),
        )
    )


def _formatted_tensor(data: np.ndarray, name: str) -> FiberTensor:
    match name:
        case "csr":
            return _csr_tensor(data)
        case "dcsr":
            return _dcsr_tensor(data)
        case _:
            raise ValueError(f"Unknown sparse test format: {name}")


class RecordingFDFormatter(FDFormatter):
    def __init__(self, loader):
        super().__init__(loader)
        self.output_ftypes = []

    def get_tensor_ftype(self, fill_value, shape_type, stats):
        tensor_ftype = super().get_tensor_ftype(fill_value, shape_type, stats)
        self.output_ftypes.append(tensor_ftype)
        return tensor_ftype


def _compute_sparse_axis_sum(level):
    from finch.autoschedule import COMPILE_JULIA

    arg = FiberTensor(DenseLevel(level, ROWS))
    expr = ft.sum(ft.defer(arg), axis=1)

    with with_default_scheduler(COMPILE_JULIA):
        return ft.compute(expr)


def _compile_julia_fd(formatter):
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(DefaultLoopOrderer(formatter)),
            stats_factory=FDStatsFactory(),
        )
    )


def _to_csr(fbr: FiberTensor) -> FiberTensor:
    """Reformat any 2D FiberTensor into CSR (Dense-over-SparseList) via Finch.jl's
    own reformat, regardless of its current level structure (e.g. SparseHash)."""
    jl_obj = tensor_to_jl(fbr)
    csr_level = jl.Dense(jl.SparseList(jl.Element(fbr.fill_value)))
    return jl_tensor_to_python(jl.Tensor(csr_level, jl_obj))


@pytest.fixture
def sparse_diagonal_data():
    return np.array(
        [
            [1, 2, 0, 3, 0],
            [4, 5, 6, 0, 0],
            [7, 0, 0, 8, 0],
            [0, 9, 0, 10, 11],
            [0, 0, 0, 0, 0],
            [12, 0, 13, 0, 14],
        ],
        dtype=DTYPE,
    )


@pytest.mark.parametrize("sparse_format", ["csr", "dcsr"])
@pytest.mark.parametrize("k", [-2, 0, 1])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_compile_julia_sparse_diagonal(sparse_diagonal_data, sparse_format, k, dtype):
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA

    data = sparse_diagonal_data
    arg = _formatted_tensor(data, sparse_format)
    mask = EyeTensor(data.shape, k=k, dtype=dtype)
    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(ft.defer(arg) * ft.defer(mask))

    expected = data * np.eye(*data.shape, k=k, dtype=dtype)
    np.testing.assert_array_equal(_to_csr(result).to_scipy().toarray(), expected)


def test_compile_julia_sparse_diagonal_lowering(sparse_diagonal_data, file_regression):
    _requires_julia_backend()
    from finch.compile_jl.compiler import (
        FinchJLCompiler,
        FinchJLGenerator,
        handle_fills,
    )
    from finch.compile_jl.julia import jl

    class RecordingJLCompiler(FinchJLCompiler):
        def __init__(self):
            super().__init__()
            self.sources = []

        def __call__(self, prgm):
            for func in prgm.children:
                func, _ = handle_fills(func)
                source = FinchJLGenerator()(func)
                # @finch_kernel returns the expanded function expression;
                # omit the outer eval to inspect Finch's sparse loops.
                expanded = jl.seval(source.removeprefix("eval(").removesuffix(")"))
                self.sources.append(str(jl.string(expanded)))
            return super().__call__(prgm)

    compiler = RecordingJLCompiler()
    scheduler = _compile_julia_fd(
        FDFormatter(CompilerFormLowerer(LogicCompiler(compiler)))
    )
    data = sparse_diagonal_data
    mask = EyeTensor(data.shape, dtype=DTYPE)
    with with_default_scheduler(scheduler):
        result = ft.compute(ft.defer(_csr_tensor(data)) * ft.defer(mask))

    expected = data * np.eye(*data.shape, dtype=DTYPE)
    np.testing.assert_array_equal(_to_csr(result).to_scipy().toarray(), expected)
    lowered = "\n\n".join(compiler.sources)
    file_regression.check(lowered, extension=".jl")


def test_compile_julia_sums_sparse_list_level():
    _requires_julia_backend()
    level = SparseListLevel(
        _element_level(STORED_VALUES),
        COLS,
        ROW_PTR,
        COL_IDX,
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64, np.int64])
@pytest.mark.parametrize("scheduler", ["COMPILE_JULIA", "COMPILE_JULIA_GALLEY"])
def test_compile_julia_sparse_fill_value_dtypes(dtype, scheduler):
    """Kernel prototypes must carry the fill type of non-default-width dtypes."""
    _requires_julia_backend()
    from finch import autoschedule

    elem_ftype = element(dtype(0), ftype(dtype), ftype(np.intp), NumpyBufferFType)
    level = SparseListLevel(
        ElementLevel(elem_ftype, NumpyBuffer(STORED_VALUES.astype(dtype))),
        COLS,
        ROW_PTR,
        COL_IDX,
    )
    arg = ft.defer(FiberTensor(DenseLevel(level, ROWS)))

    with with_default_scheduler(getattr(autoschedule, scheduler)):
        result = ft.compute(ft.sum(arg + arg, axis=1))

    np.testing.assert_array_equal(result.to_numpy(), 2 * EXPECTED_ROW_SUMS)


def test_compile_julia_sums_sparse_coo_level():
    _requires_julia_backend()
    level = SparseCOOLevel(
        _element_level(STORED_VALUES),
        (COLS,),
        ROW_PTR,
        (COL_IDX,),
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_sums_sparse_bytemap_level():
    _requires_julia_backend()
    stored_positions = np.array([0, 2, 4, 6, 8], dtype=np.intp)
    table = np.zeros(9, dtype=np.bool_)
    table[stored_positions] = True
    data = np.array([1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=DTYPE)
    level = SparseByteMapLevel(
        _element_level(data),
        COLS,
        ROW_PTR,
        NumpyBuffer(table),
        NumpyBuffer(stored_positions),
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_sums_sparse_bytemap_level_with_narrow_dimension_type():
    _requires_julia_backend()
    stored_positions = np.array([0, 2, 4, 6, 8], dtype=np.intp)
    table = np.zeros(9, dtype=np.bool_)
    table[stored_positions] = True
    data = np.array([1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=DTYPE)
    level = SparseByteMapLevel(
        _element_level(data),
        np.int32(COLS),
        ROW_PTR,
        NumpyBuffer(table),
        NumpyBuffer(stored_positions),
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_with_fd_formatter_uses_dense_output_levels():
    _requires_julia_backend()
    from finch.compile_jl.compiler import FinchJLCompiler

    formatter = RecordingFDFormatter(
        CompilerFormLowerer(LogicCompiler(FinchJLCompiler()))
    )
    scheduler = _compile_julia_fd(formatter)
    data = np.array([[1, 0, 2], [0, 3, 4]], dtype=DTYPE)
    arg = ft.asarray(data)
    expr = ft.defer(arg) + ft.defer(arg)

    with with_default_scheduler(scheduler):
        result = ft.compute(expr)

    np.testing.assert_array_equal(result.to_numpy(), data + data)
    assert formatter.output_ftypes
    output_ftype = formatter.output_ftypes[-1]
    assert isinstance(output_ftype, ft.FiberTensorFType)
    assert isinstance(output_ftype.lvl_t, ft.DenseLevelFType)
    assert isinstance(output_ftype.lvl_t.lvl_t, ft.DenseLevelFType)
    assert isinstance(output_ftype.lvl_t.lvl_t.lvl_t, ft.ElementLevelFType)


@pytest.mark.parametrize(
    ("left_format", "right_format", "op_name"),
    [
        ("csr", "csr", "add"),
        ("csr", "dcsr", "multiply"),
        ("dcsr", "dcsr", "matmul"),
        ("csr", "csr", "matmul"),
    ],
)
def test_compile_julia_fd_formatter_sparse_end_to_end(
    left_format,
    right_format,
    op_name,
):
    _requires_julia_backend()
    from finch.compile_jl.compiler import FinchJLCompiler

    sparse_a = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=DTYPE)
    sparse_b = np.array([[0, 6, 0, 7], [8, 0, 0, 0], [0, 9, 10, 0]], dtype=DTYPE)

    match op_name:
        case "add":
            expected = sparse_a + sparse_a
        case "multiply":
            expected = sparse_a * sparse_a
        case "matmul":
            expected = sparse_a @ sparse_b
        case _:
            raise ValueError(f"Unknown sparse end-to-end op: {op_name}")

    formatter = FDFormatter(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())))
    scheduler = _compile_julia_fd(formatter)
    left_data = sparse_a
    right_data = sparse_b if op_name == "matmul" else sparse_a
    left = _formatted_tensor(left_data, left_format)
    right = _formatted_tensor(right_data, right_format)

    match op_name:
        case "add":
            expr = ft.defer(left) + ft.defer(right)
        case "multiply":
            expr = ft.defer(left) * ft.defer(right)
        case "matmul":
            expr = ft.matmul(ft.defer(left), ft.defer(right))
        case _:
            raise ValueError(f"Unknown sparse end-to-end op: {op_name}")

    with with_default_scheduler(scheduler):
        result = ft.compute(expr)

    csr_result = _to_csr(result)
    np.testing.assert_allclose(csr_result.to_scipy().toarray(), expected)


@pytest.mark.parametrize(
    "op, args, expected_code, expected",
    [
        (ffuncs.add, (1, 2, 3), "(1 + 2 + 3)", 6),
        (ffuncs.and_, (7, 3, 1), "(7 & 3 & 1)", 1),
        (ffuncs.or_, (4, 2, 1), "(4 | 2 | 1)", 7),
        (
            ffuncs.logical_and,
            (True, True, False),
            "Finch.and(true,true,false)",
            False,
        ),
        (
            ffuncs.logical_or,
            (False, False, True),
            "Finch.or(false,false,true)",
            True,
        ),
    ],
)
def test_compile_julia_evaluates_variadic_and_or_call(
    op, args, expected_code, expected
):
    _requires_julia_backend()
    from finch.compile_jl.compiler import FinchJLGenerator

    call = ntn.Call(
        ntn.Literal(op),
        tuple(ntn.Literal(arg) for arg in args),
    )
    generated = FinchJLGenerator().generate_julia(call)
    assert generated == expected_code

    from finch.compile_jl.julia import jl

    assert jl.seval(generated) == expected


def test_compile_julia_galley_dense_matmul():
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA_GALLEY

    rng = np.random.default_rng(0)
    a = rng.random((8, 5))
    b = rng.random((5, 3))

    result = ft.compute(
        ft.defer(ft.asarray(a)) @ ft.defer(ft.asarray(b)),
        ctx=COMPILE_JULIA_GALLEY,
    )

    # Galley picks the output levels, so densify through CSR rather than
    # assuming any particular level structure.
    np.testing.assert_allclose(_to_csr(result).to_scipy().toarray(), a @ b)


def test_compile_julia_galley_sparse_matmul():
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA_GALLEY

    rng = np.random.default_rng(0)
    a = np.where(rng.random((50, 40)) < 0.05, rng.random((50, 40)), 0.0)
    b = np.where(rng.random((40, 30)) < 0.05, rng.random((40, 30)), 0.0)

    result = ft.compute(
        ft.defer(ft.asarray(a)) @ ft.defer(ft.asarray(b)),
        ctx=COMPILE_JULIA_GALLEY,
    )

    np.testing.assert_allclose(_to_csr(result).to_scipy().toarray(), a @ b)


def test_compile_julia_galley_chained_matmul():
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA_GALLEY

    rng = np.random.default_rng(0)
    a = rng.random((6, 4))
    b = np.where(rng.random((4, 7)) < 0.25, rng.random((4, 7)), 0.0)
    c = rng.random((7, 2))

    result = ft.compute(
        ft.defer(ft.asarray(a)) @ ft.defer(ft.asarray(b)) @ ft.defer(ft.asarray(c)),
        ctx=COMPILE_JULIA_GALLEY,
    )

    np.testing.assert_allclose(_to_csr(result).to_scipy().toarray(), a @ b @ c)


@pytest.mark.parametrize("scheduler", ["COMPILE_JULIA", "COMPILE_JULIA_GALLEY"])
def test_compile_julia_full_reduction_returns_a_scalar(scheduler):
    _requires_julia_backend()
    from finch import autoschedule

    data = np.arange(12, dtype=np.float64).reshape(4, 3)

    result = ft.compute(
        ft.sum(ft.defer(ft.asarray(data))),
        ctx=getattr(autoschedule, scheduler),
    )

    assert result.shape == ()
    np.testing.assert_allclose(np.array(result), data.sum())


def _float_csr(data: np.ndarray, fill: float) -> FiberTensor:
    ptr, idx, vals = [0], [], []
    for row in data:
        stored = np.flatnonzero(row != fill)
        idx.extend(stored)
        vals.extend(row[stored])
        ptr.append(len(vals))
    elem_t = element(fill, ftype(np.float64), ftype(np.intp), NumpyBufferFType)
    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(elem_t, NumpyBuffer(np.array(vals, dtype=np.float64))),
                np.intp(data.shape[1]),
                NumpyBuffer(np.array(ptr, dtype=np.intp)),
                NumpyBuffer(np.array(idx, dtype=np.intp)),
            ),
            np.intp(data.shape[0]),
        )
    )


def _julia_binary(op, a, b, out_fill, compiler):
    from finch import dense, fiber_tensor, sparse_list
    from finch import finch_logic as lgc

    i, j = lgc.Field("i"), lgc.Field("j")
    A, B, Y = lgc.Alias("A"), lgc.Alias("B"), lgc.Alias("Y")
    body = lgc.MapJoin(lgc.Literal(op), (lgc.Table(A, (i, j)), lgc.Table(B, (i, j))))
    rhs = lgc.Aggregate(
        lgc.Literal(ffuncs.overwrite),
        lgc.Literal(out_fill),
        lgc.Reorder(body, (i, j)),
        (),
    )
    prgm = lgc.Plan((lgc.Query(lgc.Table(Y, (i, j)), rhs), lgc.Produces((Y,))))
    out_t = fiber_tensor(
        dense(
            sparse_list(
                element(out_fill, ftype(np.float64), ftype(np.intp)), ftype(np.intp)
            ),
            ftype(np.intp),
        )
    )
    bindings = {A: a, B: b, Y: out_t.construct(a.shape)}
    lib, *_ = CompilerFormLowerer(LogicCompiler(compiler))(
        prgm, {k: v.ftype for k, v in bindings.items()}, {}, None
    )
    (res,) = lib.main(*bindings.values())
    return np.transpose(jl.Array(res.raw_julia_obj).to_numpy())


def test_julia_floor_divide_floors_like_numpy():
    _requires_julia_backend()
    a = np.array([[-7.0, 7.0], [1.0, -1.0]])
    b = np.array([[2.0, 2.0], [0.1, 0.1]])
    got = _julia_binary(
        ffuncs.floordiv,
        _float_csr(a, 0.0),
        _float_csr(b, 0.0),
        np.float64(np.nan),
        FinchJLCompiler(),
    )
    np.testing.assert_array_equal(got, np.floor_divide(a, b))


def test_julia_coo_round_trip_keeps_python_dim_order():
    _requires_julia_backend()
    dense = np.zeros((2, 3, 4), dtype=DTYPE)
    coords = np.array([[0, 0, 1], [0, 2, 3], [1, 1, 0], [1, 2, 2]], dtype=np.intp)
    dense[tuple(coords.T)] = [1, 2, 3, 4]
    level = SparseCOOLevel(
        _element_level([1, 2, 3, 4]),
        (np.intp(2), np.intp(3), np.intp(4)),
        NumpyBuffer(np.array([0, 4], dtype=np.intp)),
        tuple(NumpyBuffer(np.ascontiguousarray(c)) for c in coords.T),
    )
    tensor = FiberTensor(level)

    jl_obj = tensor_to_jl(tensor)
    np.testing.assert_array_equal(np.transpose(jl.Array(jl_obj).to_numpy()), dense)

    back = jl_tensor_to_python(jl_obj).lvl
    assert tuple(int(s) for s in back.coo_shape) == (2, 3, 4)
    for got, want in zip(back.tbl, coords.T, strict=True):
        np.testing.assert_array_equal(got.arr, want)


def test_julia_kernel_writes_coo_output_in_python_dim_order():
    _requires_julia_backend()
    from finch import fiber_tensor, sparse_coo
    from finch import finch_logic as lgc

    data = np.array([[0.0, 1.5, 0.0], [2.5, 0.0, 3.5]])
    i, j = lgc.Field("i"), lgc.Field("j")
    A, Y = lgc.Alias("A"), lgc.Alias("Y")
    rhs = lgc.Aggregate(
        lgc.Literal(ffuncs.overwrite),
        lgc.Literal(0.0),
        lgc.Reorder(lgc.Table(A, (i, j)), (i, j)),
        (),
    )
    prgm = lgc.Plan((lgc.Query(lgc.Table(Y, (i, j)), rhs), lgc.Produces((Y,))))
    out_t = fiber_tensor(
        sparse_coo(element(0.0, ftype(np.float64), ftype(np.intp)), 2, ftype(np.intp))
    )
    bindings = {A: _float_csr(data, 0.0), Y: out_t.construct(data.shape)}
    lib, *_ = CompilerFormLowerer(LogicCompiler(FinchJLCompiler()))(
        prgm, {k: v.ftype for k, v in bindings.items()}, {}, None
    )
    (res,) = lib.main(*bindings.values())

    np.testing.assert_array_equal(
        np.transpose(jl.Array(res.raw_julia_obj).to_numpy()), data
    )
    lvl = jl_tensor_to_python(res.raw_julia_obj).lvl
    assert tuple(int(s) for s in lvl.coo_shape) == data.shape
    rows, cols = (np.asarray(c.arr)[: int(lvl.ptr.arr[-1])] for c in lvl.tbl)
    np.testing.assert_array_equal(rows, [0, 1, 1])
    np.testing.assert_array_equal(cols, [1, 0, 2])


def test_compute_shape_vars_accepts_broadcast_reorder():
    from finch import finch_logic as lgc
    from finch.finch_logic.stages import compute_shape_vars

    a, out = lgc.Alias("a"), lgc.Alias("out")
    i, j = lgc.Field("i"), lgc.Field("j")
    prgm = lgc.Plan(
        (
            lgc.Query(lgc.Table(out, (i, j)), lgc.Reorder(lgc.Table(a, (i,)), (i, j))),
            lgc.Produces((out,)),
        )
    )
    tensor = ft.asarray(np.arange(3, dtype=np.int64))
    shape_vars = compute_shape_vars(prgm, {a: tensor.ftype})
    assert len(shape_vars[a]) == 1


def test_minus_one_buffer_arr_shifts_every_tuple_field():
    from finch.compile_jl.buffer import MinusOneBuffer

    raw = np.zeros(3, dtype=[("element_1", np.intp), ("element_2", np.intp)])
    raw["element_1"] = [1, 2, 3]
    raw["element_2"] = [4, 5, 6]
    arr = MinusOneBuffer(NumpyBuffer(raw)).arr

    np.testing.assert_array_equal(arr["element_1"], [0, 1, 2])
    np.testing.assert_array_equal(arr["element_2"], [3, 4, 5])
    np.testing.assert_array_equal(raw["element_1"], [1, 2, 3])


def test_register_julia_op_maps_and_defines_once():
    _requires_julia_backend()
    from finch.compile_jl import compiler

    class _Op:
        ftype = object()

    op = _Op()
    source = "finch_test_twice(x) = 2x"
    compiler.register_julia_op(op, "finch_test_twice", source)
    compiler.register_julia_op(op, "finch_test_twice", source)

    assert compiler._JULIA_OPS[op.ftype] == "finch_test_twice"
    assert int(jl.finch_test_twice(3)) == 6
    del compiler._JULIA_OPS[op.ftype]


def test_compile_julia_negative_zero_fill():
    # Python ftypes equate -0.0 and 0.0 fills, so they share Julia kernels.
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA

    arg = np.arange(4.0).reshape(2, 2)
    square = arg @ arg
    with with_default_scheduler(COMPILE_JULIA):
        positive = ft.einop("P[i,j] += A[i,k] * A[k,j]", A=ft.asarray(arg))
        negative = ft.negative(positive)
        assert np.signbit(negative.fill_value)
        results = [
            ft.subtract(positive, positive),
            ft.subtract(negative, negative),
            ft.subtract(negative, positive),
            ft.einop("T[i,j] = F[j,i]", F=negative),
            ft.einop("T[i,j] = F[j,i]", F=positive),
        ]
        expected = [0 * square, 0 * square, -2 * square, -square.T, square.T]
        for result, value in zip(results, expected, strict=True):
            np.testing.assert_array_equal(result.to_numpy(), value)


def test_compile_julia_dynamic_fill_value_types():
    # Dynamic fills are equal by dtype, so fills whose values have different
    # types share a kernel, and must pin to zeros of that dtype.
    _requires_julia_backend()
    import scipy.sparse as sps

    from finch.autoschedule import COMPILE_JULIA

    matrix = sps.random(3, 3, density=0.5, format="csr", random_state=0)
    tensor = ft.asarray(matrix)
    with with_default_scheduler(COMPILE_JULIA):
        for value in (np.float32(0.0), np.float64(0.0)):
            arg = tensor.with_fill(DynamicFill(value, ftype(np.float64)))
            result = ft.compute(ft.defer(arg) * 2.0)
            np.testing.assert_array_equal(result.to_numpy(), matrix.toarray() * 2)


@pytest.mark.parametrize(
    "op, args",
    [
        (ft.logical_not, (np.array([0, 1, 2, -3]),)),
        (ft.logical_not, (np.array([0.0, 0.5, np.nan, -1.0]),)),
        (ft.logical_and, (np.array([0, 1, 2, -3]), np.array([5, 0, 0, 1]))),
        (ft.logical_or, (np.array([0, 1, 0, -3]), np.array([0, 0, 2, 1]))),
        (ft.logical_xor, (np.array([0, 1, 2, -3]), np.array([5, 0, 0, 1]))),
    ],
    ids=["not_int", "not_float", "and_int", "or_int", "xor_int"],
)
def test_compile_julia_logical_ops_on_numbers(op, args):
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA

    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(op(*(ft.defer(arg) for arg in args)))

    np_op = getattr(np, op.__name__)
    np.testing.assert_array_equal(result.to_numpy(), np_op(*args))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_compile_julia_complex_arithmetic(dtype):
    _requires_julia_backend()
    from finch.autoschedule import COMPILE_JULIA

    arg = np.array([[1 + 2j, 0], [3j, -1]], dtype=dtype)
    with with_default_scheduler(COMPILE_JULIA):
        result = ft.compute(ft.defer(arg) * ft.defer(arg) + dtype(1 - 1j))
        total = ft.compute(ft.sum(ft.defer(arg), axis=0))

    np.testing.assert_allclose(result.to_numpy(), arg * arg + dtype(1 - 1j))
    np.testing.assert_allclose(total.to_numpy(), arg.sum(axis=0))
