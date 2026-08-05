import importlib.util

import pytest

import numpy as np

import finch as ft
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
    ftype,
)
from finch.autoschedule import (
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    FDFormatter,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    with_default_scheduler,
)
from finch.autoschedule.tensor_stats import FDStatsFactory

DTYPE = np.int64
ROWS = np.intp(3)
COLS = np.intp(3)
ROW_PTR = NumpyBuffer(np.array([0, 2, 3, 5], dtype=np.intp))
COL_IDX = NumpyBuffer(np.array([0, 2, 1, 0, 2], dtype=np.intp))
STORED_VALUES = np.array([1, 2, 3, 4, 5], dtype=DTYPE)
EXPECTED_ROW_SUMS = np.array([3, 3, 9], dtype=DTYPE)


def _requires_julia_backend():
    if importlib.util.find_spec("juliapkg") is None:
        pytest.skip("juliapkg is not installed")
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("juliacall is not installed")


def test_julia_element_ftype_can_customize_vector_lowering():
    _requires_julia_backend()
    from finch.compile_jl import JuliaElementFType
    from finch.compile_jl import dtypes as jl_dtypes
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
    from finch.compile_jl import COMPILE_JULIA

    arg = FiberTensor(DenseLevel(level, ROWS))
    expr = ft.sum(ft.lazy(arg), axis=1)

    with with_default_scheduler(COMPILE_JULIA):
        return ft.compute(expr)


def _compile_julia_fd(formatter):
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicOptimizer(DefaultLoopOrderer(formatter)),
            stats_factory=FDStatsFactory(),
        )
    )


def _to_csr(fbr: FiberTensor) -> FiberTensor:
    """Reformat any 2D FiberTensor into CSR (Dense-over-SparseList) via Finch.jl's
    own reformat, regardless of its current level structure (e.g. SparseHash)."""
    from finch.compile_jl.interop import jl_tensor_to_python, tensor_to_jl
    from finch.compile_jl.julia import jl

    jl_obj = tensor_to_jl(fbr)
    csr_level = jl.Dense(jl.SparseList(jl.Element(fbr.fill_value)))
    return jl_tensor_to_python(jl.Tensor(csr_level, jl_obj))


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


def test_compile_julia_with_fd_formatter_uses_dense_output_levels():
    _requires_julia_backend()
    from finch.compile_jl.compiler import FinchJLCompiler

    formatter = RecordingFDFormatter(LogicCompiler(FinchJLCompiler()))
    scheduler = _compile_julia_fd(formatter)
    data = np.array([[1, 0, 2], [0, 3, 4]], dtype=DTYPE)
    arg = ft.asarray(data)
    expr = ft.lazy(arg) + ft.lazy(arg)

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

    formatter = FDFormatter(LogicCompiler(FinchJLCompiler()))
    scheduler = _compile_julia_fd(formatter)
    left_data = sparse_a
    right_data = sparse_b if op_name == "matmul" else sparse_a
    left = _formatted_tensor(left_data, left_format)
    right = _formatted_tensor(right_data, right_format)

    match op_name:
        case "add":
            expr = ft.lazy(left) + ft.lazy(right)
        case "multiply":
            expr = ft.lazy(left) * ft.lazy(right)
        case "matmul":
            expr = ft.matmul(ft.lazy(left), ft.lazy(right))
        case _:
            raise ValueError(f"Unknown sparse end-to-end op: {op_name}")

    with with_default_scheduler(scheduler):
        result = ft.compute(expr)

    csr_result = _to_csr(result)
    np.testing.assert_allclose(csr_result.to_scipy().toarray(), expected)
