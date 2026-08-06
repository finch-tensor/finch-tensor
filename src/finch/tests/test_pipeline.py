import importlib.util
import logging
import time
from pathlib import Path

import pytest

import numpy as np
import scipy.io
import scipy.sparse

import finch as ft
from finch import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    element,
    ffuncs,
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
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.tensor_stats import FDStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.julia import jl
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.util.logging import FORMAT, get_logger_handler

from .conftest import finch_assert_allclose
from .test_julia_backend import RecordingFDFormatter

jl.seval("Base.cumulative_compile_timing(true)")
BOEING_PATH = Path("tests/data/ct20stif.mtx")
BLOCK_SIZE = 300


def requires_boeing_matrix():
    if not BOEING_PATH.exists():
        raise ValueError("Path to boeing needs to be provided")


def requires_julia_backend():
    if importlib.util.find_spec("juliapkg") is None:
        pytest.skip("juliapkg is not installed")
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("juliacall is not installed")


def csr_tensor_from_scipy(M_csr, dtype=np.float64) -> FiberTensor:
    elem_ftype = element(dtype(0), ftype(dtype), ftype(np.intp), NumpyBufferFType)
    elem_level = ElementLevel(
        elem_ftype, NumpyBuffer(np.asarray(M_csr.data, dtype=dtype))
    )
    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                elem_level,
                np.intp(M_csr.shape[1]),
                NumpyBuffer(np.asarray(M_csr.indptr, dtype=np.intp)),
                NumpyBuffer(np.asarray(M_csr.indices, dtype=np.intp)),
            ),
            np.intp(M_csr.shape[0]),
        )
    )


def describe_format(tensor_ftype):
    lvl = tensor_ftype.lvl_t
    axes = []
    while type(lvl).__name__ != "ElementLevelFType":
        axes.append("dense" if "Dense" in type(lvl).__name__ else "Sparse")
        lvl = lvl.lvl_t
    return axes


def fd_scheduler(formatter):
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicOptimizer(DefaultLoopOrderer(formatter)),
            stats_factory=FDStatsFactory(),
            cache=True,
        )
    )


@pytest.fixture
def boeing_slice():
    requires_boeing_matrix()
    M_full = scipy.io.mmread(BOEING_PATH).tocsr()
    return M_full[:BLOCK_SIZE, :BLOCK_SIZE].tocsr()


def test_boeing_matrix_load(boeing_slice):
    assert boeing_slice.shape == (BLOCK_SIZE, BLOCK_SIZE)
    assert boeing_slice.nnz > 0


def test_built_levels(boeing_slice):
    tensor = csr_tensor_from_scipy(boeing_slice)
    assert describe_format(tensor.ftype) == ["dense", "Sparse"]

    # finch_assert_allclose(tensor.to_numpy(),boeing_slice.toarray()) -
    # no to_numpy() in Sparselevel

    outer = tensor.lvl
    inner = outer.lvl
    np.testing.assert_array_equal(inner.ptr.arr, boeing_slice.indptr)
    np.testing.assert_array_equal(inner.idx.arr, boeing_slice.indices)
    np.testing.assert_allclose(inner.lvl.val.arr, boeing_slice.data)
    assert outer.dimension == boeing_slice.shape[0]


def test_matmul_on_boeing_slice(boeing_slice, numba_compiler):
    tensor = csr_tensor_from_scipy(boeing_slice)
    result = ft.compute(ft.matmul(ft.lazy(tensor), ft.lazy(tensor)))
    finch_assert_allclose(
        result.to_numpy(), (boeing_slice @ boeing_slice).toarray(), rtol=1e-5, atol=1e-3
    )


def test_fd_formatter_decides_Sparse_output(boeing_slice):
    tensor = csr_tensor_from_scipy(boeing_slice)
    i, k, j = Field("i"), Field("k"), Field("j")
    A, B, C = Alias("A"), Alias("B"), Alias("C")
    stats_factory = FDStatsFactory()
    stats = {A: stats_factory(tensor, (i, k)), B: stats_factory(tensor, (k, j))}
    plan = Plan(
        (
            Query(
                C,
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0.0),
                    MapJoin(Literal(ffuncs.mul), (Table(A, (i, k)), Table(B, (k, j)))),
                    (k,),
                ),
            ),
            Produces((C,)),
        )
    )
    capture = LogicCapture()
    FDFormatter(capture).lower(
        plan, {A: tensor.ftype, B: tensor.ftype}, stats, stats_factory
    )
    assert describe_format(capture.last_bindings[C]) == ["Sparse", "Sparse"]


"""
def test_full_pipeline_through_julia_backend(boeing_slice):
    requires_julia_backend()

    tensor = csr_tensor_from_scipy(boeing_slice)
    formatter = RecordingFDFormatter(LogicCompiler(FinchJLCompiler()))
    scheduler = fd_scheduler(formatter)
    ft.set_default_scheduler(ctx=scheduler)

    with with_default_scheduler(scheduler):
        result = ft.compute(ft.matmul(ft.lazy(tensor), ft.lazy(tensor)))

    finch_assert_allclose(
        result.to_numpy(), (boeing_slice @ boeing_slice).toarray(), rtol=1e-5, atol=1e-3
    )
    assert describe_format(formatter.output_ftypes[-1]) == ["Sparse", "Sparse"]
"""


def test_run_full_boeing():
    handler = get_logger_handler(filter_pattern="r.n,r.a")
    logging.basicConfig(
        level=logging.DEBUG, handlers=[handler], format=FORMAT, force=True
    )
    t0 = time.time()
    M = scipy.io.mmread(BOEING_PATH).tocsr()
    time_load = time.time() - t0
    print(f"[{time_load:6.2f}s] for loading matrix:\n shape={M.shape}, nnz={M.nnz}")

    t0 = time.time()
    _scipy_result = M @ M
    time_scipy = time.time() - t0
    print(f"{time_scipy:6.2f}s for scipy boeing matmul")

    tensor = csr_tensor_from_scipy(M)
    t0 = time.time()
    formatter = RecordingFDFormatter(LogicCompiler(FinchJLCompiler()))

    scheduler = fd_scheduler(formatter)
    ft.set_default_scheduler(ctx=scheduler)

    with with_default_scheduler(scheduler):
        _result = ft.compute(ft.matmul(ft.lazy(tensor), ft.lazy(tensor)))
    time_finchjl_exec_compile = time.time() - t0
    print(f"{time_finchjl_exec_compile:6.2f}s for boeing julia matmul")
    t0 = time.time()
    with with_default_scheduler(scheduler):
        _result2 = ft.compute(ft.matmul(ft.lazy(tensor), ft.lazy(tensor)))
    time_finchjl_exec = time.time() - t0
    print(f"second call with julia_finch : {time.time() - t0:6.2f}s")
    print(
        "output format decided by FDFormatter:",
        describe_format(formatter.output_ftypes[-1]),
    )
    print(f"Scipy is {time_finchjl_exec / time_scipy}x faster than finch-jl")


def make_matrix(M):
    matrix_M = scipy.sparse.lil_matrix(M.shape, dtype=M.dtype)
    matrix_M[0, 0] = 1.0
    return matrix_M.tocsr()


def run_split(expr):
    t_start = time.time()
    c0 = jl.seval("Base.cumulative_compile_time_ns()")
    result = ft.compute(expr)
    c1 = jl.seval("Base.cumulative_compile_time_ns()")
    t_end = time.time()

    total = t_end - t_start
    julia_compile = (c1[0] - c0[0]) / 1e9

    print(f"Total time = {total}s \n Julia compile time = {julia_compile}")
    return result


def test_separate_compile_time():
    M = scipy.io.mmread(BOEING_PATH).tocsr()
    test_M = make_matrix(M)
    test_tensor_M = csr_tensor_from_scipy(test_M)
    formatter = RecordingFDFormatter(LogicCompiler(FinchJLCompiler()))
    scheduler = fd_scheduler(formatter)
    ft.set_default_scheduler(ctx=scheduler)

    with with_default_scheduler(scheduler):
        run_split(ft.matmul(ft.lazy(test_tensor_M), ft.lazy(test_tensor_M)))


"""
def test_toy_pipeline():

    A = np.array([[1,0,2],[0,3,0],[4,0,5]],dtype=np.int64)
    B = np.array([[0,6,0],[8,0,0],[0,9,0]],dtype=np.int64)

    expr = ft.matmul(ft.lazy(A),ft.lazy(B))
    result = ft.compute(expr)

    expected = A@B
    calculated = result.to_numpy()

    #print(f"expected : {expected}")
    #print(f"finchtensor output :{calculated}")
    #assert np.array_equal(calculated,expected), "Doesn't match"
    #print(f"default scheduler used : {type(ft.get_default_scheduler()).__name__}")


def test_parse_boeing():
    M = scipy.io.mmread('data/ct20stif.mtx')
    print(f"type after mmread : {type(M)}")
    print(f"shape : {M.shape}")
    print(f"nnz stored: {M.nnz}")

    #Converting to CSR
    M_csr = M.tocsr()
    print(f"type as csr : {type(M)}")
    print(f"ptr length: {len(M_csr.indptr)}")
    print(f"indices(idx) length: {len(M_csr.indices)}")
    print(f"few values : {M_csr.data[:5]}")
"""
