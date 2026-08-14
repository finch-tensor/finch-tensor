import importlib.util
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
from finch.compile_jl.compiler import FinchJLCompiler, FinchJLKernel
from finch.compile_jl.interop import jl_tensor_to_python, tensor_to_jl
from finch.compile_jl.julia import get_jl, jl
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

from .conftest import finch_assert_allclose
from .test_julia_backend import RecordingFDFormatter

jl.seval("Base.cumulative_compile_timing(true)")
BOEING_PATH = Path("data/ct20stif.mtx")
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


def test_timed_boeing():

    M = scipy.io.mmread(BOEING_PATH).tocsr()
    tensor = csr_tensor_from_scipy(M)

    timings = {
        "seval": [],
        "julia_call": [],
        "arg_conversion": [],
        "result_conversion": [],
        "gc_ns": [],
    }
    captured = {}

    def timed_init(self, func_name, jl_code):
        print(self)
        self.jl_code = jl_code
        self.func_name = func_name
        t0 = time.time()
        jl.seval(self.jl_code)
        timings["seval"].append(time.time() - t0)

    def timed_call(self, *args):
        t0 = time.time()
        raw_args = [tensor_to_jl(arg) for arg in args]
        timings["arg_conversion"].append(time.time() - t0)

        # Only julia call
        finch_fn = getattr(jl, self.func_name)
        gc0 = jl.seval("Base.gc_time_ns()")
        t0 = time.time()
        result = finch_fn(*raw_args)
        timings["julia_call"].append(time.time() - t0)  # 0 -> cold, 1 -> warm
        gc1 = jl.seval("Base.gc_time_ns()")
        timings["gc_ns"].append(gc1 - gc0)

        t0 = time.time()
        if jl.isa(result, jl.Finch.Tensor):
            out = (jl_tensor_to_python(result),)
        else:
            out = tuple(jl_tensor_to_python(res) for res in result)
        timings["result_conversion"].append(time.time() - t0)

        captured["raw_args"] = raw_args
        captured["func_name"] = self.func_name
        return out

    orig_init = FinchJLKernel.__init__
    orig_call = FinchJLKernel.__call__
    FinchJLKernel.__init__ = timed_init
    FinchJLKernel.__call__ = timed_call

    try:
        t0 = time.time()
        _scipy_result = M @ M
        time_scipy = time.time() - t0
        formatter = RecordingFDFormatter(LogicCompiler(FinchJLCompiler()))
        scheduler = fd_scheduler(formatter)
        ft.set_default_scheduler(ctx=scheduler)
        expr = ft.matmul(ft.lazy(tensor), ft.lazy(tensor))
        t0 = time.time()
        with with_default_scheduler(scheduler):
            ft.compute(expr)
        total_cold = time.time() - t0

        # jl.seval("GC.gc()")

        t0 = time.time()
        with with_default_scheduler(scheduler):
            ft.compute(expr)
        total_warm = time.time() - t0
        logic_exec = scheduler.ctx
        ((mod, *_),) = logic_exec.cached_kernels.values()
        # kernel_key = next(iter(mod.kernel_dict.keys()))

        kernel = next(iter(mod.kernel_dict.values()))
        print(f"\n{kernel.jl_code}")

    finally:
        FinchJLKernel.__init__ = orig_init
        FinchJLKernel.__call__ = orig_call

    julia_seval_time = timings["seval"][0]
    cold_arg_conv, warm_arg_conv = timings["arg_conversion"]

    cold_julia_call, warm_julia_call = timings["julia_call"]

    cold_result_conv, warm_result_conv = timings["result_conversion"]

    python_time = (
        total_cold
        - julia_seval_time
        - cold_arg_conv
        - cold_julia_call
        - cold_result_conv
    )
    julia_compile_time = cold_julia_call - warm_julia_call
    julia_run_time = warm_julia_call

    print("GC time during warm call:", timings["gc_ns"][1] / 1e9)
    print(f"scipy time for boeing matmul : {time_scipy:.3f}")
    print(f"python pipeline time : {python_time:.3f}s")
    print(f"julia seval time : {julia_seval_time}")
    print(f"julia compile time : {julia_compile_time:.3f}s")
    print(f"cold argument conversion time : {cold_arg_conv}")
    print(f"warm argument conversion time : {warm_arg_conv}")
    print(f"result conversion time :{warm_result_conv}")
    print(f"julia run time : {julia_run_time:.3f}s")
    print(f"total pipeline time (compile and run) : {total_cold:.3f}")
    print(f"total pipeline runtime : {total_warm:.3f}")

    Main = get_jl()
    for i, a in enumerate(captured["raw_args"]):
        # print(f"Type of {a} is {type(a)}")
        setattr(Main, f"bencharg{i}", a)
    # countstored = jl.seval("Finch.countstored")
    # Test to check whether writing to a fresh or filled arguments accounts
    # for the time difference
    # Baically checking if sparsehash fresh or being written to when
    # filled creates a big difference
    """"
    v1_python = jl_tensor_to_python(Main.bencharg1)
    v2_python = jl_tensor_to_python(Main.bencharg2)

    print("\n --------------------------------------------- \n")
    print(
        "stored counts in prefilled args :",
        countstored(Main.bencharg1),
        countstored(Main.bencharg2),
    )
    print("@time for prefilled args ")
    jl.seval(f"@time {captured['func_name']}(bencharg0, bencharg1, bencharg2);")

    v1_fresh = tensor_to_jl(v1_python.ftype.construct(v1_python.shape))
    v2_fresh = tensor_to_jl(v2_python.ftype.construct(v2_python.shape))
    Main.v1_fresh = v1_fresh
    Main.v2_fresh = v2_fresh
    print(
        "fresh args stored counts :",
        countstored(Main.v1_fresh),
        countstored(Main.v2_fresh),
    )
    print("fresh args @time:")
    jl.seval(f"@time {captured['func_name']}(bencharg0, v1_fresh, v2_fresh);")

    arglist = ", ".join(f"$bencharg{i}" for i in range(len(captured["raw_args"])))
    bench_kernel = jl.seval(f"@benchmark {captured['func_name']}({arglist})")
    print(jl.seval("string")(bench_kernel))
    countstored = jl.seval("Finch.countstored")
    #print(f"bencharg stored entries : {countstored(Main.bencharg1)}")
    #print(f"bencharg stored entries : {countstored(Main.bencharg2)}")

    #To check if calling the same function through python adds an overhead
    finch_fn_fresh = getattr(jl, captured["func_name"])
    t0 = time.time()
    _ = finch_fn_fresh(*captured["raw_args"])
    print("fresh python-side recall next to benchmark:", time.time() - t0)

    #To check the difference between the types of argument we provide in the
    # pipeline and manually
    #Made single_write = True -> improvement

    jl.seval('''
    using Finch
    m, n = size(bencharg0)
    v1_fresh = Tensor(Dense(SparseHash(Element(0.0))), m, n)
    v2_fresh = Tensor(SparseHash(SparseHash(Element(0.0))), m, n)
    ''')

    bench2 = jl.seval(f"@benchmark {
    captured['func_name']}($bencharg0, $v1_fresh, $v2_fresh)")
    print(jl.seval("string")(bench2))

    typeof = jl.seval("typeof")
    print("our pipeline's _v1 type:", typeof(Main.bencharg1))
    print("manual v1_fresh type:", typeof(Main.v1_fresh))

    #Checking if the call overhead is accounting for the time difference
    jl.seval("_noop(x,y,z)=(x,y,z)")
    noop_fn = jl.seval("_noop")
    t0 = time.time()
    _ = noop_fn(*captured["raw_args"])
    print(f"call overhead from python-side call : {time.time()-t0}")

    bench_noop = jl.seval("@benchmark _noop($bencharg0, $bencharg1, $bencharg2)")
    print(jl.seval("string")(bench_noop))

    #Tiny example to check julia roundtrip time
    A = np.arange(9,dtype=np.float64).reshape(3,3)
    jl_func = '''
    function identity_kernel(x)
        return x
    end

    '''

    kernel = FinchJLKernel("identity_kernel",jl_func)
    t0 = time.perf_counter()
    (out,) = kernel(A)
    t1 = time.perf_counter()
    print(f"time for cold call : {t1-t0}")

    t2 = time.perf_counter()
    (out,) = kernel(A)
    t3 = time.perf_counter()
    print(f"time for warm call : {t3-t2}")
    """
