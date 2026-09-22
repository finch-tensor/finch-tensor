"""
Galley ASV benchmark: report Galley ``optimize_time`` and ``downstream_time`` for a
10-matrix matmul chain lowered through the julia backend. Inputs are CSR; one case
gives the last matrix values while the other leaves it empty.

Run: ``pixi run --environment=benchmark-julia pytest --codspeed
benchmarks/test_galley.py``
"""

from functools import reduce
from typing import Literal

import pytest

import numpy as np
import scipy.sparse as sps

import finch.interface as fl_interface
from finch.autoschedule import (
    LogicExecutor,
    LogicNormalizer,
)
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.formatter import GalleyFormatter
from finch.autoschedule.loop_orderer import BFSLoopOrderer
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.julia import julia_available
from finch.compile_jl.runtime import DefaultFinchJLRuntime
from finch.finch_logic import (
    Alias,
    Field,
    LogicSimplify,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.symbolic import gensym
from finch.tensor.fiber_tensor import FiberTensor

from .utils import patch_benchmark

pytestmark = pytest.mark.skipif(
    not julia_available(),
    reason="the julia extra (juliapkg, juliacall) is not installed",
)

CHAIN_LEN = 10
MAT_DIM = 8

CASES = {
    "matmul10_dense": False,
    "matmul10_empty_last": True,
}


def _plan_from_lazy(expr):
    """
    Build the same `Plan` as `finch.interface.fuse.compute`.
    """
    args = (expr,)
    vars_ = tuple(Alias(gensym("A")) for _ in args)
    ctx = args[0].ctx.join()
    bodies = tuple(
        Query(
            var,
            Table(a.data, tuple(Field(gensym("i")) for _ in range(len(a.shape)))),
        )
        for a, var in zip(args, vars_, strict=True)
    )
    return Plan(ctx.trace() + bodies + (Produces(vars_),))


def _csr(matrix: np.ndarray) -> FiberTensor:
    """
    Store `matrix` as CSR, the dense-over-sparse-list layout galley formats for.
    """
    return FiberTensor.from_scipy_csr(sps.csr_matrix(matrix))


def _build_expr(empty_last):
    """
    Build a chain of 10 matmuls with the last matrix either filled or empty
    """
    rng = np.random.default_rng(42)
    mats = [rng.standard_normal((MAT_DIM, MAT_DIM)) for _ in range(CHAIN_LEN)]
    if empty_last:
        mats[-1].fill(0)
    lazies = [fl_interface.defer(_csr(m)) for m in mats]
    return reduce(lambda a, b: a @ b, lazies)


def _make_pipeline():
    formatter = GalleyFormatter(LogicCompiler(FinchJLCompiler(DefaultFinchJLRuntime())))
    optimizer = GalleyLogicFactorizer(LogicSimplify(BFSLoopOrderer(formatter)))
    executor = LogicExecutor(optimizer, stats_factory=DCStatsFactory())
    return LogicNormalizer(executor), formatter


@pytest.mark.parametrize("metric", ["optimize", "downstream"])
@pytest.mark.parametrize(
    "empty_last",
    [
        pytest.param(True, id="empty_last"),
        pytest.param(False, id="dense_last", marks=pytest.mark.slow),
    ],
)
def test_galley_matmul_chain(
    benchmark, monkeypatch, empty_last: bool, metric: Literal["optimize", "downstream"]
) -> None:
    import finch.autoschedule.factorizer.galley_factorizer.galley_optimize as galley

    # Warmup
    pipeline, formatter = _make_pipeline()
    plan = _plan_from_lazy(_build_expr(empty_last))
    pipeline(plan)

    # Benchmark
    if metric == "optimize":
        patch_benchmark(benchmark, monkeypatch, galley, "optimize_plan")
    else:
        patch_benchmark(benchmark, monkeypatch, formatter, "lower")

    pipeline(plan)
