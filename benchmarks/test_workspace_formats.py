"""
Workspace format benchmarks: time the Julia kernel for C = A @ B, looping over
(i, j, k), when C is written directly in each format, or when each row of C is
accumulated in a workspace and copied in order into a sorted list, as the
workspaces of ``WorkspaceLoopOrderer`` are. Inputs are CSR. Every result is
checked against SciPy before it is timed.

The looping order scatters the writes to each row of C. A dense output takes
them cheaply, but stores and resets all n^2 entries, while hash and byte map
outputs pay for every scattered write. A byte map workspace takes the scattered
writes of one row, and its copy into the sorted list only visits the entries
the row stored, so it wins whenever the rows of C are sparse, by more as n
grows. Where each row of C is mostly full, as in the block and dense workloads,
a dense output or workspace is faster. A dense workspace visits every entry of
each row when it is copied, so it only pays off for dense rows.

Run: ``pixi run --environment=benchmark-julia pytest --codspeed
benchmarks/test_workspace_formats.py``
"""

from functools import cache

import pytest

import numpy as np
import scipy.sparse as sps

import finch as ft
import finch.finch_logic as lgc
from finch import ffuncs
from finch.algebra import StaticFillFType, ftypes
from finch.autoschedule import NotationGenerator
from finch.autoschedule.loop_orderer.loop_order_workspace import add_workspace
from finch.autoschedule.util import fuse_loop_order
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.interop import jl_tensor_to_python, tensor_to_jl
from finch.compile_jl.julia import jl, julia_available
from finch.tensor import (
    dense,
    element,
    fiber_tensor,
    sparse_bytemap,
    sparse_hash,
    sparse_list,
)

pytestmark = pytest.mark.skipif(
    not julia_available(),
    reason="the julia extra (juliapkg, juliacall) is not installed",
)

# The level of the workspace, if any, and the level of each row of C.
STRATEGIES = {
    "direct_dense": (None, dense),
    "direct_hash": (None, sparse_hash),
    "direct_bytemap": (None, sparse_bytemap),
    "workspace_dense": (dense, sparse_list),
    "workspace_bytemap": (sparse_bytemap, sparse_list),
}

# Dense outputs of more entries than this are skipped.
MAX_DENSE_OUTPUT = 2000 * 2000


def _random_rows(rows, cols, per_row, col_limit, seed):
    """`per_row` distinct columns per row, uniform over the first `col_limit`."""
    rng = np.random.default_rng(seed)
    idx = np.stack(
        [np.sort(rng.choice(col_limit, per_row, replace=False)) for _ in range(rows)]
    )
    ptr = np.arange(0, rows * per_row + 1, per_row)
    data = rng.random(rows * per_row)
    return sps.csr_array((data, idx.ravel(), ptr), shape=(rows, cols))


def _random(n):
    # Few products land on each entry of C.
    return _random_rows(n, n, 8, n, 1), _random_rows(n, n, 8, n, 2)


def _banded(n):
    # A @ A for a band of half-width 8, with about 9 products per entry of C.
    offsets = range(-8, 9)
    a = sps.diags([np.ones(n - abs(d)) for d in offsets], offsets, format="csr")
    a.data = np.random.default_rng(1).random(a.nnz)
    return a, a


def _powerlaw(n):
    # A @ A for a graph whose columns are drawn from a Zipf law, so that a few
    # columns of C are hit by many rows.
    rng = np.random.default_rng(1)
    weights = 1.0 / np.arange(1, n + 1) ** 1.1
    cols = rng.permutation(n)[rng.choice(n, n * 8, p=weights / weights.sum())]
    rows = np.repeat(np.arange(n), 8)
    a = sps.csr_array((rng.random(n * 8), (rows, cols)), shape=(n, n))
    a.sum_duplicates()
    return a, a


def _block(n):
    # Every row of A reads rows of B which share 128 columns, so about four
    # products land on each entry of C.
    a = _random_rows(n, n, 16, 64, 1)
    b = sps.vstack([_random_rows(64, n, 32, 128, 2), sps.csr_array((n - 64, n))])
    return a, b.tocsr()


def _dense(n):
    return _random_rows(n, n, n // 3, n, 1), _random_rows(n, n, n // 3, n, 2)


WORKLOADS = {
    "random_2000": (_random, 2000),
    "banded_2000": (_banded, 2000),
    "powerlaw_2000": (_powerlaw, 2000),
    "block_2000": (_block, 2000),
    "dense_300": (_dense, 300),
    "random_10000": (_random, 10000),
    "powerlaw_10000": (_powerlaw, 10000),
}


@cache
def _inputs(workload):
    generate, n = WORKLOADS[workload]
    return generate(n)


def _kernel(workload, strategy):
    """Compile C = A @ B with the formats of `strategy`, and its arguments."""
    i, j, k = lgc.Field("i"), lgc.Field("j"), lgc.Field("k")
    a, b, c, w = (lgc.HardAlias(name) for name in "abcw")
    data_a, data_b = _inputs(workload)
    tns_a, tns_b = ft.asarray(data_a), ft.asarray(data_b)
    rhs = lgc.Aggregate(
        lgc.Literal(ffuncs.add),
        lgc.Literal(0.0),
        lgc.MapJoin(
            lgc.Literal(ffuncs.mul), (lgc.Table(a, (i, j)), lgc.Table(b, (j, k)))
        ),
        (j,),
    )
    plan = lgc.Plan(
        (
            fuse_loop_order(lgc.Query(lgc.Table(c, (i, k)), rhs), (i, j, k)),
            lgc.Produces((c,)),
        )
    )
    leaf = element(StaticFillFType(np.float64(0.0)), ft.float64, ftypes.intp)
    workspace, row = STRATEGIES[strategy]
    bindings = {a: tns_a.ftype, b: tns_b.ftype}
    args = [tns_a, tns_b]
    if workspace is not None:
        plan = add_workspace(plan, c, 1, w)
        bindings[w] = fiber_tensor(workspace(leaf, ftypes.intp))
        args.append(bindings[w].construct((0,)))
    bindings[c] = fiber_tensor(dense(row(leaf, ftypes.intp), ftypes.intp))
    args.append(bindings[c].construct((0, 0)))
    program = NotationGenerator()(plan, bindings, {}, None)
    return FinchJLCompiler()(program).main, args


def _to_scipy(result):
    csr = jl.Tensor(jl.Dense(jl.SparseList(jl.Element(0.0))), tensor_to_jl(result))
    return jl_tensor_to_python(csr).to_scipy()


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("workload", WORKLOADS)
def test_spgemm_workspace_formats(benchmark, workload, strategy):
    data_a, data_b = _inputs(workload)
    if strategy == "direct_dense" and np.prod(data_a.shape) > MAX_DENSE_OUTPUT:
        pytest.skip("the dense output is too large")
    kernel, args = _kernel(workload, strategy)

    # Warmup
    (result,) = kernel(*args)
    assert abs(_to_scipy(result) - data_a @ data_b).max() < 1e-9

    # Benchmark
    benchmark(kernel, *args)
