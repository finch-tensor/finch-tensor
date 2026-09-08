"""
Julia backend CodSpeed benchmark: Bellman-Ford shortest paths kernel.

Run: ``pixi run --environment=benchmark-julia pytest --codspeed
benchmarks/test_julia_bellman_ford.py``
"""

from pathlib import Path

import pytest

import numpy as np
import scipy.io
from scipy.sparse.csgraph import shortest_path

import finch as ft
from finch.algebra import ftype
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.codegen import NumpyBuffer
from finch.compile_jl.julia import julia_available
from finch.tensor import DenseLevel, ElementLevel, FiberTensor, SparseListLevel

try:
    import ssgetpy
except ImportError:
    ssgetpy = None

pytestmark = pytest.mark.skipif(
    not julia_available() or ssgetpy is None,
    reason="Julia backend (juliacall/juliapkg) or ssgetpy not installed",
)


@ft.jit
def bellman_ford(G, D, max_iter, xp):
    t = 0
    while t < max_iter:
        D = xp.minimum(D, xp.min(xp.expand_dims(D, 1) + G, axis=0))
        t += 1
    return D


def as_finch_csr_with_fill(matrix, fill_value):
    """Wrap a SciPy CSR matrix while preserving an implicit nonzero fill."""
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()

    index_type = ftype(matrix.indices.dtype)
    element_format = ft.element(
        fill_value=fill_value,
        element_type=ftype(matrix.data.dtype),
        position_type=index_type,
    )
    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(element_format, NumpyBuffer(matrix.data)),
                dimension=index_type(matrix.shape[1]),
                ptr=NumpyBuffer(matrix.indptr),
                idx=NumpyBuffer(matrix.indices),
            ),
            dimension=index_type(matrix.shape[0]),
        )
    )


@pytest.fixture(scope="session")
def bcsstk15_graph():
    matrix_info = ssgetpy.search(name="bcsstk15")[0]
    localdestpath, _ = matrix_info.download(format="MM", extract=True)
    mtx_path = Path(localdestpath) / "bcsstk15.mtx"
    matrix = scipy.io.mmread(mtx_path).tocsr()
    matrix.data = np.abs(matrix.data)
    matrix.data[matrix.data == 0] = 1.0
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    return as_finch_csr_with_fill(matrix, np.inf), matrix


def test_julia_bellman_ford(bcsstk15_graph, benchmark):
    G, matrix = bcsstk15_graph
    n = matrix.shape[0]
    d = np.full(n, np.inf)
    d[0] = 0.0
    D = ft.asarray(d)
    scipy_distances = shortest_path(matrix, method="BF", indices=0)

    with with_default_scheduler(COMPILE_JULIA):
        # We know this converges in 39 iterations on this graph
        result = bellman_ford(G, D, 40, ft)
        np.testing.assert_allclose(ft.to_numpy(result), scipy_distances)

        benchmark(bellman_ford, G, D, 40, ft)
