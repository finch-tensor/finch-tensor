"""
Julia backend CodSpeed benchmark: Bellman-Ford shortest paths kernel.

Run: ``pixi run --environment=benchmark-julia pytest --codspeed
benchmarks/test_julia_bellman_ford.py``
"""

import pytest

import numpy as np
import scipy.io
from scipy.sparse.csgraph import shortest_path

import finch as ft
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.compile_jl.julia import julia_available
from finch.tests.stats_cases import DATA_DIR

pytestmark = pytest.mark.skipif(
    not julia_available(),
    reason="Julia backend (juliacall/juliapkg) not installed",
)


@ft.jit
def bellman_ford(G, D, max_iter, xp):
    t = 0
    while t < max_iter:
        D = xp.min(xp.expand_dims(D, 1) + G, axis=0)
        t += 1
    return D


@pytest.fixture(scope="session")
def ca_grqc_graph():
    matrix = scipy.io.mmread(DATA_DIR / "ca-GrQc.mtx").tocsr()
    matrix.setdiag(0.0)
    graph = matrix.toarray()
    graph[graph == 0] = np.inf
    np.fill_diagonal(graph, 0.0)
    return ft.asarray(graph), matrix


def test_julia_bellman_ford(ca_grqc_graph, benchmark):
    G, matrix = ca_grqc_graph
    n = matrix.shape[0]
    d = np.full(n, np.inf)
    # node 0 is in a small component, node 2 in the largest one
    d[2] = 0.0
    D = ft.asarray(d)
    scipy_distances = shortest_path(matrix, method="BF", indices=2)

    with with_default_scheduler(COMPILE_JULIA):
        # We know this converges in 13 iterations on this graph
        result = bellman_ford(G, D, 20, ft)
        np.testing.assert_allclose(ft.to_numpy(result), scipy_distances)

        benchmark(bellman_ford, G, D, 20, ft)
