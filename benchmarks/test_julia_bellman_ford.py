"""
Julia backend CodSpeed benchmark: Bellman-Ford shortest paths kernel with
auto active-set semiring relaxation.

Run: ``pixi run --environment=benchmark-julia pytest --codspeed
benchmarks/test_julia_bellman_ford.py``
"""

from pathlib import Path

import pytest

import numpy as np
import scipy.io

import finch as ft
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.compile_jl.julia import julia_available

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
        D = xp.min(xp.expand_dims(D, 1) + G, axis=0)
        t += 1
    return D


@pytest.fixture(scope="session")
def bcsstk15_graph():
    matrix_info = ssgetpy.search(name="bcsstk15")[0]
    localdestpath, _ = matrix_info.download(format="MM", extract=True)
    mtx_path = Path(localdestpath) / "bcsstk15.mtx"
    matrix = scipy.io.mmread(mtx_path).tocsr()
    return ft.asarray(matrix), matrix.shape[0]


def test_julia_bellman_ford(bcsstk15_graph, benchmark):
    G, n = bcsstk15_graph
    d = np.full(n, np.inf)
    d[0] = 0.0
    D = ft.asarray(d)

    with with_default_scheduler(COMPILE_JULIA):
        # We know this converges in 39 iterations on this graph
        bellman_ford(G, D, 40, ft)

        benchmark(bellman_ford, G, D, 40, ft)
