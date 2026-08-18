"""
Julia backend ASV benchmark: sparse-sparse matmul on the Boeing ct20stif
matrix (from the SuiteSparse Matrix Collection), fetched via ssgetpy.

Skipped if the Julia backend (juliacall/juliapkg) or ssgetpy aren't
installed -- both are part of the ``julia`` extra, see pyproject.toml.

Run: ``pixi run --environment=test-julia pytest --codspeed
benchmarks/test_julia_boeing.py``
"""

from pathlib import Path

import pytest

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


@pytest.fixture(scope="session")
def boeing_tensor():
    matrix_info = ssgetpy.search(name="ct20stif", group="Boeing")[0]
    localdestpath, _ = matrix_info.download(format="MM", extract=True)
    mtx_path = Path(localdestpath) / "ct20stif.mtx"
    matrix = scipy.io.mmread(mtx_path).tocsr()
    return ft.asarray(matrix)


def test_julia_matmul_ct20stif(boeing_tensor, benchmark):
    with with_default_scheduler(COMPILE_JULIA):
        expr = ft.matmul(ft.defer(boeing_tensor), ft.defer(boeing_tensor))

        # Warmup: JIT-compile the kernel once, outside the timed region.
        ft.compute(expr)

        benchmark(ft.compute, expr)
