"""
Julia backend benchmarks: sparse-sparse matmul and statistics construction on
the Boeing ct20stif matrix, fetched from SuiteSparse via ssgetpy.

Skipped if the Julia backend (juliacall/juliapkg) or ssgetpy aren't
installed -- both are part of the ``julia`` extra, see pyproject.toml.

Run: ``pixi run --environment=benchmark-julia pytest --codspeed
benchmarks/test_julia_boeing.py``

Statistics use default factory settings and warm kernel/factory caches. Matrix
loading and compilation are excluded. SamplingStats leaves its scan deferred;
ExactStats copies the tensor and defers counting. BlockedStats is
excluded because its block extraction builds large dense selectors.
"""

from pathlib import Path

import pytest

import scipy.io

import finch as ft
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.autoschedule.tensor_stats import (
    BlockedUniformStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    DummyStatsFactory,
    FDStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
)
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.compile_jl.julia import julia_available
from finch.finch_logic import Field

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


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(DummyStatsFactory, id="dummy"),
        pytest.param(DenseStatsFactory, id="dense"),
        pytest.param(FDStatsFactory, id="fd"),
        pytest.param(UniformStatsFactory, id="uniform"),
        pytest.param(VPStatsFactory, id="vp"),
        pytest.param(DCStatsFactory, id="dc"),
        pytest.param(LPStatsFactory, id="lp"),
        pytest.param(SamplingStatsFactory, id="sampling"),
        pytest.param(ExactStatsFactory, id="exact"),
        pytest.param(BlockedUniformStatsFactory, id="blocked_uniform"),
    ],
)
def test_julia_stats_ct20stif(boeing_tensor, benchmark, factory):
    stats_factory = factory()
    fields = (Field("i"), Field("j"))
    with with_default_scheduler(COMPILE_JULIA):
        stats_factory(boeing_tensor, fields)
        benchmark(stats_factory, boeing_tensor, fields)
