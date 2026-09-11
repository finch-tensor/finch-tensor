from pathlib import Path

import pytest

import scipy.io

import finch as ft
from finch import ffuncs
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.autoschedule.tensor_stats import (
    UniformStatsFactory,
)
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


def est_spgemm(factory, a, b):
    s_a = factory(a, (Field("i"), Field("k")))
    s_b = factory(b, (Field("k"), Field("j")))
    product = factory.mapjoin(ffuncs.mul, s_a, s_b)

    return factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        product,
    ).estimate_non_fill_values()


def est_spgemm2(factory, a, b, c):
    s_a = factory(a, (Field("i"), Field("l")))
    s_b1 = factory(b, (Field("l"), Field("k")))
    product = factory.mapjoin(ffuncs.mul, s_a, s_b1)
    product = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("l"),),
        product,
    )

    s_c = factory(c, (Field("k"), Field("j")))
    product = factory.mapjoin(ffuncs.mul, product, s_c)

    return factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        product,
    ).estimate_non_fill_values()


def est_spgemm3(factory, a, b, c, d):
    s_a = factory(a, (Field("i"), Field("m")))
    s_b = factory(b, (Field("m"), Field("l")))
    product = factory.mapjoin(ffuncs.mul, s_a, s_b)
    product = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("m"),),
        product,
    )

    s_c = factory(c, (Field("l"), Field("k")))
    product = factory.mapjoin(ffuncs.mul, product, s_c)
    product = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("l"),),
        product,
    )

    s_d = factory(d, (Field("k"), Field("j")))
    product = factory.mapjoin(ffuncs.mul, product, s_d)

    return factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        product,
    ).estimate_non_fill_values()


def est_spgemm4(factory, a, b, c, d, e):
    s_a = factory(a, (Field("i"), Field("n")))
    s_b = factory(b, (Field("n"), Field("m")))
    product = factory.mapjoin(ffuncs.mul, s_a, s_b)
    product = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("n"),),
        product,
    )

    s_c = factory(c, (Field("m"), Field("l")))
    product = factory.mapjoin(ffuncs.mul, product, s_c)
    product = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("m"),),
        product,
    )

    s_d = factory(d, (Field("l"), Field("k")))
    product = factory.mapjoin(ffuncs.mul, product, s_d)
    product = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("l"),),
        product,
    )

    s_e = factory(e, (Field("k"), Field("j")))
    product = factory.mapjoin(ffuncs.mul, product, s_e)

    return factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        product,
    ).estimate_non_fill_values()


@pytest.mark.parametrize(
    "estimator, count",
    [
        pytest.param(est_spgemm, 2, id="spgemm-1"),
        pytest.param(est_spgemm2, 3, id="spgemm-2"),
        pytest.param(est_spgemm3, 4, id="spgemm-3"),
        pytest.param(est_spgemm4, 5, id="spgemm-4"),
    ],
)
def test_estimated_spgemm(
    boeing_tensor,
    benchmark,
    estimator,
    count,
):
    ops = [boeing_tensor] * count
    factory = UniformStatsFactory()

    with with_default_scheduler(COMPILE_JULIA):
        # Warmup
        estimator(factory, *ops)

        # Benchmark
        benchmark(estimator, factory, *ops)
