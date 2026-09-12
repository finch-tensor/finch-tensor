from pathlib import Path

import pytest

import scipy.io

import finch as ft
from finch import ffuncs
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.autoschedule.tensor_stats import (
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
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
def tensor(request):
    name, group = request.param
    matrix_info = next(
        matrix
        for matrix in ssgetpy.search(name=name, group=group)
        if matrix.name == name and matrix.group == group
    )
    localdestpath, _ = matrix_info.download(format="MM", extract=True)
    mtx_path = Path(localdestpath) / f"{name}.mtx"
    matrix = scipy.io.mmread(mtx_path).tocsr()
    return matrix, ft.asarray(matrix)


def actual_hadamard(a, b):
    return max(int(a.multiply(b).count_nonzero()), 1)


def actual_spgemm(a, b):
    result = a.astype(bool).astype(float) @ b.astype(bool).astype(float)
    return max(int(result.count_nonzero()), 1)


def actual_spgemm2(a, b):
    result = (a.astype(bool).astype(float) @ b.astype(bool).astype(float) > 0).astype(
        float
    )
    result = result @ b.astype(bool).astype(float)
    return max(int(result.count_nonzero()), 1)


def actual_triangle(a):
    result = a.astype(bool).astype(float) @ a.astype(bool).astype(float)
    return max(int(result.multiply(a).count_nonzero()), 1)


def est_hadamard(factory, tns_a, tns_b):
    s_a = factory(tns_a, (Field("i"), Field("j")))
    s_b = factory(tns_b, (Field("i"), Field("j")))
    return factory.mapjoin(
        ffuncs.mul,
        s_a,
        s_b,
    ).estimate_non_fill_values()


def est_spgemm(factory, tns_a, tns_b):
    s_a = factory(tns_a, (Field("i"), Field("k")))
    s_b = factory(tns_b, (Field("k"), Field("j")))
    mm = factory.mapjoin(ffuncs.mul, s_a, s_b)

    return factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        mm,
    ).estimate_non_fill_values()


def est_spgemm2(factory, tns_a, tns_b):
    s_a = factory(tns_a, (Field("i"), Field("l")))
    s_b1 = factory(tns_b, (Field("l"), Field("k")))
    mm1 = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("l"),),
        factory.mapjoin(ffuncs.mul, s_a, s_b1),
    )

    s_b2 = factory(tns_b, (Field("k"), Field("j")))
    mm2 = factory.mapjoin(ffuncs.mul, mm1, s_b2)

    return factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        mm2,
    ).estimate_non_fill_values()


def est_triangle(factory, tns_a):
    s_a1 = factory(tns_a, (Field("i"), Field("k")))
    s_a2 = factory(tns_a, (Field("k"), Field("j")))
    mm = factory.aggregate(
        ffuncs.add,
        0.0,
        (Field("k"),),
        factory.mapjoin(ffuncs.mul, s_a1, s_a2),
    )

    s_a3 = factory(tns_a, (Field("i"), Field("j")))
    return factory.mapjoin(
        ffuncs.mul,
        mm,
        s_a3,
    ).estimate_non_fill_values()


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(("web-NotreDame", "SNAP"), id="snap-web-notredame"),
        pytest.param(("ct20stif", "Boeing"), id="boeing-ct20stif"),
        pytest.param(("bcsstk39", "Boeing"), id="boeing-bcsstk39"),
        pytest.param(("ca-GrQc", "SNAP"), id="snap-ca-grqc"),
        pytest.param(("ca-HepTh", "SNAP"), id="snap-ca-hepth"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(UniformStatsFactory(), id="uniform"),
        pytest.param(DCStatsFactory(), id="dc"),
        pytest.param(LPStatsFactory(), id="lp"),
        pytest.param(VPStatsFactory(), id="vp"),
        pytest.param(DenseStatsFactory(), id="dense"),
    ],
)
@pytest.mark.parametrize(
    "estimator, actual, count",
    [
        pytest.param(est_hadamard, actual_hadamard, 2, id="hadamard"),
        pytest.param(est_spgemm, actual_spgemm, 2, id="spgemm"),
        pytest.param(est_spgemm2, actual_spgemm2, 2, id="spgemm-2"),
        pytest.param(est_triangle, actual_triangle, 1, id="triangle"),
    ],
)
def test_estimated_kernel(
    tensor,
    factory,
    estimator,
    actual,
    count,
    benchmark,
    record_property,
):
    matrix, finch_tensor = tensor
    act_nnz = actual(*([matrix] * count))
    ops = [finch_tensor] * count

    with with_default_scheduler(COMPILE_JULIA):
        # warmup
        estimator(factory, *ops)

        est_nnz = benchmark(estimator, factory, *ops)

    ratio = max(est_nnz, 1) / act_nnz

    record_property("actual_nnz", act_nnz)
    record_property("estimated_nnz", est_nnz)
    record_property("ratio", ratio)
