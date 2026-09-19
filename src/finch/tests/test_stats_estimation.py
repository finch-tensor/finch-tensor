import csv
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

# Matrices used by the statistics benchmarks.
# Each tuple contains the SuiteSparse matrix name and group.
TARGET_MATRICES = [
    pytest.param(("ct20stif", "Boeing"), id="boeing-ct20stif"),
    pytest.param(("bcsstk39", "Boeing"), id="boeing-bcsstk39"),
    pytest.param(("ca-GrQc", "SNAP"), id="snap-ca-grqc"),
    pytest.param(("ca-HepTh", "SNAP"), id="snap-ca-hepth"),
    pytest.param(("web-NotreDame", "SNAP"), id="snap-web-notredame"),
]

ACTUAL_NNZ = Path(__file__).with_name("data") / "stats_actual_nnz.csv"


@pytest.fixture(scope="session")
def stats_data():
    path = Path("junit/stats_accuracy.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "matrix",
                "factory",
                "kernel",
                "actual_nnz",
                "estimated_nnz",
                "ratio",
            ],
        )
        writer.writeheader()

    with ACTUAL_NNZ.open(newline="") as input_file:
        actual_nnz = {
            (row["matrix"], row["group"], row["kernel"]): int(row["actual_nnz"])
            for row in csv.DictReader(input_file)
        }

    return path, actual_nnz


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
    return name, group, ft.asarray(matrix)


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
    TARGET_MATRICES,
    indirect=True,
)
@pytest.mark.parametrize(
    "factory_name, factory",
    [
        pytest.param("uniform", UniformStatsFactory(), id="uniform"),
        pytest.param("dc", DCStatsFactory(), id="dc"),
        pytest.param("lp", LPStatsFactory(), id="lp"),
        pytest.param("vp", VPStatsFactory(), id="vp"),
        pytest.param("dense", DenseStatsFactory(), id="dense"),
    ],
)
@pytest.mark.parametrize(
    "kernel_name, estimator, count",
    [
        pytest.param("hadamard", est_hadamard, 2, id="hadamard"),
        pytest.param("spgemm", est_spgemm, 2, id="spgemm"),
        pytest.param("spgemm-2", est_spgemm2, 2, id="spgemm-2"),
        pytest.param("triangle", est_triangle, 1, id="triangle"),
    ],
)
def test_estimated_kernel(
    tensor,
    factory_name,
    factory,
    kernel_name,
    estimator,
    count,
    benchmark,
    stats_data,
):
    stats_csv, actual_nnz = stats_data
    matrix_name, matrix_group, finch_tensor = tensor
    act_nnz = actual_nnz[(matrix_name, matrix_group, kernel_name)]
    ops = [finch_tensor] * count

    with with_default_scheduler(COMPILE_JULIA):
        # Warmup
        estimator(factory, *ops)

        # Benchmark
        est_nnz = benchmark(estimator, factory, *ops)

    ratio = max(est_nnz, 1) / act_nnz

    with stats_csv.open("a", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                matrix_name,
                factory_name,
                kernel_name,
                act_nnz,
                est_nnz,
                ratio,
            ]
        )
