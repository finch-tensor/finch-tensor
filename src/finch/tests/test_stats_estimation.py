"""
Checks estimator accuracy against actual kernel results. One .yml per factory.

Run:
    pixi run --environment=test-julia pytest -s \
        src/finch/tests/test_stats_estimation.py
The test accepts a new baseline result when you add --force-regen to the run command.
"""

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

DATA_DIR = Path(__file__).parent / "data"

pytestmark = pytest.mark.skipif(
    not julia_available(),
    reason="Julia backend (juliacall/juliapkg) not installed",
)


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


def act_hadamard(a, b):
    return max(a.multiply(b).nnz, 1)


def act_spgemm(a, b):
    return max((a @ b).nnz, 1)


def act_spgemm2(a, b):
    return max(((a @ b) @ b).nnz, 1)


def act_triangle(a):
    return max((a @ a).multiply(a).nnz, 1)


MATRICES = [
    ("ct20stif", "Boeing"),
    ("bcsstk39", "Boeing"),
    ("ca-GrQc", "SNAP"),
    ("ca-HepTh", "SNAP"),
    ("web-NotreDame", "SNAP"),
]


KERNELS = [
    ("hadamard", est_hadamard, act_hadamard, 2),
    ("spgemm", est_spgemm, act_spgemm, 2),
    ("spgemm-2", est_spgemm2, act_spgemm2, 2),
    ("triangle", est_triangle, act_triangle, 1),
]


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
def test_estimated_kernel(factory_name, factory, data_regression):
    rows = []

    for matrix_name, matrix_group in MATRICES:
        matrix = scipy.io.mmread(DATA_DIR / f"{matrix_name}.mtx").tocsr()
        finch_tensor = ft.asarray(matrix)
        pattern = matrix.astype(bool)
        pattern.eliminate_zeros()

        for kernel_name, estimator, actual_kernel, count in KERNELS:
            act_nnz = actual_kernel(*([pattern] * count))
            est_ops = [finch_tensor] * count

            with with_default_scheduler(COMPILE_JULIA):
                est_nnz = estimator(factory, *est_ops)

            ratio = max(est_nnz, 1) / act_nnz

            rows.append(
                {
                    "group": matrix_group,
                    "matrix": matrix_name,
                    "kernel": kernel_name,
                    "factory": factory_name,
                    "actual_nnz": act_nnz,
                    "estimated_nnz": float(f"{est_nnz:.6g}"),
                    "ratio": float(f"{ratio:.6g}"),
                }
            )

    data_regression.check(rows)
