from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.autoschedule.default_schedulers import COMPILE_JULIA
from finch.autoschedule.tensor_stats import (
    BlockedStatsFactory,
    BlockedUniformStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
)
from finch.autoschedule.tensor_stats.errorlogger_stats import (
    ErrorLog,
    ErrorLoggerStatsFactory,
    ErrorReport,
)
from finch.compile_jl.julia import julia_available
from finch.finch_logic import StatsFactory
from finch.tests.stats_cases import DATASETS, make_kernel_stats

STATS: list[StatsFactory[Any]] = [
    DenseStatsFactory(),
    UniformStatsFactory(),
    DCStatsFactory(),
    LPStatsFactory(),
    VPStatsFactory(),
    SamplingStatsFactory(estimator="uj1"),
    SamplingStatsFactory(estimator="sj1"),
    SamplingStatsFactory(estimator="uj2"),
    SamplingStatsFactory(estimator="schlosser"),
    SamplingStatsFactory(estimator="sh2"),
    SamplingStatsFactory(estimator="sh3"),
    SamplingStatsFactory(estimator="good1"),
    SamplingStatsFactory(estimator="silly"),
    BlockedStatsFactory(DenseStatsFactory()),
    BlockedStatsFactory(UniformStatsFactory()),
    BlockedStatsFactory(DCStatsFactory()),
    BlockedStatsFactory(LPStatsFactory()),
    BlockedStatsFactory(VPStatsFactory()),
    BlockedStatsFactory(SamplingStatsFactory(estimator="uj1")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="sj1")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="uj2")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="schlosser")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="sh2")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="sh3")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="good1")),
    BlockedStatsFactory(SamplingStatsFactory(estimator="silly")),
    BlockedUniformStatsFactory(),
]


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(), reason="errorlogger statistics require the Julia backend"
)
def test_errorlogger_matmul_124():
    matrix = sps.csr_array(
        (np.random.default_rng().random((124, 124)) < 0.03).astype(np.float64)
    )
    tensor = ft.asarray(matrix)
    log = ErrorLog()
    for factory in STATS:
        factory = deepcopy(factory)
        logger = ErrorLoggerStatsFactory(factory, error_log=log)
        product = make_kernel_stats(logger, tensor)("SpGEMM")
        with ft.with_default_scheduler(COMPILE_JULIA):
            actual = product.exact.estimate_non_fill_values()
        estimate = product.estimates[factory]
        count = estimate.estimate_non_fill_values()
        logger.record(product)
        assert log.samples[factory] == (actual, count)
    ErrorReport(log).plot(
        str(Path(__file__).parent / "reference" / "test_errorlogger_matmul_124.png")
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(),
    reason="full web-NotreDame statistics require the Julia backend",
)
def test_errorlogger_web_notre_dame_matmul():
    matrix = sps.csr_array(DATASETS["web-NotreDame"]())
    tensor = ft.asarray(matrix)
    log = ErrorLog()
    for factory in STATS:
        factory = deepcopy(factory)
        logger = ErrorLoggerStatsFactory(factory, error_log=log)
        product = make_kernel_stats(logger, tensor)("SpGEMM")
        with ft.with_default_scheduler(COMPILE_JULIA):
            actual = product.exact.estimate_non_fill_values()
        estimate = product.estimates[factory]
        count = estimate.estimate_non_fill_values()
        logger.record(product)
        assert log.samples[factory] == (actual, count)
    ErrorReport(log).plot(
        str(
            Path(__file__).parent
            / "reference"
            / "test_errorlogger_web_notre_dame_matmul.png"
        )
    )
