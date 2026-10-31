from pathlib import Path

import pytest

import scipy.sparse as sps

import finch as ft
from finch.autoschedule import (
    LogicExecutor,
    LogicNormalizer,
)
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.default_schedulers import FinchJLCompiler
from finch.autoschedule.factorizer.optimize import DefaultLogicFactorizer
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer import DefaultLoopOrderer
from finch.autoschedule.tensor_stats.errorlogger_stats import (
    ErrorLog,
    ErrorLoggerStatsFactory,
    ErrorReport,
)
from finch.compile_jl.julia import julia_available
from finch.finch_logic import LogicSimplify
from finch.tests.stats_cases import DATASETS, make_kernel_stats, make_models

jl_load = DefaultLogicFactorizer(
    LogicSimplify(
        DefaultLoopOrderer(
            DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())))
        )
    )
)


def matmul_nnz(matrix, batch_rows=128):
    pattern = matrix.astype(bool)
    actual = 0
    for start in range(0, pattern.shape[0], batch_rows):
        rows = pattern[start : start + batch_rows]
        squared = rows @ pattern
        actual += int(squared.count_nonzero())
    return actual


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(), reason="errorlogger statistics require the Julia backend"
)
def test_errorlogger_uniform_random_matmul():
    matrix = sps.csr_array(DATASETS["Uniform Random"]())
    tensor = ft.asarray(matrix)
    factories = tuple(make_models().values())
    logger = ErrorLoggerStatsFactory(*factories, log=ErrorLog())
    schedule = LogicNormalizer(LogicExecutor(jl_load, stats_factory=logger))
    kernel = make_kernel_stats(logger, tensor)("SpGEMM")
    with ft.with_default_scheduler(schedule):
        count = kernel.estimate_non_fill_values(exact_count=matmul_nnz(matrix))
        assert set(logger.log.samples) == set(factories)
        assert count == pytest.approx(logger.log.samples[factories[0]][1])
    ErrorReport(logger.log).plot(
        str(
            Path(__file__).parent
            / "reference"
            / "test_errorlogger_uniform_random_matmul.png"
        )
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(),
    reason="errorlogger statistics require the Julia backend",
)
def test_errorlogger_web_notre_dame_matmul():
    matrix = sps.csr_array(DATASETS["web-NotreDame"]())
    tensor = ft.asarray(matrix)
    factories = tuple(make_models().values())
    logger = ErrorLoggerStatsFactory(*factories, log=ErrorLog())
    schedule = LogicNormalizer(LogicExecutor(jl_load, stats_factory=logger))
    kernel = make_kernel_stats(logger, tensor)("SpGEMM")
    with ft.with_default_scheduler(schedule):
        count = kernel.estimate_non_fill_values(exact_count=matmul_nnz(matrix))
        assert set(logger.log.samples) == set(factories)
        assert count == pytest.approx(logger.log.samples[factories[0]][1])
    ErrorReport(logger.log).plot(
        str(
            Path(__file__).parent
            / "reference"
            / "test_errorlogger_web_notre_dame_matmul.png"
        )
    )
