from pathlib import Path

import pytest

import scipy.sparse as sps

import finch as ft
from finch.autoschedule import LogicExecutor, LogicNormalizer
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.formatter import GalleyFormatter
from finch.autoschedule.loop_orderer import DefaultLoopOrderer
from finch.autoschedule.tensor_stats.errorlogger_stats import (
    ErrorLoggerStatsFactory,
    ErrorReport,
)
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.julia import julia_available
from finch.finch_logic import LogicSimplify
from finch.tests.stats_cases import DATASETS, make_models

jl_load = GalleyLogicFactorizer(
    LogicSimplify(
        DefaultLoopOrderer(
            GalleyFormatter(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())))
        )
    ),
    optimizer="bfs",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not julia_available(), reason="errorlogger statistics require the Julia backend"
)
def test_errorlogger_uniform_random_matmul():
    tensor = ft.asarray(sps.csr_array(DATASETS["Uniform Random"]()))
    factories = tuple(make_models().values())
    logger = ErrorLoggerStatsFactory(*factories)
    schedule = LogicNormalizer(LogicExecutor(jl_load, stats_factory=logger))
    with ft.with_default_scheduler(schedule):
        ft.compute(ft.defer(tensor) @ ft.defer(tensor))
    assert set(logger.log.samples) == set(factories)
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
def test_errorlogger_ca_grqc_matmul():
    tensor = ft.asarray(sps.csr_array(DATASETS["ca-GrQc"]()))
    factories = tuple(make_models().values())
    logger = ErrorLoggerStatsFactory(*factories)
    schedule = LogicNormalizer(LogicExecutor(jl_load, stats_factory=logger))
    with ft.with_default_scheduler(schedule):
        ft.compute(ft.defer(tensor) @ ft.defer(tensor))
    assert set(logger.log.samples) == set(factories)
    ErrorReport(logger.log).plot(
        str(Path(__file__).parent / "reference" / "test_errorlogger_ca_grqc_matmul.png")
    )
