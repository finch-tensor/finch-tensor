import threading
from contextlib import contextmanager

from finch.autoschedule.tensor_stats.fd_stats import FDStatsFactory
from finch.codegen import MLIRCompiler, NumbaCompiler
from finch.compile import NotationCompiler
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.julia import julia_available
from finch.finch_assembly import (
    AssemblyInterpreter,
    AssemblySimplify,
    LowerPackedStructSlots,
)
from finch.finch_logic import (
    LogicInterpreter,
    MockLogicLoader,
)
from finch.finch_logic.stages import LogicEvaluator
from finch.finch_notation.interpreter import NotationInterpreter

from .compiler import LogicCompiler
from .executor import LogicExecutor
from .formatter import DefaultLogicFormatter
from .galley_optimize import GalleyLogicalOptimizer
from .loop_ordering import DefaultLoopOrderer
from .normalize import LogicNormalizer
from .optimize import DefaultLogicOptimizer
from .smart_formatter import FDFormatter

INTERPRET_LOGIC = LogicInterpreter()
OPTIMIZE_LOGIC = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(DefaultLogicFormatter(MockLogicLoader()))
        )
    )
)
INTERPRET_NOTATION = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )
    )
)
INTERPRET_ASSEMBLY = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    LogicCompiler(NotationCompiler(AssemblyInterpreter()))
                )
            )
        )
    )
)
COMPILE_NUMBA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    LogicCompiler(
                        NotationCompiler(
                            NumbaCompiler(),
                            ctx_transforms=(
                                LowerPackedStructSlots(),
                                AssemblySimplify(),
                            ),
                        )
                    )
                )
            )
        ),
        cache=True,
    )
)

COMPILE_NUMBA_GALLEY = LogicNormalizer(
    LogicExecutor(
        GalleyLogicalOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    LogicCompiler(
                        NotationCompiler(
                            NumbaCompiler(),
                            ctx_transforms=(
                                LowerPackedStructSlots(),
                                AssemblySimplify(),
                            ),
                        )
                    )
                )
            )
        )
    )
)

INTERPRET_NOTATION_GALLEY = LogicNormalizer(
    LogicExecutor(
        GalleyLogicalOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )
    )
)

COMPILE_MLIR = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(
                    LogicCompiler(
                        NotationCompiler(
                            MLIRCompiler(),
                            ctx_transforms=(
                                LowerPackedStructSlots(),
                                AssemblySimplify(),
                            ),
                        )
                    )
                )
            )
        ),
        cache=True,
    )
)

COMPILE_JULIA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(FDFormatter(LogicCompiler(FinchJLCompiler())))
        ),
        stats_factory=FDStatsFactory(),
    )
)

# This scheduler is used for computing tensor statistics.
# Crucially, in order to avoid a circular dependency, this scheduler does not
# cannot rely on a stats factory that itself calls the interface.
_NON_RECURSIVE_BACKEND = (
    FDFormatter(LogicCompiler(FinchJLCompiler()))
    if julia_available()
    else DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
)

NON_RECURSIVE_SCHEDULER = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(DefaultLoopOrderer(_NON_RECURSIVE_BACKEND)),
        stats_factory=FDStatsFactory(),
        cache=True,
    )
)

_DEFAULT_SCHEDULER = threading.local()


def set_default_scheduler(ctx: LogicEvaluator):
    if ctx is not None:
        _DEFAULT_SCHEDULER.value = ctx


def get_default_scheduler() -> LogicEvaluator:
    try:
        return _DEFAULT_SCHEDULER.value
    except AttributeError:
        return INTERPRET_NOTATION


@contextmanager
def with_default_scheduler(ctx: LogicEvaluator):
    old_value = getattr(_DEFAULT_SCHEDULER, "value", None)
    _DEFAULT_SCHEDULER.value = ctx
    try:
        yield
    finally:
        if old_value is None:
            if hasattr(_DEFAULT_SCHEDULER, "value"):
                delattr(_DEFAULT_SCHEDULER, "value")
        else:
            _DEFAULT_SCHEDULER.value = old_value
