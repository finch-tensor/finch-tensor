from typing import Any

from finchlite.autoschedule import (
    DefaultLogicOptimizer,
    LogicCompiler,
    LogicExecutor,
    LogicFormatter,
    LogicNormalizer,
    LogicStandardizer,
)
from finchlite.finch_logic import LogicLoader

from .compiler import FinchJLCompiler
from .levels import Dense, Element


class FinchJLLogicFormatter(LogicFormatter):
    def __init__(
        self,
        loader: LogicLoader | None = None,
    ):
        super().__init__(loader)

    def get_output_tns_ftype(self, fill_value: Any, shape_type: tuple[Any, ...]):
        lvl = Element(fill_value)
        for _ in len(shape_type):
            lvl = Dense(lvl)
        return lvl


COMPILE_JULIA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(FinchJLLogicFormatter(LogicCompiler(FinchJLCompiler())))
        )
    )
)
