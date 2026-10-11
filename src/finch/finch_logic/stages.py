from __future__ import annotations

from abc import abstractmethod

from finch.algebra.tensor import Tensor, TensorFType
from finch.finch_assembly import AssemblyLibrary
from finch.symbolic import Stage

from . import nodes as lgc
from .tensor_stats import StatsFactory, TensorStats


class LogicEvaluator(Stage):
    @abstractmethod
    def lower(
        self, term: lgc.LogicNode, bindings: dict[lgc.Alias, Tensor] | None = None
    ):
        """
        Evaluate the given logic.
        """


class LogicLoader(Stage):
    @abstractmethod
    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        lgc.LogicStatement,
    ]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called `main` which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel, and
        the final program as lowered, for bind-time inference (e.g. resolving
        dynamic output fills against actual argument fills).
        """
