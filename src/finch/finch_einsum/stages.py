from abc import abstractmethod
from typing import Any

from finch.algebra import TensorFType
from finch.symbolic import Stage, UnvalidatedForm

from . import nodes as ein


class EinsumEvaluator(Stage):
    @abstractmethod
    def lower(
        self,
        term: ein.EinsumNode,
        bindings: dict[ein.Alias, Any] | None = None,
    ):  # TODO eventually Any->Tensor
        """
        Evaluate the given logic.
        """


class EinsumLoader(Stage):
    @abstractmethod
    def lower(
        self,
        term: ein.EinsumStatement,
        bindings: dict[ein.Alias, TensorFType],
        *,
        defined: set[ein.Alias] | None = None,
    ):
        """
        Generate Finch Library from the given logic and input types, with a
        single method called `main` which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel.
        Aliases in `defined` are defined by the program and have no input value.
        """


class EinsumTransform(Stage):
    @abstractmethod
    def lower(
        self, term: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Transform the given logic term into another logic term.
        """


class OptEinsumLoader(UnvalidatedForm, EinsumLoader):
    def __init__(self, *opts: EinsumTransform, ctx: EinsumLoader):
        self.ctx = ctx
        self.opts = opts

    def lower(
        self,
        term: ein.EinsumStatement,
        bindings: dict[ein.Alias, TensorFType],
        *,
        defined: set[ein.Alias] | None = None,
    ):
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings, defined=defined)
