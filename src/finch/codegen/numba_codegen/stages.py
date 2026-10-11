from abc import abstractmethod

from finch import finch_assembly as asm
from finch.symbolic import CompilerMode, Stage


class NumbaCode:
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code


class NumbaLowerer(Stage):
    @abstractmethod
    def lower(self, prgm: asm.Module, *, mode: CompilerMode | None = None) -> NumbaCode:
        """
        Lower the given assembly program to Numba code.
        """


__all__ = ["NumbaCode", "NumbaLowerer"]
