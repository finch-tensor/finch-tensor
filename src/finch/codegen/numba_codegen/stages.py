from abc import abstractmethod
from typing import Any

from finch import finch_assembly as asm
from finch.symbolic import CompilerMode, Stage


class NumbaCode:
    def __init__(self, code: str, structs: dict[str, Any] | None = None):
        self.code = code
        # The jitclasses that the code refers to by its own names.
        self.structs = structs if structs is not None else {}

    def __str__(self) -> str:
        return self.code


class NumbaLowerer(Stage):
    @abstractmethod
    def lower(self, prgm: asm.Module, *, mode: CompilerMode | None = None) -> NumbaCode:
        """
        Lower the given assembly program to Numba code.
        """


__all__ = ["NumbaCode", "NumbaLowerer"]
