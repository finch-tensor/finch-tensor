from abc import abstractmethod

from finch import finch_assembly as asm
from finch.symbolic import CompilerMode, Stage


class MLIRCode:
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code


class MLIRLowerer(Stage):
    @abstractmethod
    def lower(self, prgm: asm.Module, *, mode: CompilerMode | None = None) -> MLIRCode:
        """
        Lower the given assembly program to textual MLIR.

        """


__all__ = ["MLIRCode", "MLIRLowerer"]
