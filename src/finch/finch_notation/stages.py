from abc import abstractmethod

from finch import finch_assembly as asm
from finch.symbolic import CompilerMode, Stage

from . import nodes as ntn


class NotationLoader(Stage):
    mode = CompilerMode()

    @abstractmethod
    def lower(self, prgm: ntn.Module) -> asm.AssemblyLibrary:
        """
        Load the given notation program into a runnable module.
        """


class NotationTransform(Stage):
    @abstractmethod
    def lower(self, prgm: ntn.Module) -> ntn.Module:
        """
        Transform the given notation program into another notation program.
        """
