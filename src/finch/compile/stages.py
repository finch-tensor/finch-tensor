from abc import abstractmethod

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.symbolic import CompilerMode, Stage


class NotationLowerer(Stage):
    mode = CompilerMode()

    @abstractmethod
    def lower(self, term: ntn.Module) -> asm.Module:
        """
        Compile the given notation term into an assembly term.
        """
