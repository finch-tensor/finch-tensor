from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from finch.algebra import CallableFType, FType, FTyped
from finch.symbolic import CompilerMode, Stage

if TYPE_CHECKING:
    from . import nodes as asm


@dataclass(eq=False, frozen=True)
class AssemblyKernelFType(CallableFType):
    """The identity and signature of a function defined in a Finch module."""

    name: str
    arg_types: tuple[FType, ...]
    result_type: FType
    definition: str = field(default_factory=lambda: uuid4().hex)

    def __eq__(self, other):
        match other:
            case AssemblyKernelFType():
                return (
                    self.name,
                    self.arg_types,
                    self.result_type,
                    self.definition,
                ) == (other.name, other.arg_types, other.result_type, other.definition)
        return NotImplemented

    def __hash__(self):
        return hash((self.name, self.arg_types, self.result_type, self.definition))

    def __call__(self, value):
        if isinstance(value, AssemblyKernel) and value.ftype == self:
            return value
        raise TypeError(f"Expected a kernel of type {self}")

    def return_type(self, *args: FType) -> FType:
        if args != self.arg_types:
            raise TypeError(f"{self.name} expects {self.arg_types}, got {args}")
        return self.result_type


class AssemblyKernel(FTyped, ABC):
    """A callable implementation of an AssemblyKernelFType."""

    def __init__(self, type_: AssemblyKernelFType):
        self._ftype = type_

    @property
    def ftype(self) -> AssemblyKernelFType:
        return self._ftype

    @abstractmethod
    def __call__(self, *args) -> Any: ...


class AssemblyLibrary(ABC):
    """
    Represents a module containing assembly kernels.
    """

    @abstractmethod
    def __getattr__(self, name: str) -> AssemblyKernel:
        """
        Get the assembly Kernel corresponding to the given name.
        """
        ...


class AssemblyLoader(Stage):
    @abstractmethod
    def lower(
        self, term: asm.Module, *, mode: CompilerMode | None = None
    ) -> AssemblyLibrary:
        """
        Load the given assembly program into a runnable module.
        """


class AssemblyTransform(Stage):
    @abstractmethod
    def lower(self, term: asm.Module) -> asm.Module:
        """
        Transform the given assembly term into another assembly term.
        """
