"""
live variables for finch assembly.

a backward may-analysis: a variable is live at a point if some path from there
reads it before assigning it. reads in conditions and loop bounds show up as the
synthetic asserts and loop counter assignments of the cfg.
"""

from collections.abc import Iterable

from .cfg_builder import NumberedStatement
from .dataflow import AbstractAssemblyDataflow, variables_read
from .nodes import AssemblyNode, Assign, Variable


class AssemblyLiveVariables(AbstractAssemblyDataflow):
    """live variables, see the module docstring. a state maps live names to True."""

    def direction(self) -> str:
        return "backward"

    def join(self, state_1: dict, state_2: dict) -> dict:
        return state_1 | state_2

    def transfer(self, stmts: Iterable[AssemblyNode], state: dict) -> dict:
        state = dict(state)
        for stmt in reversed(list(stmts)):
            match stmt:
                case NumberedStatement(Assign(Variable(name, _), _), _) | Assign(
                    Variable(name, _), _
                ):
                    state.pop(name, None)
            state.update(dict.fromkeys(variables_read(stmt), True))
        return state

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        return [("live", ", ".join(sorted(state)) or "-")]
