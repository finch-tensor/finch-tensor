"""
reaching definitions for finch assembly.

a forward may-analysis: for every variable, the ids of the statements whose
definition may reach a program point. function arguments are defined by the
synthetic prologue statements of the cfg, and a setattr is a partial definition
of the variable at the root of its target, so it adds to the reaching ones.
"""

from collections import defaultdict
from collections.abc import Iterable

from .cfg_builder import NumberedStatement
from .dataflow import AbstractAssemblyDataflow, root_of, variables_read
from .nodes import AssemblyNode, Assign, SetAttr, Variable


class AssemblyReachingDefinitions(AbstractAssemblyDataflow):
    """
    reaching definitions, see the module docstring. a state maps variable names to
    frozensets of statement ids.
    """

    def direction(self) -> str:
        return "forward"

    def join(self, state_1: dict, state_2: dict) -> dict:
        return {
            name: state_1.get(name, frozenset()) | state_2.get(name, frozenset())
            for name in state_1.keys() | state_2.keys()
        }

    def transfer(self, stmts: Iterable[AssemblyNode], state: dict) -> dict:
        state = dict(state)
        for stmt in stmts:
            # unnumbered statements only define the cfg's own loop counters
            match stmt:
                case NumberedStatement(Assign(Variable(name, _), _), sid):
                    state[name] = frozenset((sid,))
                case NumberedStatement(SetAttr(obj, _, _), sid):
                    match root_of(obj):
                        case Variable(name, _):
                            state[name] = state.get(name, frozenset()) | {sid}
        return state

    def use_def_chains(self) -> dict[tuple[int, str], frozenset[int]]:
        """map (statement id, variable it reads) to the definitions reaching it"""
        return {
            (stmt.sid, name): state.get(name, frozenset())
            for stmt, state in self.statement_states()
            for name in variables_read(stmt)
        }

    def def_use_chains(self) -> dict[int, frozenset[int]]:
        """map each definition to the statements whose reads it reaches"""
        uses: dict[int, set[int]] = defaultdict(set)
        for (sid, _), defs in self.use_def_chains().items():
            for def_sid in defs:
                uses[def_sid].add(sid)
        return {def_sid: frozenset(sids) for def_sid, sids in uses.items()}

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        return [
            (name, "{" + ", ".join(map(str, sorted(state.get(name, ())))) + "}")
            for name in variables_read(stmt)
        ]
