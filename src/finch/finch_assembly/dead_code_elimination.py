"""
dead code elimination for finch assembly.

`assembly_dead_code_elimination` removes assignments whose variable is dead
afterwards (see `AssemblyLiveVariables`) and that have no effects, plus pure bare
expression statements, until nothing changes. an assignment that declares its
variable stays unless the next statement in its block that mentions the variable
re-assigns it, since otherwise a later assignment in a nested scope would become
a local declaration. empty loops and ifs left behind are for `AssemblySimplify`.
"""

from collections.abc import Sequence

from finch.symbolic import UnvalidatedForm

from .cfg_builder import NumberedStatement
from .dataflow import ScopedRewriter, assembly_dataflow_run, has_effects
from .liveness import AssemblyLiveVariables
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyTree,
    Assign,
    Module,
    Variable,
)
from .stages import AssemblyTransform


def assembly_dead_code_elimination(node: AssemblyNode) -> AssemblyNode:
    """remove dead assignments without effects until nothing changes"""
    while True:
        result = assembly_dataflow_run(
            node,
            AssemblyLiveVariables,
            lambda pre_node, ctx: _DeadCodeRemover(ctx)(pre_node),
        )
        if result == node:
            return result
        node = result


class AssemblyDeadCodeEliminationTransform(UnvalidatedForm, AssemblyTransform):
    def lower(self, term: Module) -> Module:
        result = assembly_dead_code_elimination(term)
        assert isinstance(result, Module)
        return result


def _names(node: AssemblyNode) -> set[str]:
    match node:
        case NumberedStatement(stmt, _):
            return _names(stmt)
        case Variable(name, _):
            return {name}
        case AssemblyTree():
            return set().union(*map(_names, node.children))
    return set()


class _DeadCodeRemover(ScopedRewriter):
    """drops the dead statements, using what is live after each one"""

    def __init__(self, ctx: AssemblyLiveVariables):
        super().__init__()
        self.live = {stmt.sid: state for stmt, state in ctx.statement_states()}

    def block(self, bodies: Sequence) -> tuple:
        return tuple(
            self.statement(stmt)
            for i, stmt in enumerate(bodies)
            if not self.dead(stmt, bodies[i + 1 :])
        )

    def dead(self, stmt, rest: Sequence) -> bool:
        match stmt:
            case NumberedStatement(Assign(Variable(name, _), _) as assign, sid):
                after = self.live.get(sid)
                if after is None or name in after or has_effects(assign):
                    return False
                if self.visible(name):
                    return True
                # a declaration can go if the next mention in its block re-assigns
                for later in rest:
                    if name in _names(later):
                        match later:
                            case NumberedStatement(Assign(Variable(other, _), rhs), _):
                                return other == name and name not in _names(rhs)
                        return False
                return True
            case AssemblyExpression():
                return not has_effects(stmt)
        return False
