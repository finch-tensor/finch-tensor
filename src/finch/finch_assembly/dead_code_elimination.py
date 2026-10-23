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
from .dataflow import assembly_dataflow_run, has_effects
from .liveness import AssemblyLiveVariables
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyStatement,
    AssemblyTree,
    Assign,
    Block,
    BufferLoop,
    ForLoop,
    Function,
    If,
    IfElse,
    Module,
    Variable,
    WhileLoop,
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


class _DeadCodeRemover:
    """
    walks the numbered program tracking the variables in scope, and drops the
    dead statements, using what is live after each one.
    """

    def __init__(self, ctx: AssemblyLiveVariables):
        self.live = {stmt.sid: state for stmt, state in ctx.statement_states()}
        self.scopes: list[set[str]] = []

    def __call__(self, node: AssemblyNode) -> AssemblyNode:
        match node:
            case Module(funcs):
                return Module(tuple(self.function(func) for func in funcs))
            case Function():
                return self.function(node)
        return node

    def function(self, func: Function) -> Function:
        args = {arg.name for arg in func.args}
        return Function(func.name, func.args, self.scoped(func.body, args))

    def scoped(
        self, body: AssemblyStatement, decls: set[str] | None = None
    ) -> AssemblyStatement:
        self.scopes.append(set(decls or ()))
        try:
            match body:
                case Block(bodies):
                    return Block(self.block(bodies))
            kept = self.block((body,))
            match kept:
                case (AssemblyStatement() as stmt,):
                    return stmt
            return Block(kept)
        finally:
            self.scopes.pop()

    def block(self, bodies: Sequence) -> tuple:
        kept = []
        for i, stmt in enumerate(bodies):
            if not self.dead(stmt, bodies[i + 1 :]):
                kept.append(self.statement(stmt))
        return tuple(kept)

    def statement(self, node):
        match node:
            case Block():
                return self.scoped(node)
            case If(cond, body):
                return If(cond, self.scoped(body))
            case IfElse(cond, body, else_body):
                return IfElse(cond, self.scoped(body), self.scoped(else_body))
            case WhileLoop(cond, body):
                return WhileLoop(cond, self.scoped(body))
            case ForLoop(var, start, end, body):
                return ForLoop(var, start, end, self.scoped(body, {var.name}))
            case BufferLoop(buf, var, body):
                return BufferLoop(buf, var, self.scoped(body, {var.name}))
            case NumberedStatement(Assign(Variable(name, _), _), _) if not any(
                name in scope for scope in self.scopes
            ):
                self.scopes[-1].add(name)
        return node

    def dead(self, stmt, rest: Sequence) -> bool:
        match stmt:
            case NumberedStatement(Assign(Variable(name, _), _) as assign, sid):
                after = self.live.get(sid)
                if after is None or name in after or has_effects(assign):
                    return False
                if any(name in scope for scope in self.scopes):
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
