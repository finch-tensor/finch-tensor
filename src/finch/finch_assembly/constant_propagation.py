"""
constant propagation for finch assembly.

a forward must-analysis that maps a variable to a `Constant` when every path
gives it the same literal, compared by exact type and bits. calls to pure
operators on constants are folded, unless folding would raise or warn.

`assembly_constant_propagation` replaces constant variables and expressions in
each statement with literals. conditions and loop bounds are left as they are.
"""

from collections.abc import Iterable

from finch.symbolic import Constant, PostWalk, Rewrite, UnvalidatedForm
from finch.symbolic.simplification import try_evaluate

from .cfg_builder import NumberedStatement
from .dataflow import (
    AbstractAssemblyDataflow,
    assembly_dataflow_run,
    has_effects,
    literal_constant,
    pure_op_key,
    root_of,
    variables_read,
)
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyStatement,
    Assert,
    Assign,
    Call,
    Literal,
    Load,
    Module,
    Resize,
    Return,
    SetAttr,
    Store,
    Variable,
)
from .stages import AssemblyTransform


def assembly_constant_propagation(node: AssemblyNode) -> AssemblyNode:
    """replace constant variables and expressions with literals"""

    def apply(pre_node: AssemblyNode, ctx: AssemblyConstantPropagation):
        states = {stmt.sid: state for stmt, state in ctx.statement_states()}

        def rw(node):
            match node:
                case NumberedStatement(stmt, sid) if sid in states:
                    return NumberedStatement(ctx.rewrite(stmt, states[sid]), sid)
            return None

        return Rewrite(PostWalk(rw))(pre_node)

    return assembly_dataflow_run(node, AssemblyConstantPropagation, apply)


class AssemblyConstantPropagationTransform(UnvalidatedForm, AssemblyTransform):
    def lower(self, term: Module) -> Module:
        result = assembly_constant_propagation(term)
        assert isinstance(result, Module)
        return result


class AssemblyConstantPropagation(AbstractAssemblyDataflow):
    """
    constant propagation, see the module docstring. a state maps variable names to
    their constants.
    """

    def direction(self) -> str:
        return "forward"

    def join(self, state_1: dict, state_2: dict) -> dict:
        return {
            name: const for name, const in state_1.items() if state_2.get(name) is const
        }

    def transfer(self, stmts: Iterable[AssemblyNode], state: dict) -> dict:
        state = dict(state)
        for stmt in stmts:
            match stmt:
                case NumberedStatement(stmt, _):
                    pass
            match stmt:
                case Assign(Variable(name, _), rhs):
                    const = None if has_effects(stmt) else self.constant(rhs, state)
                    if const is None:
                        state.pop(name, None)
                    else:
                        state[name] = const
                case SetAttr(obj, _, _):
                    match root_of(obj):
                        case Variable(name, _):
                            state.pop(name, None)
        return state

    def constant(self, node: AssemblyExpression, state: dict) -> Constant | None:
        """the value of `node`, if it is a known constant"""
        match node:
            case Literal(val):
                return literal_constant(val)
            case Variable(name, _):
                return state.get(name)
            case Call(Literal(op) as op_lit, args) if pure_op_key(op) is not None:
                consts = [self.constant(arg, state) for arg in args]
                lits = [Literal(const.val) for const in consts if const is not None]
                if len(lits) < len(args):
                    return None
                match try_evaluate(op_lit, lits):
                    case Literal(val):
                        return literal_constant(val)
        return None

    def rewrite(self, stmt: AssemblyStatement, state: dict) -> AssemblyStatement:
        """replace constant subexpressions of `stmt` with literals"""

        def replace(node):
            match node:
                case Variable() | Call():
                    const = self.constant(node, state)
                    if const is not None and _same_type(node, Literal(const.val)):
                        return Literal(const.val)
            # never rewrite buffer operands
            match node:
                case Call(op, args):
                    new_args = tuple(replace(arg) for arg in args)
                    # a call that didn't fold, e.g. a division by zero, keeps a
                    # variable operand, so no backend compiler folds it its own way
                    if all(isinstance(arg, Literal) for arg in new_args):
                        return node
                    if any(
                        new is not old for new, old in zip(new_args, args, strict=True)
                    ):
                        return Call(op, new_args)
                case Load(buf, idx):
                    new_idx = replace(idx)
                    if new_idx is not idx:
                        return Load(buf, new_idx)
            return node

        match stmt:
            case Assign(lhs, rhs):
                return Assign(lhs, replace(rhs))
            case Store(buf, idx, val):
                return Store(buf, replace(idx), replace(val))
            case Resize(buf, size):
                return Resize(buf, replace(size))
            case SetAttr(obj, attr, val):
                return SetAttr(obj, attr, replace(val))
            case Return(arg):
                return Return(replace(arg))
            case Assert(exp):
                return Assert(replace(exp))
        return stmt

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        return [
            (name, repr(state[name].val))
            for name in variables_read(stmt)
            if name in state
        ]


def _same_type(node: AssemblyExpression, lit: Literal) -> bool:
    try:
        return node.result_type == lit.result_type
    except (AssertionError, AttributeError, KeyError, TypeError):
        return False
