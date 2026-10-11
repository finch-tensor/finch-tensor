import numpy as np

from finch import finch_assembly as asm
from finch.algebra import FinchOperatorFType, ffuncs, is_annihilator, is_identity
from finch.algebra.ftypes import FDTypeBoolean, FDTypeInteger
from finch.symbolic import PostOrderDFS, UnvalidatedForm, simplify_rules
from finch.symbolic.rewriters import Chain, Fixpoint, PostWalk, Rewrite

from .dataflow import has_effects
from .nodes import AssemblyExpression, AssemblyStatement
from .stages import AssemblyTransform


class AssemblySimplify(UnvalidatedForm, AssemblyTransform):
    def lower(self, term: asm.Module) -> asm.Module:
        rules = [*simplify_rules(), self.simplify]
        return Rewrite(Fixpoint(PostWalk(Chain(rules))))(term)

    @classmethod
    def simplify(cls, term: asm.AssemblyNode):
        from finch.tensor.scalar import Scalar

        match term:
            case asm.Call(
                asm.AssemblyExpression(result_type=ffuncs._CastFType(dtype)), (arg,)
            ) if arg.result_type == dtype:
                return arg
            # overwrite(x, y) => y
            case asm.Call(op, (_, y)) if op.result_type == ffuncs.overwrite.ftype:
                return y
            # op(..., arg, ...) where arg is anihilator => arg
            case asm.Call(
                asm.AssemblyExpression(result_type=FinchOperatorFType() as op_type),
                args,
            ):
                for arg in args:
                    match arg:
                        case asm.Literal(val) if isinstance(
                            val, Scalar
                        ) and is_annihilator(op_type, val.val):
                            return arg
                return None
            # slot(a, idx) = op(slot(a, idx), arg) where RHS is:
            #   1. init_write(x)(slot(a, idx), x)
            #   2. op(slot(a, idx), arg) and arg is an identity for op
            # is removed
            case asm.Block(
                (
                    *_,
                    asm.Store(
                        asm.Slot(_) as s1,
                        idx1,
                        asm.Call(
                            asm.AssemblyExpression(
                                result_type=FinchOperatorFType() as op_type
                            ),
                            (asm.Load(asm.Slot(_) as s2, idx2), asm.Literal(arg)),
                        ),
                    ),
                ) as bodies
            ) if s1 == s2 and idx1 == idx2:
                arg_val = arg.val if isinstance(arg, Scalar) else arg
                if is_identity(op_type, arg_val):
                    return asm.Block(bodies[:-1])
            # loop(...) {} is removed
            case asm.ForLoop(_, start, end, asm.Block(())) if not (
                has_effects(start) or has_effects(end)
            ):
                return asm.Block(())
            # loops over an empty literal range are removed
            case asm.ForLoop(_, asm.Literal(start), asm.Literal(end), _) if (
                _truth(start) is not None and _truth(end) is not None and start >= end
            ):
                return asm.Block(())
            # while(false) { ... } is removed
            case asm.WhileLoop(asm.Literal(cond), _) if _truth(cond) is False:
                return asm.Block(())
            # if(...) {} and if(false) { ... } are removed
            case asm.If(cond, asm.Block(())) if not has_effects(cond):
                return asm.Block(())
            case asm.If(asm.Literal(cond), _) if _truth(cond) is False:
                return asm.Block(())
            # if(x == x) { ... } => if(true) { ... }, unless x can be nan
            case asm.If(asm.Call(asm.Literal(ffuncs.eq), (arg1, arg2)), body) if (
                arg1 == arg2 and _never_nan(arg1) and not has_effects(arg1)
            ):
                return asm.If(asm.Literal(True), body)
            # if(c) { a } else { b } with a literal c => if(true) { a or b }
            case asm.IfElse(asm.Literal(cond), body, else_body) if (
                _truth(cond) is not None
            ):
                return asm.If(asm.Literal(True), body if _truth(cond) else else_body)
            case asm.Block(bodies):
                for i, b in enumerate(bodies):
                    match b:
                        # block(..., block(), ...) => block(...)
                        case asm.Block(()):
                            return asm.Block((*bodies[:i], *bodies[i + 1 :]))
                        # block(..., if(true) { a }, ...) => block(..., a, ...)
                        case asm.If(asm.Literal(cond), body) if _truth(
                            cond
                        ) and _can_inline(body, bodies[:i], bodies[i + 1 :]):
                            match body:
                                case asm.Block(inner):
                                    pass
                                case _:
                                    inner = (body,)
                            return asm.Block((*bodies[:i], *inner, *bodies[i + 1 :]))
        return None


def _truth(val) -> bool | None:
    """truth value of a literal condition or bound, if it has one"""
    match val:
        case bool() | int() | np.bool_() | np.integer():
            return bool(val)
    return None


def _never_nan(node: AssemblyExpression) -> bool:
    try:
        type_ = node.result_type
    except (AssertionError, AttributeError, KeyError, TypeError):
        return False
    match type_:
        case FDTypeInteger() | FDTypeBoolean():
            return True
    return False


def _can_inline(body: AssemblyStatement, before, rest) -> bool:
    """
    the body of an if opens a scope, so it can only be inlined if no later
    statement touches the variables it declares, and it doesn't return or break
    before the rest.
    """
    declared = set()
    for stmt in before:
        match stmt:
            case asm.Assign(asm.Variable(name, _), _):
                declared.add(name)
    assigned = set()
    for node in PostOrderDFS(body):
        match node:
            case asm.Assign(asm.Variable(name, _), _) if name not in declared:
                assigned.add(name)
            case asm.Return() | asm.Break() if rest:
                return False
    for stmt in rest:
        for node in PostOrderDFS(stmt):
            match node:
                case asm.Variable(name, _) if name in assigned:
                    return False
    return True
