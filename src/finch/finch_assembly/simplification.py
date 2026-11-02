import numpy as np

from finch import finch_assembly as asm
from finch.algebra import (
    FinchOperatorFType,
    FType,
    ffuncs,
    is_annihilator,
    is_identity,
)
from finch.algebra.ftypes import FDTypeBoolean, FDTypeInteger
from finch.symbolic import UnvalidatedForm, simplify_rules
from finch.symbolic.rewriters import Chain, Fixpoint, PostWalk, Rewrite
from finch.symbolic.simplification import annihilate

from .dataflow import has_effects
from .nodes import AssemblyExpression
from .stages import AssemblyTransform


class AssemblySimplify(UnvalidatedForm, AssemblyTransform):
    def lower(self, term: asm.Module) -> asm.Module:
        rules = [*map(_exact, simplify_rules()), self.simplify]
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
                return asm.If(asm.Literal(np.True_), body)
            case asm.Block(bodies):
                for i, b in enumerate(bodies):
                    match b:
                        # block(..., block(), ...) => block(...)
                        case asm.Block(()):
                            return asm.Block((*bodies[:i], *bodies[i + 1 :]))
        return None


def _exact(rule):
    # the generic rules ignore types and fold `0 * x` to `0`, which a nan or
    # inf `x` breaks, so a rewrite must keep the type and only integers and
    # bools are annihilated
    def checked(node):
        result = rule(node)
        match node:
            case asm.Call() if result is not None and (
                _result_type(result) != _result_type(node)
                or (rule is annihilate and not _never_nan(node))
            ):
                return None
        return result

    return checked


def _truth(val) -> bool | None:
    match val:
        case bool() | int() | np.bool_() | np.integer():
            return bool(val)
    return None


def _result_type(node: AssemblyExpression) -> FType | None:
    try:
        return node.result_type
    except (AssertionError, AttributeError, KeyError, TypeError):
        return None


def _never_nan(node: AssemblyExpression) -> bool:
    match _result_type(node):
        case FDTypeInteger() | FDTypeBoolean():
            return True
    return False
