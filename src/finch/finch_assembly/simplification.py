from finch import finch_assembly as asm
from finch.algebra import FinchOperatorFType, ffuncs, is_annihilator, is_identity
from finch.symbolic import UnvalidatedForm, simplify_rules
from finch.symbolic.rewriters import Chain, Fixpoint, PostWalk, Rewrite

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
            case asm.ForLoop(_, _, _, asm.Block(())):
                return asm.Block(())
            # if(...) {} is removed
            case asm.If(_, asm.Block(())):
                return asm.Block(())
            # if(x == x) { ... } => { ... }
            case asm.If(asm.Call(asm.Literal(ffuncs.eq), (arg1, arg2)), body) if (
                arg1 == arg2
            ):
                return body
            # block(..., block(), ...) => block(...)
            case asm.Block(bodies):
                for i, b in enumerate(bodies):
                    match b:
                        case asm.Block(()):
                            return asm.Block((*bodies[:i], *bodies[i + 1 :]))
        return None
