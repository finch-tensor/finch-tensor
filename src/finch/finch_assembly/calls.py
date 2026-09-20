from finch.algebra import SingletonOperatorFType, ffuncs, is_dynamic

from . import nodes as asm


def lower_callable(op: asm.AssemblyExpression, args):
    match op.result_type:
        case SingletonOperatorFType(operator):
            return asm.Call(asm.Literal(operator), args)
        case ffuncs._InitWriteFType(fill=fill):
            x, y = args
            value = (
                asm.GetAttr(op, asm.Literal("value"))
                if is_dynamic(fill)
                else asm.Literal(fill.value)
            )
            return asm.Call(
                asm.Literal(ffuncs.where),
                (asm.Call(asm.Literal(ffuncs.eq), (y, value)), x, y),
            )
        case _:
            raise NotImplementedError(f"Cannot lower a call through {op.result_type}")
