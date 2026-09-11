import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple
from finch.symbolic import PostWalk, Rewrite, TermTree


def find_reset_arg_positions(func: ntn.Function) -> frozenset[int]:
    """
    Return the positions of arguments which are initialized during this function
    invocation. These are the arguments for which buffers can be reused from a
    pool since they don't depend on a prior result.
    """

    def get_slot_aliases(arg: ntn.Variable) -> set[str]:
        aliases = {arg.name}

        def rule(node):
            match node:
                case ntn.Unpack(ntn.Slot(name, _), ntn.Variable(rhs_name, _)):
                    if rhs_name in aliases:
                        aliases.add(name)
            return

        Rewrite(PostWalk(rule))(func.body)
        return aliases

    def get_references(node, aliases: set[str]) -> bool:
        if isinstance(node, (ntn.Variable, ntn.Slot)):
            return node.name in aliases
        if isinstance(node, TermTree):
            return any(get_references(child, aliases) for child in node.children)
        return False

    def is_reset_arg(node, aliases: set[str]) -> bool | None:
        match node:
            case ntn.Block(bodies):
                for body in bodies:
                    result = is_reset_arg(body, aliases)
                    if result is not None:
                        return result
                return None
            case ntn.If(cond, body):
                if get_references(cond, aliases):
                    return False
                body_result = is_reset_arg(body, aliases)
                if body_result is False:
                    return False
                return None
            case ntn.IfElse(cond, then_body, else_body):
                if get_references(cond, aliases):
                    return False
                then_result = is_reset_arg(then_body, aliases)
                else_result = is_reset_arg(else_body, aliases)
                if then_result is False or else_result is False:
                    return False
                if then_result is None or else_result is None:
                    return None
                return True
            case ntn.Loop(_, extent, body):
                if get_references(extent, aliases):
                    return False
                body_result = is_reset_arg(body, aliases)
                if body_result is False:
                    return False
                return None
            case ntn.Assign(_, ntn.Dimension()):
                return None
            case ntn.Unpack():
                return None
            case ntn.Declare(tns, _, _, _):
                if isinstance(tns, (ntn.Variable, ntn.Slot)) and tns.name in aliases:
                    return True
                return None
            case _:
                if get_references(node, aliases):
                    return False
                return None

    return frozenset(
        position
        for position, arg in enumerate(func.args)
        if is_reset_arg(func.body, get_slot_aliases(arg))
    )


def find_return_arg_positions(func: ntn.Function) -> tuple[int, ...] | None:
    """Find the position of return arguments in the function header."""

    arg_positions = {arg.name: position for position, arg in enumerate(func.args)}
    return_values = []

    def rule(node):
        match node:
            case ntn.Return(ntn.Call(ntn.Literal(op), values)) if op == make_tuple:
                return_values.append(values)
            case ntn.Return(ntn.Variable() as value):
                return_values.append((value,))
            case ntn.Return():
                return_values.append(None)

    Rewrite(PostWalk(rule))(func.body)
    if len(return_values) != 1:
        return None
    values = return_values[0]
    if values is None or not all(
        isinstance(value, ntn.Variable) and value.name in arg_positions
        for value in values
    ):
        return None
    return tuple(arg_positions[value.name] for value in values)
