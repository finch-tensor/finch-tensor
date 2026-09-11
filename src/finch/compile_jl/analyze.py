import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple
from finch.symbolic import PostWalk, Rewrite


def find_reset_arg_positions(func: ntn.Function) -> frozenset[int]:
    """Find arguments initialized on every path before their contents are read."""

    def slot_aliases(arg: ntn.Variable) -> set[str]:
        aliases = {arg.name}

        def rule(node):
            match node:
                case ntn.Unpack(ntn.Slot(name, _), ntn.Variable(rhs_name, _)):
                    if rhs_name in aliases:
                        aliases.add(name)
            return

        Rewrite(PostWalk(rule))(func.body)
        return aliases

    def references(node, aliases: set[str]) -> bool:
        found = False

        def rule(inner):
            nonlocal found
            if isinstance(inner, (ntn.Variable, ntn.Slot)) and inner.name in aliases:
                found = True
            return

        Rewrite(PostWalk(rule))(node)
        return found

    def states(node, aliases: set[str], state: str) -> set[str]:
        if state != "unseen":
            return {state}
        match node:
            case ntn.Block(bodies):
                result = {state}
                for body in bodies:
                    result = {
                        next_state
                        for prior_state in result
                        for next_state in states(body, aliases, prior_state)
                    }
                return result
            case ntn.If(cond, body):
                branch_state = "read" if references(cond, aliases) else state
                return states(body, aliases, branch_state) | {branch_state}
            case ntn.IfElse(cond, then_body, else_body):
                branch_state = "read" if references(cond, aliases) else state
                return states(then_body, aliases, branch_state) | states(
                    else_body, aliases, branch_state
                )
            case ntn.Loop(_, extent, body):
                loop_state = "read" if references(extent, aliases) else state
                return states(body, aliases, loop_state) | {loop_state}
            case ntn.Assign(lhs, ntn.Dimension()):
                return {"read" if references(lhs, aliases) else state}
            case ntn.Dimension():
                return {state}
            case ntn.Unpack(ntn.Slot(name, _), ntn.Variable(rhs_name, _)):
                if name in aliases and rhs_name in aliases:
                    return {state}
                return {"read" if references(node, aliases) else state}
            case ntn.Declare(tns, init, op, shape):
                if (
                    isinstance(tns, (ntn.Variable, ntn.Slot))
                    and tns.name in aliases
                    and not references(init, aliases)
                    and not references(op, aliases)
                    and not any(references(dim, aliases) for dim in shape)
                ):
                    return {"reset"}
                return {"read" if references(node, aliases) else state}
            case _:
                return {"read" if references(node, aliases) else state}

    return frozenset(
        position
        for position, arg in enumerate(func.args)
        if states(func.body, slot_aliases(arg), "unseen") == {"reset"}
    )


def find_return_arg_positions(func: ntn.Function) -> tuple[int, ...] | None:
    """Find a fixed return layout that aliases function arguments."""

    arg_positions = {arg.name: position for position, arg in enumerate(func.args)}
    layouts = []

    def rule(node):
        match node:
            case ntn.Return(ntn.Call(ntn.Literal(op), args)) if op == make_tuple:
                values = args
            case ntn.Return(ntn.Variable() as value):
                values = (value,)
            case ntn.Return():
                layouts.append(None)
                return
            case _:
                return
        if all(isinstance(value, ntn.Variable) for value in values):
            layouts.append(tuple(arg_positions.get(value.name) for value in values))
        else:
            layouts.append(None)

    Rewrite(PostWalk(rule))(func.body)
    if (
        len(layouts) != 1
        or layouts[0] is None
        or any(position is None for position in layouts[0])
    ):
        return None
    return tuple(layouts[0])
