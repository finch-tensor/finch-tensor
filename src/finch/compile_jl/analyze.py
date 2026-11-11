"""Facts about Julia kernels used to manage reusable result buffers."""

from __future__ import annotations

from functools import partial
from typing import Any

import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple
from finch.symbolic import PostWalk, Rewrite, TermTree
from finch.tensor.patterns import PatternTensorFType


def _alias_rule(node: Any, names: set[str]) -> None:
    match node:
        case ntn.Unpack(ntn.Slot(name, _), ntn.Variable(source, _)) if source in names:
            names.add(name)


def _argument_aliases(function: ntn.Function, argument: ntn.Variable) -> set[str]:
    names = {argument.name}
    Rewrite(PostWalk(partial(_alias_rule, names=names)))(function.body)
    return names


def _references(node: Any, names: set[str]) -> bool:
    if isinstance(node, (ntn.Variable, ntn.Slot)):
        return node.name in names
    return isinstance(node, TermTree) and any(
        _references(child, names) for child in node.children
    )


def _is_reset_before_read(node: Any, names: set[str]) -> bool | None:
    match node:
        case ntn.Block(bodies):
            for body in bodies:
                result = _is_reset_before_read(body, names)
                if result is not None:
                    return result
            return None
        case ntn.If(condition, body):
            if _references(condition, names):
                return False
            # When the condition is false, the previous values remain.
            return False if _is_reset_before_read(body, names) is False else None
        case ntn.IfElse(condition, then_body, else_body):
            if _references(condition, names):
                return False
            then_result = _is_reset_before_read(then_body, names)
            else_result = _is_reset_before_read(else_body, names)
            return (
                True
                if then_result is True and else_result is True
                else (False if False in (then_result, else_result) else None)
            )
        case ntn.Loop(_, extent, body):
            if _references(extent, names):
                return False
            return False if _is_reset_before_read(body, names) is False else None
        case ntn.Assign(_, ntn.Dimension()) | ntn.Unpack():
            return None
        case ntn.Declare(tensor, _, _, _) if isinstance(
            tensor.result_type, PatternTensorFType
        ):
            # A pattern tensor has no storage to reset.
            return None
        case ntn.Declare(tensor, _, _, _) if isinstance(
            tensor, (ntn.Variable, ntn.Slot)
        ):
            return True if tensor.name in names else None
        case _:
            return False if _references(node, names) else None


def reset_argument_positions(function: ntn.Function) -> frozenset[int]:
    """Return arguments reset before the kernel reads their previous values."""
    return frozenset(
        position
        for position, argument in enumerate(function.args)
        if _is_reset_before_read(function.body, _argument_aliases(function, argument))
    )


def _return_rule(node: Any, values: list[tuple[Any, ...] | None]) -> None:
    match node:
        case ntn.Return(ntn.Call(ntn.Literal(operator), arguments)) if (
            operator == make_tuple
        ):
            values.append(arguments)
        case ntn.Return(ntn.Variable() as value):
            values.append((value,))
        case ntn.Return():
            values.append(None)


def returned_argument_positions(function: ntn.Function) -> tuple[int, ...]:
    """Return the formal argument positions returned by a Julia kernel."""
    positions = {
        argument.name: position for position, argument in enumerate(function.args)
    }
    returned: list[tuple[Any, ...] | None] = []
    Rewrite(PostWalk(partial(_return_rule, values=returned)))(function.body)
    if len(returned) != 1 or returned[0] is None:
        raise ValueError("Julia kernels must contain one concrete return")
    values = returned[0]
    if not all(
        isinstance(value, ntn.Variable) and value.name in positions for value in values
    ):
        raise ValueError("Julia kernels must return formal arguments")
    return tuple(positions[value.name] for value in values)


def _read_before_reset(node: Any, names: set[str], reset: bool) -> tuple[bool, bool]:
    """
    Whether running node may read a tensor of names before it is reset, given
    whether it is reset beforehand, and whether it is reset afterwards. A loop
    or if statement may skip its body, so it leaves the tensor as it found it.
    """
    match node:
        case ntn.Block(bodies):
            for body in bodies:
                read, reset = _read_before_reset(body, names, reset)
                if read:
                    return True, reset
            return False, reset
        case ntn.Loop(_, extent, body):
            if not reset and _references(extent, names):
                return True, reset
            read, _ = _read_before_reset(body, names, reset)
            return read, reset
        case ntn.If(condition, body):
            if not reset and _references(condition, names):
                return True, reset
            read, _ = _read_before_reset(body, names, reset)
            return read, reset
        case ntn.IfElse(condition, then_body, else_body):
            if not reset and _references(condition, names):
                return True, reset
            then_read, then_reset = _read_before_reset(then_body, names, reset)
            else_read, else_reset = _read_before_reset(else_body, names, reset)
            return then_read or else_read, then_reset and else_reset
        case ntn.Declare(tensor, init, _, _) if (
            isinstance(tensor, (ntn.Variable, ntn.Slot)) and tensor.name in names
        ):
            return not reset and _references(init, names), True
        case ntn.Unpack() | ntn.Repack() | ntn.Freeze() | ntn.Thaw():
            # These move a tensor between modes, but don't read its values.
            return False, reset
        case _:
            return not reset and _references(node, names), reset


def local_argument_positions(function: ntn.Function) -> tuple[int, ...]:
    """
    Return the arguments whose values neither enter nor leave the kernel: it
    doesn't return them, and resets them before every read. A kernel may keep
    these in storage of its own rather than the arguments it is passed.
    """
    returned = set(returned_argument_positions(function))
    return tuple(
        position
        for position, argument in enumerate(function.args)
        if position not in returned
        and not _read_before_reset(
            function.body, _argument_aliases(function, argument), False
        )[0]
    )
