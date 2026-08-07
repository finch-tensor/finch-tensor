"""
Basic algebraic simplification.

This module provides an IR-agnostic constant folding and algebraic
simplification pass, in the spirit of `simplify.jl` in Finch.jl. It applies to
any Finch IR whose call and literal nodes implement the `CallTerm` and
`LiteralTerm` interfaces, which today means FinchLogic (`MapJoin`),
FinchNotation (`Call`), and FinchAssembly (`Call`).

No rule mentions a particular operator. Every rule is driven by the algebraic
properties the operator declares about itself (`is_associative`,
`is_commutative`, `is_idempotent`, `is_identity`, `is_annihilator`), so an
operator earns these simplifications by describing its own algebra. Rules which
need to know about specific operators, or about IR nodes other than calls,
belong in that IR's own simplification pass.

Following `simplify.jl`, the pass is a flat list of small rewrite rules which
are `Chain`ed together, driven to a `Fixpoint` at each node, and applied
bottom-up over the whole term by a `PostWalk`. Each rule takes a term and
returns either a rewritten term or `None` if it does not apply.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

from finch.algebra import (
    is_annihilator,
    is_associative,
    is_commutative,
    is_idempotent,
    is_identity,
)

from .rewriters import Chain, Fixpoint, PostWalk, Rewrite, RwCallable
from .term import CallTerm, LiteralTerm, Term

T = TypeVar("T", bound=Term)


def _literal_like(model: LiteralTerm, val: Any) -> LiteralTerm:
    """Build a literal holding `val` in the same IR as `model`."""
    return model.make_term(model.head(), val)


def _call_like(node: CallTerm, args: Sequence[Term]) -> Term:
    """Rebuild `node` with the same operator but new arguments."""
    return node.make_term(node.head(), node.op, *args)


def _value_holds(pred: Any, op: Any, val: Any) -> bool:
    """
    Ask an operator whether `val` is an identity/annihilator for it.

    These predicates are written against numbers, but a literal may hold
    anything an IR cares to store in one, such as a tensor, a buffer, or a
    type. A value the operator cannot compare against is simply not an
    identity or an annihilator, so an unanswerable question is a `False`.
    """
    try:
        return bool(pred(op, val))
    except (TypeError, ValueError, AttributeError):
        return False


def fold_constants(node: Term) -> Term | None:
    """`f(a...)` => `literal(f(a...))` when every argument is a literal."""
    match node:
        case CallTerm(op=op, args=args) if args:
            vals = [arg.val for arg in args if isinstance(arg, LiteralTerm)]
            if len(vals) != len(args):
                return None
            try:
                return _literal_like(op, op.val(*vals))
            except Exception:  # noqa: BLE001
                # `op.val` is arbitrary user code being run at compile time on
                # whatever the literals happen to hold. If it cannot handle
                # these values, that is a term we decline to fold, not an error
                # in the program being compiled.
                return None
    return None


def flatten_associative(node: Term) -> Term | None:
    """`f(a..., f(b...), c...)` => `f(a..., b..., c...)` for associative `f`."""
    match node:
        case CallTerm(op=op, args=args) if is_associative(op.val):
            for i, arg in enumerate(args):
                match arg:
                    case CallTerm(op=inner_op, args=inner_args) if inner_op == op:
                        return _call_like(
                            node, [*args[:i], *inner_args, *args[i + 1 :]]
                        )
    return None


def hoist_literals(node: Term) -> Term | None:
    """
    `f(a..., x, b...)` => `f(a..., b..., x)` for literal `x` and abelian `f`.

    Finch.jl canonicalizes the arguments of a commutative call by sorting them
    on a static hash. We only float the literals to the end, which is what
    `fold_literals` needs in order to reach non-adjacent literals, and which
    leaves the order of the remaining arguments -- and so the order of the
    loads and calls in the generated code -- alone.
    """
    match node:
        case CallTerm(op=op, args=args) if is_associative(op.val) and is_commutative(
            op.val
        ):
            hoisted = [
                *(arg for arg in args if not isinstance(arg, LiteralTerm)),
                *(arg for arg in args if isinstance(arg, LiteralTerm)),
            ]
            if hoisted != list(args):
                return _call_like(node, hoisted)
    return None


def dedup_idempotent(node: Term) -> Term | None:
    """`f(a..., x, b..., x, c...)` => `f(a..., x, b..., c...)` for idempotent `f`."""
    match node:
        case CallTerm(op=op, args=args) if (
            is_idempotent(op.val) and is_associative(op.val) and is_commutative(op.val)
        ):
            # Compared by equality rather than hashed into a set, since a
            # literal is free to hold an unhashable value.
            unique: list[Term] = []
            for arg in args:
                if arg not in unique:
                    unique.append(arg)
            if len(unique) != len(args):
                return _call_like(node, unique)
    return None


def fold_literals(node: Term) -> Term | None:
    """`f(a..., x, y, b...)` => `f(a..., f(x, y), b...)` for associative `f`."""
    match node:
        case CallTerm(op=op, args=args) if is_associative(op.val):
            for i, (x, y) in enumerate(zip(args, args[1:], strict=False)):
                if not (isinstance(x, LiteralTerm) and isinstance(y, LiteralTerm)):
                    continue
                try:
                    folded = _literal_like(op, op.val(x.val, y.val))
                except Exception:  # noqa: BLE001
                    continue  # See `fold_constants`.
                return _call_like(node, [*args[:i], folded, *args[i + 2 :]])
    return None


def annihilate(node: Term) -> Term | None:
    """`f(a..., z, b...)` => `z` when `z` is an annihilator for `f`."""
    match node:
        case CallTerm(op=op, args=args):
            for arg in args:
                match arg:
                    case LiteralTerm(val=val) if _value_holds(
                        is_annihilator, op.val, val
                    ):
                        return arg
    return None


def drop_identities(node: Term) -> Term | None:
    """
    `f(a..., e, b...)` => `f(a..., b...)` when `e` is an identity for `f`.

    Restricted to associative `f`, because a one-sided identity is not safe to
    drop from either side: `truediv` reports `1` as an identity, but only as a
    divisor, and `truediv(1, x)` is not `x`.
    """
    match node:
        case CallTerm(op=op, args=args) if len(args) > 1 and is_associative(op.val):
            kept = [
                arg
                for arg in args
                if not (
                    isinstance(arg, LiteralTerm)
                    and _value_holds(is_identity, op.val, arg.val)
                )
            ]
            if len(kept) == len(args):
                return None
            # Every argument was an identity, so the call is worth exactly one
            # of them.
            return _call_like(node, kept or [args[-1]])
    return None


def unwrap_singleton(node: Term) -> Term | None:
    """`f(x)` => `x`, the one-argument reduction of an associative `f`."""
    match node:
        case CallTerm(op=op, args=(arg,)) if is_associative(op.val):
            return arg
    return None


def simplify_rules() -> list[RwCallable]:
    """
    The default rules, in the order they are tried at each node.

    Ordered cheapest and most reductive first: rules which collapse a call
    outright run before rules which only rearrange its arguments to expose work
    for the others.
    """
    return [
        fold_constants,
        annihilate,
        flatten_associative,
        hoist_literals,
        dedup_idempotent,
        fold_literals,
        drop_identities,
        unwrap_singleton,
    ]


def simplify(root: T, rules: Sequence[RwCallable] | None = None) -> T:
    """
    Algebraically simplify `root` and every term below it.

    Args:
        root: The term to simplify.
        rules: The rewrite rules to apply, defaulting to `simplify_rules()`.
            Pass an extended list to add IR-specific rules to the algebraic
            ones.

    Returns:
        The simplified term, or `root` itself if no rule applied anywhere.
    """
    if rules is None:
        rules = simplify_rules()
    return Rewrite(PostWalk(Fixpoint(Chain(rules))))(root)
