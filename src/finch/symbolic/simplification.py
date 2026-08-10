"""
Basic algebraic simplification.

This module provides an IR-agnostic algebraic simplification pass.
It applies to any Finch IR whose call and literal nodes implement the
`CallTerm` and `LiteralTerm` interfaces, which today means
FinchLogic (`MapJoin`), FinchNotation (`Call`), and FinchAssembly (`Call`).

No rule mentions a particular operator. Rules are triggered by algebraic properties
`is_associative`, `is_commutative`, `is_idempotent`, `is_identity`,
`is_annihilator`, and `arity`. Rules which need to know about specific operators,
or about IR nodes other than calls, belong in that IR's own simplification pass.
Rules are applied until fixpoint.
"""

import math
from collections.abc import Sequence
from typing import TypeVar

from finch.algebra import (
    arity,
    is_annihilator,
    is_associative,
    is_commutative,
    is_idempotent,
    is_identity,
)
from finch.algebra.ftypes import FDTypeBoolean, FDTypeInteger

from .rewriters import Chain, Fixpoint, PostWalk, Rewrite, RwCallable
from .term import CallTerm, LiteralTerm, Term

T = TypeVar("T", bound=Term)


def _call_like(node: CallTerm, args: Sequence[Term]) -> Term:
    """Rebuild `node` with the same operator but new arguments."""
    return node.make_term(node.head(), node.op, *args)


def _same_op_leaves(node: CallTerm) -> list[Term]:
    """Flatten the maximal subtree of same-operator calls into its leaves."""
    leaves: list[Term] = []
    for arg in node.args:
        match arg:
            case CallTerm(op=inner_op) if inner_op == node.op:
                leaves.extend(_same_op_leaves(arg))
            case _:
                leaves.append(arg)
    return leaves


def _nest(node: CallTerm, args: Sequence[Term]) -> Term:
    """Rebuild `args` as a left-nested chain of two-argument calls."""
    result = args[0]
    for arg in args[1:]:
        result = _call_like(node, [result, arg])
    return result


def _run_of_literals(args: Sequence[Term]) -> tuple[int, int]:
    """Return the bounds of the longest run of adjacent literal arguments."""
    best_start, best_stop = 0, 0
    start = 0
    for i, arg in enumerate(args):
        if not isinstance(arg, LiteralTerm):
            start = i + 1
        elif i + 1 - start > best_stop - best_start:
            best_start, best_stop = start, i + 1
    return best_start, best_stop


def _evaluate(op: LiteralTerm, args: Sequence[Term]) -> LiteralTerm:
    """Run `op` on literal `args` now."""
    vals = [arg.val for arg in args if isinstance(arg, LiteralTerm)]
    return op.make_term(op.head(), op.val(*vals))


def _lift_nested_literals(node: CallTerm) -> Term | None:
    """
    `f(x, f(k, y))` => `f(f(k, x), y)` for literal `k` and abelian `f`.
    """
    leaves = _same_op_leaves(node)
    if len(leaves) == len(node.args):
        return None  # nothing is nested, so the flat branch covers it
    literals = [leaf for leaf in leaves if isinstance(leaf, LiteralTerm)]
    if not literals:
        return None
    rest = [leaf for leaf in leaves if not isinstance(leaf, LiteralTerm)]
    rebuilt = _nest(node, [*literals, *rest])
    return rebuilt if rebuilt != node else None


def canonicalize_associative(node: Term) -> Term | None:
    """
    Bring a nest of same-operator calls into canonical form.

    - An n-ary `f` absorbs the whole nest into one wide call:
      `f(a..., f(b...), c...)` => `f(a..., b..., c...)`.
    - A fixed-arity `f` is re-associated instead, using
     `_lift_nested_literals`.
    """
    match node:
        case CallTerm(op=op, args=args):
            if not (is_associative(op.val) and is_commutative(op.val)):
                return None
            if not math.isinf(arity(op.val)):
                return _lift_nested_literals(node) if is_commutative(op.val) else None
            leaves = _same_op_leaves(node)
            leaves = [
                *(arg for arg in leaves if isinstance(arg, LiteralTerm)),
                *(arg for arg in leaves if not isinstance(arg, LiteralTerm)),
            ]
            return _call_like(node, leaves)
    return None


def dedup_idempotent(node: Term) -> Term | None:
    """`f(a..., x, b..., x, c...)` => `f(a..., x, b..., c...)` for idempotent `f`."""
    match node:
        case CallTerm(op=op, args=args) if (
            is_idempotent(op.val) and is_associative(op.val) and is_commutative(op.val)
        ):
            # Does not use set because arg may be unhashable.
            unique: list[Term] = []
            for arg in args:
                if arg not in unique:
                    unique.append(arg)
            if len(unique) != len(args):
                return _call_like(node, unique)
    return None


def fold_literals(node: Term) -> Term | None:
    """
    Evaluate literal arguments at compile time.

    Three branches, each asking more of the operator than the last:

    - `f(x, y)` => `literal(f(x, y))` when every argument is a literal. The
      call disappears, so this is sound for any operator.
    - `f(a..., x, b..., y, c...)` => `f(f(x, y), a..., b..., c...)` when `f` is
      commutative as well as associative.
    - `f(a..., x, y, b...)` => `f(a..., f(x, y), b...)` when `f` is only
      associative, so nothing is reordered and only adjacent literals fold.
    """
    match node:
        case CallTerm(op=op, args=args) if args:
            literals = [arg for arg in args if isinstance(arg, LiteralTerm)]
            if len(literals) == len(args):
                return _evaluate(op, args)
            if (
                not math.isinf(arity(op.val))
                or len(literals) < 2
                or not is_associative(op.val)
            ):
                return None
            if is_commutative(op.val):
                rest = [arg for arg in args if not isinstance(arg, LiteralTerm)]
                return _call_like(node, [_evaluate(op, literals), *rest])
            start, stop = _run_of_literals(args)
            if stop - start < 2:
                return None
            folded = _evaluate(op, args[start:stop])
            return _call_like(node, [*args[:start], folded, *args[stop:]])
    return None


def _can_annihilate(node: CallTerm) -> bool:
    """
    Whether an annihilator of `node`'s operator absorbs *every* operand.

    Over the integers it does: `n * 0` is `0` for every `n`. Over the floats it
    does not, because `nan * 0` and `inf * 0` are `nan`.
    """
    try:
        result_type = node.result_type  # type: ignore[attr-defined]
    except (AttributeError, AssertionError, NotImplementedError):
        return False
    return isinstance(result_type, FDTypeInteger | FDTypeBoolean)


def annihilate(node: Term) -> Term | None:
    """
    `f(a..., z, b...)` => `z` when `z` is an annihilator for `f`.

    Only where the annihilator absorbs unconditionally; see
    `_absorbs_unconditionally`.
    """
    match node:
        case CallTerm(op=op, args=args) if _can_annihilate(node):
            for arg in args:
                match arg:
                    case LiteralTerm(val=val) if is_annihilator(op.val, val):
                        return arg
    return None


def drop_identities(node: Term) -> Term | None:
    """
    `f(a..., e, b...)` => `f(a..., b...)` when `e` is an identity for `f`.
    """
    match node:
        case CallTerm(op=op, args=args) if len(args) > 1 and is_associative(op.val):
            kept = [
                arg
                for arg in args
                if not (isinstance(arg, LiteralTerm) and is_identity(op.val, arg.val))
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
    return [
        fold_literals,
        annihilate,
        canonicalize_associative,
        dedup_idempotent,
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
    return Rewrite(Fixpoint(PostWalk(Chain(rules))))(root)
