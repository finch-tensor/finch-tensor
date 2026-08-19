"""
Basic algebraic simplification.

This module is a standard library of IR-agnostic rewrite rules. They apply to
any Finch IR whose call and literal nodes implement the `CallTerm` and
`LiteralTerm` interfaces, which today means FinchLogic (`MapJoin`),
FinchNotation (`Call`), and FinchAssembly (`Call`).

No rule mentions a particular operator. Rules are triggered by algebraic
properties `is_associative`, `is_commutative`, `is_idempotent`, `is_identity`,
`is_annihilator`, and `arity`. Rules which need to know about specific
operators, or about IR nodes other than calls, belong in that IR's own
simplification pass.

Each rule takes one small step and leaves the rest to iteration: the IR passes
run these rules with `Rewrite(Fixpoint(PostWalk(Chain(...))))`, so a rule may
assume its node's children are already simplified and rewrite only the node at
hand.
"""

import math
from collections.abc import Sequence

from finch.algebra import (
    arity,
    is_annihilator,
    is_associative,
    is_commutative,
    is_idempotent,
    is_identity,
    return_type,
)

from .rewriters import RwCallable
from .term import CallTerm, LiteralTerm, Term


def _call_like(node: CallTerm, args: Sequence[Term]) -> Term:
    """Rebuild `node` with the same operator but new arguments."""
    return node.make_term(node.head(), node.op, *args)


def _evaluate(op: LiteralTerm, args: Sequence[Term]) -> LiteralTerm:
    """
    Run `op` on literal `args`. Apply `return_type` to ensure that we
    don't alter the type of the output.
    """
    vals = [arg.val for arg in args if isinstance(arg, LiteralTerm)]
    return op.make_term(op.head(), return_type(op.val, *vals)(op.val(*vals)))


def canonicalize_associative(node: Term) -> Term | None:
    """
    Take one step toward the canonical form of a nest of abelian calls.

    An n-ary `f` absorbs its immediate `f`-children and moves literals to the
    front, next to each other: `f(a..., f(b...), c...)` => `f(k..., rest...)`.

    A fixed-arity `f` is instead rotated one step at a time so that literals
    bubble up and to the left, where adjacent ones merge:

    - `f(x, k)`         => `f(k, x)`
    - `f(k1, f(k2, y))` => `f(f(k1, k2), y)`
    - `f(f(k, x), y)`   => `f(k, f(x, y))`
    - `f(x, f(k, y))`   => `f(k, f(x, y))`
    """
    match node:
        case CallTerm(op=op, args=args) if is_associative(op.val) and is_commutative(
            op.val
        ):
            if math.isinf(arity(op.val)):
                # Children were already flattened when they were visited, so
                # one level of splicing completes the job.
                flat = [
                    leaf
                    for arg in args
                    for leaf in (
                        arg.args
                        if isinstance(arg, CallTerm) and arg.op == op
                        else (arg,)
                    )
                ]
                flat = sorted(flat, key=lambda leaf: not isinstance(leaf, LiteralTerm))
                return _call_like(node, flat) if flat != list(args) else None
            match args:
                case (x, LiteralTerm() as k) if not isinstance(x, LiteralTerm):
                    return _call_like(node, [k, x])
                case (
                    LiteralTerm() as k1,
                    CallTerm(op=inner, args=(LiteralTerm() as k2, y)),
                ) if inner == op:
                    return _call_like(node, [_evaluate(op, (k1, k2)), y])
                case (CallTerm(op=inner, args=(LiteralTerm() as k, x)), y) if (
                    inner == op
                ):
                    return _call_like(node, [k, _call_like(node, [x, y])])
                case (x, CallTerm(op=inner, args=(LiteralTerm() as k, y))) if (
                    inner == op and not isinstance(x, LiteralTerm)
                ):
                    return _call_like(node, [k, _call_like(node, [x, y])])
    return None


def dedup_idempotent(node: Term) -> Term | None:
    """`f(a..., x, b..., x, c...)` => `f(a..., x, b..., c...)` for idempotent `f`."""
    match node:
        case CallTerm(op=op, args=args) if (
            is_idempotent(op.val) and is_associative(op.val) and is_commutative(op.val)
        ):
            # Does not use set because arg may be unhashable.
            unique = [arg for i, arg in enumerate(args) if arg not in args[:i]]
            if len(unique) != len(args):
                return _call_like(node, unique)
    return None


def fold_literals(node: Term) -> Term | None:
    """
    Evaluate literal arguments at compile time.

    - `f(x, y)` => `literal(f(x, y))` when every argument is a literal. The
      call disappears, so this is sound for any operator.
    - `f(a..., x, y, b...)` => `f(a..., f(x, y), b...)` folds adjacent
      literal pairs of an associative n-ary `f`.
    """
    match node:
        case CallTerm(op=op, args=args) if args:
            if all(isinstance(arg, LiteralTerm) for arg in args):
                return _evaluate(op, args)
            if not (math.isinf(arity(op.val)) and is_associative(op.val)):
                return None
            new_args = []
            running_literal = op.head()(None)
            on_run = False
            for x in args:
                if isinstance(x, LiteralTerm):
                    running_literal = (
                        x if not on_run else _evaluate(op, (running_literal, x))
                    )
                    on_run = True
                else:
                    if on_run:
                        new_args.append(running_literal)
                        on_run = False
                    new_args.append(x)
            if on_run:
                new_args.append(running_literal)
            if new_args != list(args):
                return _call_like(node, new_args)
    return None


def annihilate(node: Term) -> Term | None:
    """
    `f(a..., z, b...)` => `z` when `z` is an annihilator for `f`.

    Applied over every dtype, floats included, so `nan * 0` and `inf * 0` fold to
    `0` rather than to `nan`.
    TODO: add a safe mode for nan
    """
    match node:
        case CallTerm(op=op, args=args):
            return next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, LiteralTerm) and is_annihilator(op.val, arg.val)
                ),
                None,
            )
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
            return _call_like(node, kept or args[-1:])
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
