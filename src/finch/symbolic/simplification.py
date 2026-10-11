"""
Basic algebraic simplification.

This module is a standard library of IR-agnostic rewrite rules. They apply to
any Finch IR whose call and literal nodes implement the `CallTerm` and
`LiteralTerm` interfaces, which today means FinchLogic (`MapJoin`),
FinchNotation (`Call`), and FinchAssembly (`Call`).
"""

import math
from collections.abc import Sequence

from finch.algebra import (
    CallableFType,
    FinchOperatorFType,
    arity,
    ftype,
    is_annihilator,
    is_associative,
    is_commutative,
    is_idempotent,
    is_identity,
    return_type,
)

from .rewriters import RwCallable
from .term import CallTerm, ExpressionTerm, LiteralTerm, Term


def _call_like(node: CallTerm, args: Sequence[Term]) -> Term:
    """Rebuild `node` with the same operator but new arguments."""
    return node.make_term(node.head(), node.op, *args)


def _evaluate(op: LiteralTerm, args: Sequence[Term]) -> LiteralTerm:
    """
    Run `op` on literal `args`. Apply `return_type` to ensure that we
    don't alter the type of the output.
    """
    vals = [arg.val for arg in args if isinstance(arg, LiteralTerm)]
    op_type = ftype(op.val)
    assert isinstance(op_type, CallableFType)
    result = return_type(op_type, *(ftype(val) for val in vals))
    return op.make_term(op.head(), result(op.val(*vals)))


def canonicalize_associative(node: Term) -> Term | None:
    """
    Unwraps singleton n-ary function calls.
    - `f(x)` => `x`

    An n-ary `f` absorbs its immediate `f`-children and moves literals to the
    front, next to each other:
    - `f(a..., f(b...), c...)` => `f(k..., rest...)`.

    A fixed-arity `f` is instead rotated one step at a time so that literals
    bubble up and to the left, where adjacent ones merge:
    - `f(k1, f(k2, y))` => `f(f(k1, k2), y)`
    - `f(f(k, x), y)`   => `f(k, f(x, y))`
    - `f(x, f(k, y))`   => `f(k, f(x, y))`
    - `f(x, k)`         => `f(k, x)`
    """
    match node:
        case CallTerm(
            op=ExpressionTerm(result_type=FinchOperatorFType() as op_type) as op,
            args=args,
        ) if is_associative(op_type):
            if len(args) == 1:
                return args[0]
            if math.isinf(arity(op_type)):
                flat = [
                    leaf
                    for arg in args
                    for leaf in (
                        arg.args
                        if isinstance(arg, CallTerm) and arg.op == op
                        else (arg,)
                    )
                ]
                if is_commutative(op_type):
                    flat = sorted(
                        flat, key=lambda leaf: not isinstance(leaf, LiteralTerm)
                    )
                return _call_like(node, flat) if flat != list(args) else None
            match args:
                case (
                    LiteralTerm() as k1,
                    CallTerm(op=inner, args=(LiteralTerm() as k2, y)),
                ) if inner == op and isinstance(op, LiteralTerm):
                    return _call_like(node, [_evaluate(op, (k1, k2)), y])
                case (
                    CallTerm(op=inner, args=(LiteralTerm() as k, x)),
                    y,
                ) if inner == op:
                    return _call_like(node, [k, _call_like(node, [x, y])])
                case (
                    x,
                    CallTerm(op=inner, args=(LiteralTerm() as k, y)),
                ) if inner == op and is_commutative(op_type):
                    return _call_like(node, [k, _call_like(node, [x, y])])
                case (x, LiteralTerm() as k) if not isinstance(
                    x, LiteralTerm
                ) and is_commutative(op_type):
                    return _call_like(node, [k, x])
    return None


def dedup_idempotent(node: Term) -> Term | None:
    """`f(a..., x, b..., x, c...)` => `f(a..., x, b..., c...)` for idempotent `f`."""
    match node:
        case CallTerm(
            op=ExpressionTerm(result_type=FinchOperatorFType() as op_type), args=args
        ) if (
            is_idempotent(op_type)
            and is_associative(op_type)
            and is_commutative(op_type)
        ):
            unique = [arg for i, arg in enumerate(args) if arg not in args[:i]]
            if len(unique) != len(args):
                return unique[0] if len(unique) == 1 else _call_like(node, unique)
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
        case CallTerm(op=LiteralTerm() as op, args=args) if args:
            if all(isinstance(arg, LiteralTerm) for arg in args):
                return _evaluate(op, args)
            op_type = op.result_type
            if not (
                isinstance(op_type, FinchOperatorFType)
                and math.isinf(arity(op_type))
                and is_associative(op_type)
            ):
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
        case CallTerm(
            op=ExpressionTerm(result_type=FinchOperatorFType() as op_type), args=args
        ):
            return next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, LiteralTerm) and is_annihilator(op_type, arg.val)
                ),
                None,
            )
    return None


def drop_identities(node: Term) -> Term | None:
    """
    `f(a..., e, b...)` => `f(a..., b...)` when `e` is an identity for `f`.
    """
    match node:
        case CallTerm(
            op=ExpressionTerm(result_type=FinchOperatorFType() as op_type), args=args
        ) if is_associative(op_type):
            if len(args) == 2 and arity(op_type) == 2:
                if isinstance(args[0], LiteralTerm) and is_identity(
                    op_type, args[0].val
                ):
                    return args[1]
                if isinstance(args[1], LiteralTerm) and is_identity(
                    op_type, args[1].val
                ):
                    return args[0]
            if math.isinf(arity(op_type)):
                kept = [
                    arg
                    for arg in args
                    if not (
                        isinstance(arg, LiteralTerm) and is_identity(op_type, arg.val)
                    )
                ]
                if len(kept) == len(args):
                    return None
                return _call_like(node, kept or args[-1:])
    return None


def simplify_rules() -> list[RwCallable]:
    return [
        canonicalize_associative,
        annihilate,
        dedup_idempotent,
        fold_literals,
        drop_identities,
    ]
