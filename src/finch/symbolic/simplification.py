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
    is_variadic,
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
    return _literal_like(op, op.val(*vals))


def flatten_associative(node: Term) -> Term | None:
    """
    `f(a..., f(b...), c...)` => `f(a..., b..., c...)` for associative `f`.

    Also requires `f` to be variadic. Associativity permits the regrouping, but
    a binary-only operator like `logical_and` cannot be handed the wider
    argument list that regrouping produces.
    """
    match node:
        case CallTerm(op=op, args=args) if is_associative(op.val) and is_variadic(
            op.val
        ):
            for i, arg in enumerate(args):
                match arg:
                    case CallTerm(op=inner_op, args=inner_args) if inner_op == op:
                        return _call_like(
                            node, [*args[:i], *inner_args, *args[i + 1 :]]
                        )
    return None


def _lift_nested_literals(node: CallTerm) -> Term | None:
    """
    `f(f(x, k), y)` => `f(f(x, y), k)` for literal `k` and abelian `f`.

    A non-variadic operator cannot be flattened into one wide call, so a nest
    of two-argument calls is the only shape it has and its literals can sit at
    any depth, out of reach of every other rule. Re-associating the nest brings
    them to the top. Literals that meet there are combined on the way, because
    two of them can never share a two-argument call otherwise, and the nodes
    this builds below the top are not revisited by the surrounding `PostWalk`.
    """
    leaves = _same_op_leaves(node)
    if len(leaves) == len(node.args):
        return None  # nothing is nested, so the flat branch covers it
    literals = [leaf for leaf in leaves if isinstance(leaf, LiteralTerm)]
    if not literals:
        return None
    folded = literals[0]
    for other in literals[1:]:
        # Two at a time, which is an arity the operator certainly accepts.
        folded = _evaluate(node.op, [folded, other])
    rest = [leaf for leaf in leaves if not isinstance(leaf, LiteralTerm)]
    rebuilt = _nest(node, [*rest, folded])
    return rebuilt if rebuilt != node else None


def hoist_literals(node: Term) -> Term | None:
    """
    `f(a..., x, b...)` => `f(a..., b..., x)` for literal `x` and abelian `f`.

    Finch.jl canonicalizes the arguments of a commutative call by sorting them
    on a static hash. We only float the literals to the end, which is what
    `fold_literals` needs in order to reach non-adjacent literals, and which
    leaves the order of the remaining arguments -- and so the order of the
    loads and calls in the generated code -- alone.

    Where the operator is not variadic the literals may be nested rather than
    merely out of order, which `_lift_nested_literals` handles.
    """
    match node:
        case CallTerm(op=op, args=args) if is_associative(op.val) and is_commutative(
            op.val
        ):
            if not is_variadic(op.val):
                lifted = _lift_nested_literals(node)
                if lifted is not None:
                    return lifted
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
    """
    Evaluate literal arguments at compile time.

    Three branches, each asking more of the operator than the last:

    - `f(x, y)` => `literal(f(x, y))` when every argument is a literal. The
      call disappears, so this is sound for any operator.
    - `f(a..., x, b..., y, c...)` => `f(a..., b..., c..., f(x, y))` when `f` is
      commutative as well as associative, which lets the literals be gathered
      from wherever they sit. The folded value goes last, the same place
      `hoist_literals` would have put it.
    - `f(a..., x, y, b...)` => `f(a..., f(x, y), b...)` when `f` is only
      associative, so nothing may be reordered and only a run of adjacent
      literals can fold.
    """
    match node:
        case CallTerm(op=op, args=args) if args:
            literals = [arg for arg in args if isinstance(arg, LiteralTerm)]
            if len(literals) == len(args):
                return _evaluate(op, args)
            if not is_associative(op.val):
                return None
            if is_commutative(op.val) and len(literals) > 1:
                rest = [arg for arg in args if not isinstance(arg, LiteralTerm)]
                return _call_like(node, [*rest, _evaluate(op, literals)])
            start, stop = _run_of_literals(args)
            if stop - start < 2:
                return None
            folded = _evaluate(op, args[start:stop])
            return _call_like(node, [*args[:start], folded, *args[stop:]])
    return None


def annihilate(node: Term) -> Term | None:
    """`f(a..., z, b...)` => `z` when `z` is an annihilator for `f`."""
    match node:
        case CallTerm(op=op, args=args):
            for arg in args:
                match arg:
                    case LiteralTerm(val=val) if is_annihilator(op.val, val):
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
    """
    The default rules, in the order they are tried at each node.

    Ordered cheapest and most reductive first: rules which collapse a call
    outright run before rules which only rearrange its arguments to expose work
    for the others.
    """
    return [
        fold_literals,
        annihilate,
        flatten_associative,
        hoist_literals,
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
    return Rewrite(PostWalk(Fixpoint(Chain(rules))))(root)
