"""
Lift `if` guards out of loops whose update is a no-op for one value of a
boolean subexpression.

For a loop body `A[...] op= f(...)`, if setting some boolean subexpression `c`
of `f(...)` to `b` folds `f(...)` to a literal that is an identity of `op`, the
update does nothing whenever `c == b`. The update is then wrapped in
`if c != b`, and the guard is hoisted out of every enclosing loop whose index
it does not use.
"""

import numpy as np

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra import is_identity
from finch.algebra.fill import AbstractFill
from finch.algebra.ftypes import ftype
from finch.symbolic import Chain, Fixpoint, PostOrderDFS, PostWalk, Rewrite
from finch.symbolic.simplification import (
    annihilate,
    drop_identities,
    fold_literals,
)

_BOOL_TYPES = (ftype(True), ftype(np.True_))


def _fold_where(node):
    """`where(true, a, b)` => `a` and `where(false, a, b)` => `b`."""
    match node:
        case ntn.Call(ntn.Literal(op), (ntn.Literal(cond), a, b)) if (
            op is ffuncs.where
        ):
            return a if cond else b
    return None


def _fold_static_literals(node):
    """Fold literal calls, leaving fill-wrapped literals alone."""
    match node:
        case ntn.Call(_, args) if any(
            isinstance(arg, ntn.Literal) and isinstance(arg.val, AbstractFill)
            for arg in args
        ):
            return None
    return fold_literals(node)


_simplify = Rewrite(
    Fixpoint(PostWalk(Chain([_fold_where, annihilate, drop_identities, _fold_static_literals])))
)


def _substitute(node, target, val):
    """Replace every occurrence of `target` in `node` with the literal `val`."""
    return Rewrite(PostWalk(lambda x: ntn.Literal(val) if x == target else None))(
        node
    )


def _find_guard(op, rhs):
    """
    Find a condition under which `lhs op= rhs` must run, and the `rhs` to use
    under it, or return None.
    """
    candidates = []
    for node in PostOrderDFS(rhs):
        match node:
            case ntn.Unwrap(ntn.Access(tns, ntn.Read(), _)) if (
                tns.result_type.element_type in _BOOL_TYPES
                and node not in candidates
            ):
                candidates.append(node)
    for cond in candidates:
        for skip in (False, True):
            folded = _simplify(_substitute(rhs, cond, skip))
            if isinstance(folded, ntn.Literal) and is_identity(op, folded.val):
                guard = (
                    cond
                    if not skip
                    else ntn.Call(ntn.Literal(ffuncs.logical_not), (cond,))
                )
                return guard, _simplify(_substitute(rhs, cond, not skip))
    return None


def _written_tensors(node):
    return {
        x.lhs.tns for x in PostOrderDFS(node) if isinstance(x, ntn.Increment)
    }


def _guard_update(node):
    """`loop(i, A op= rhs)` => `loop(i, if guard (A op= rhs'))`."""
    match node:
        case ntn.Loop(
            idx,
            ext,
            ntn.Increment(ntn.Access(tns, ntn.Update(op), _) as lhs, ntn.Call() as rhs),
        ):
            found = _find_guard(op.result_type, rhs)
            if found is None:
                return None
            guard, rhs_2 = found
            if tns in PostOrderDFS(guard):
                return None
            return ntn.Loop(idx, ext, ntn.If(guard, ntn.Increment(lhs, rhs_2)))
    return None


def _lift_guard(node):
    """`loop(i, if c body)` => `if c loop(i, body)` when `c` does not use `i`."""
    match node:
        case ntn.Loop(idx, ext, ntn.If(cond, body)) if (
            idx not in PostOrderDFS(cond)
            and _written_tensors(body).isdisjoint(PostOrderDFS(cond))
        ):
            return ntn.If(cond, ntn.Loop(idx, ext, body))
    return None


def lift_guards(func: ntn.Function) -> ntn.Function:
    """Guard no-op updates in `func` and hoist the guards out of loops."""
    return Rewrite(Fixpoint(PostWalk(Chain([_guard_update, _lift_guard]))))(func)
