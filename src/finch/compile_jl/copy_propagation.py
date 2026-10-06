"""
Remove top-level tensor copies X[i...] = Y[i...].

If X is not returned and is read exactly once, that read becomes a read of Y,
provided Y is not modified between the copy and the read.

If X is returned and Y is a temporary of the kernel, Y is returned in place of
X, provided neither is touched after the copy. The caller binds returned
tensors by position, so X's storage need not hold the result.
"""

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite


def _copy_tensors(stmt):
    """Return (dst, src) if stmt is a loop nest copying src into dst."""
    idxs = []
    while isinstance(stmt, ntn.Loop):
        idxs.append(stmt.idx)
        stmt = stmt.body
    match stmt:
        case ntn.Increment(
            ntn.Access(dst, ntn.Update(op), dst_idxs),
            ntn.Unwrap(ntn.Access(src, ntn.Read(), src_idxs)),
        ) if (
            isinstance(op.result_type, ffuncs._InitWriteFType)
            and isinstance(dst, ntn.Slot)
            and isinstance(src, ntn.Slot)
            and dst != src
            and dst.type == src.type
            and dst_idxs == src_idxs
            and sorted(dst_idxs, key=str) == sorted(idxs, key=str)
        ):
            return dst, src
    return None


def _modifies(stmt, tns):
    for node in PostOrderDFS(stmt):
        match node:
            case ntn.Declare(t, _, _, _) | ntn.Thaw(t, _) if t == tns:
                return True
            case ntn.Increment(ntn.Access(t, _, _), _) if t == tns:
                return True
    return False


def _reads(stmt, tns):
    return sum(
        1
        for node in PostOrderDFS(stmt)
        if isinstance(node, ntn.Access)
        and isinstance(node.mode, ntn.Read)
        and node.tns == tns
    )


def _returned_slots(stmts):
    returned = set()
    for stmt in stmts:
        match stmt:
            case ntn.Return(ntn.Call(op, args)) if op.result_type == make_tuple.ftype:
                returned.update(args)
            case ntn.Return(val):
                returned.add(val)
    return {
        stmt.val
        for stmt in stmts
        if isinstance(stmt, ntn.Repack) and stmt.obj in returned
    }


def _propagate_one(stmts):
    """Propagate the first eligible copy in stmts, or return None."""
    returned = _returned_slots(stmts)
    for k, copy in enumerate(stmts):
        found = _copy_tensors(copy)
        if found is None:
            continue
        dst, src = found
        if dst in returned:
            continue
        others = stmts[:k] + stmts[k + 1 :]
        # The copy must be the only write of dst, besides its declaration.
        if any(
            _modifies(stmt, dst) and not isinstance(stmt, ntn.Declare)
            for stmt in others
        ):
            continue
        if sum(isinstance(s, ntn.Declare) and s.tns == dst for s in others) != 1:
            continue
        uses = [m for m, stmt in enumerate(stmts) if _reads(stmt, dst)]
        if len(uses) != 1 or _reads(stmts[uses[0]], dst) != 1:
            continue
        m = uses[0]
        if m < k or any(_modifies(stmt, src) for stmt in stmts[k + 1 : m + 1]):
            continue

        def swap(node, dst=dst, src=src):
            match node:
                case ntn.Access(t, ntn.Read(), idxs) if t == dst:
                    return ntn.Access(src, ntn.Read(), idxs)
            return None

        new_stmts = []
        for j, stmt in enumerate(stmts):
            if j == k:
                continue
            if (
                isinstance(stmt, ntn.Declare | ntn.Freeze | ntn.Thaw)
                and stmt.tns == dst
            ):
                continue
            new_stmts.append(Rewrite(PostWalk(swap))(stmt) if j == m else stmt)
        return new_stmts
    return None


def _touches(stmt, tns):
    return _modifies(stmt, tns) or _reads(stmt, tns) > 0


def _return_source_one(stmts):
    """Return the source of the first eligible returned copy, or None."""
    ret = next(
        (k for k, stmt in enumerate(stmts) if isinstance(stmt, ntn.Return)), None
    )
    if ret is None:
        return None
    match stmts[ret]:
        case ntn.Return(ntn.Call(op, args)) if op.result_type == make_tuple.ftype:
            returned = list(args)
        case _:
            return None
    objs = {stmt.lhs: stmt.rhs for stmt in stmts if isinstance(stmt, ntn.Unpack)}
    for k, copy in enumerate(stmts):
        found = _copy_tensors(copy)
        if found is None:
            continue
        dst, src = found
        dst_obj, src_obj = objs.get(dst), objs.get(src)
        if dst_obj not in returned or src_obj is None or src_obj in returned:
            continue
        if dst_obj.type_ != src_obj.type_:
            continue
        # src must be a temporary: declared before anything else touches it.
        first = next(stmt for stmt in stmts if _touches(stmt, src))
        if not (isinstance(first, ntn.Declare) and first.tns == src):
            continue
        after = stmts[k + 1 :]
        if any(_modifies(stmt, src) or _touches(stmt, dst) for stmt in after):
            continue
        # The copy's own declaration of dst is dropped along with it.
        d = max(j for j in range(k) if _touches(stmts[j], dst))
        if not (isinstance(stmts[d], ntn.Declare) and stmts[d].tns == dst):
            continue

        new_args = tuple(src_obj if arg == dst_obj else arg for arg in returned)
        new_stmts = []
        for j, stmt in enumerate(stmts):
            if j in (d, k) or (
                j > k and isinstance(stmt, ntn.Freeze) and stmt.tns == dst
            ):
                continue
            if j == ret:
                stmt = ntn.Return(ntn.Call(stmt.val.op, new_args))
            new_stmts.append(stmt)
        return new_stmts
    return None


def _flatten(stmt):
    if isinstance(stmt, ntn.Block):
        return [s for body in stmt.bodies for s in _flatten(body)]
    return [stmt]


def propagate_copies(func: ntn.Function) -> ntn.Function:
    """Remove the top-level copies of func that can be propagated."""
    stmts = _flatten(func.body)
    while (new_stmts := _propagate_one(stmts) or _return_source_one(stmts)) is not None:
        stmts = new_stmts
    return ntn.Function(func.name, func.args, ntn.Block(tuple(stmts)))
