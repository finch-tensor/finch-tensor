"""
Canonical forms of logic queries. Two queries which differ only in the names of
their fields and tensors, the order of the arguments of commutative operators,
the nesting of associative ones, the type of an init which is the identity of
its reduction, or scalar factors, have canonical forms with the same term.
"""

from dataclasses import dataclass
from itertools import count, permutations
from typing import Any

from finch import ffuncs
from finch.algebra import FinchOperator
from finch.algebra.algebra import (
    is_associative,
    is_commutative,
    is_distributive,
    is_identity,
)
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    HardAlias,
    Literal,
    LogicExpression,
    LogicNode,
    LogicTree,
    MapJoin,
    Query,
    Table,
)
from finch.symbolic import Chain, Fixpoint, PostOrderDFS, PostWalk, Rewrite

# The init of an aggregate which starts from the identity of its op, whatever
# the type of that identity.
IDENTITY = Literal(None)
# A canonical term writes OUT and reads #0, #1, ... in the order it reads them.
OUT = HardAlias("out")
SLOT = HardAlias("#")
# Every way of naming the fields of a query is tried, so their number is capped.
MAX_FIELDS = 6


@dataclass(frozen=True)
class CanonicalQuery:
    """
    A query in canonical form.

    Attributes:
        term: The query, with its fields named by position and its tensors by
            the order in which it reads them.
        operands: The tensor which each tensor of `term` stands for.
        scalars: The scalar factors hoisted out of the query, which multiply
            its result.
    """

    term: Query
    operands: tuple[Alias, ...]
    scalars: tuple[LogicExpression, ...]


def flatten_associative(node: LogicNode) -> LogicNode | None:
    """`op(a, op(b, c))` => `op(a, b, c)` for an associative op."""
    match node:
        case MapJoin(Literal(FinchOperator() as op) as lit, args) if is_associative(
            op.ftype
        ):
            flat: list[LogicExpression] = []
            for arg in args:
                match arg:
                    case MapJoin(inner, inner_args) if inner == lit:
                        flat.extend(inner_args)
                    case _:
                        flat.append(arg)
            if tuple(flat) != args:
                return MapJoin(lit, tuple(flat))
    return None


def mark_identity_init(node: LogicNode) -> LogicNode | None:
    match node:
        case Aggregate(
            Literal(FinchOperator() as op) as lit, Literal(val) as init, arg, idxs
        ) if init != IDENTITY and is_identity(op.ftype, val):
            return Aggregate(lit, IDENTITY, arg, idxs)
    return None


def split_scalars(
    args: tuple[LogicExpression, ...],
) -> tuple[tuple[LogicExpression, ...], tuple[LogicExpression, ...]]:
    scalars = tuple(arg for arg in args if not arg.fields())
    return scalars, tuple(arg for arg in args if arg.fields())


def hoist_scalars(node: LogicNode) -> LogicNode | None:
    """
    `Aggregate(op, IDENTITY, s * x, idxs)` => `s * Aggregate(op, IDENTITY, x, idxs)`
    for scalars s, where the map distributes over op.
    """
    match node:
        case Aggregate(
            Literal(FinchOperator() as op) as agg_op,
            init,
            MapJoin(Literal(FinchOperator() as f) as map_op, args),
            idxs,
        ) if init == IDENTITY and is_distributive(f.ftype, op.ftype):
            scalars, rest = split_scalars(args)
            if scalars and rest:
                body = rest[0] if len(rest) == 1 else MapJoin(map_op, rest)
                return MapJoin(map_op, (*scalars, Aggregate(agg_op, init, body, idxs)))
    return None


normalize = Rewrite(
    Fixpoint(PostWalk(Chain([flatten_associative, mark_identity_init, hoist_scalars])))
)


def order(node: Any, names: dict[Field, Field]) -> tuple[Any, tuple[Alias, ...]]:
    """
    node, with its fields renamed by names, its tables reading SLOT, and the
    arguments of commutative operators sorted, followed by the tensors which its
    tables read, in the order it reads them.
    """
    match node:
        case Field():
            return names[node], ()
        case Table(Alias() as tns, idxs):
            return Table(SLOT, tuple(names[idx] for idx in idxs)), (tns,)
        case Query(Table(Alias(), idxs), rhs):
            rhs_2, tnss = order(rhs, names)
            return Query(Table(OUT, tuple(names[idx] for idx in idxs)), rhs_2), tnss
        case MapJoin(Literal(FinchOperator() as op) as lit, args) if is_commutative(
            op.ftype
        ):
            ordered = sorted(
                (order(arg, names) for arg in args), key=lambda pair: repr(pair[0])
            )
            return (
                MapJoin(lit, tuple(arg for arg, _ in ordered)),
                tuple(tns for _, tnss in ordered for tns in tnss),
            )
        case Aggregate(op, init, arg, idxs):
            init_2, init_tnss = order(init, names)
            arg_2, arg_tnss = order(arg, names)
            reduced = tuple(sorted((names[idx] for idx in idxs), key=lambda f: f.name))
            return Aggregate(op, init_2, arg_2, reduced), init_tnss + arg_tnss
        case LogicTree():
            children = [order(child, names) for child in node.children]
            return (
                node.make_term(node.head(), *(child for child, _ in children)),
                tuple(tns for _, tnss in children for tns in tnss),
            )
    return node, ()


def number_slots(term: Query) -> Query:
    counter = count()

    def rule(node):
        match node:
            case Table(tns, idxs) if tns == SLOT:
                return Table(HardAlias(f"#{next(counter)}"), idxs)
        return None

    return Rewrite(PostWalk(rule))(term)


def canonicalize(query: Query) -> CanonicalQuery | None:
    """
    The canonical form of query, or None if it has too many fields to try every
    way of naming them.
    """
    match query:
        case Query(Table(Alias() as lhs, idxs), rhs):
            body = normalize(rhs)
            scalars: tuple[LogicExpression, ...] = ()
            match body:
                case MapJoin(Literal(ffuncs.mul) as lit, args):
                    scalars, rest = split_scalars(args)
                    if scalars and rest:
                        body = rest[0] if len(rest) == 1 else MapJoin(lit, rest)
                    else:
                        scalars = ()
            query = Query(Table(lhs, idxs), body)
        case _:
            return None
    fields: dict[Field, None] = {}
    for node in PostOrderDFS(query):
        match node:
            case Field():
                fields[node] = None
    if len(fields) > MAX_FIELDS:
        return None
    names = [Field(f"i{n}") for n in range(len(fields))]
    term, operands = min(
        (
            order(query, dict(zip(fields, perm, strict=True)))
            for perm in permutations(names)
        ),
        key=lambda pair: repr(pair[0]),
    )
    return CanonicalQuery(number_slots(term), operands, scalars)
