import logging
from typing import Any, TypeVar, cast, overload

from finch.algebra.tensor import Tensor
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Fuse,
    FusedAlias,
    HardAlias,
    Literal,
    LogicEvaluator,
    LogicNode,
    Plan,
    Produces,
    ProtocolAlias,
    Query,
    QueryInto,
    Relabel,
    Reorder,
    Table,
)
from finch.finch_logic.nodes import LogicExpression, LogicStatement, LogicTree
from finch.symbolic import Namespace, PostWalk, Rewrite, UnvalidatedForm
from finch.tensor.scalar import ConstantScalar
from finch.util.logging import LOG_LOGIC_PRE_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_PRE_OPT)

T = TypeVar("T")


@overload
def normalize_names(
    prgm: LogicStatement, bindings: dict[Alias, T]
) -> tuple[LogicStatement, dict[Alias, T]]: ...
@overload
def normalize_names(
    prgm: LogicExpression, bindings: dict[Alias, T]
) -> tuple[LogicExpression, dict[Alias, T]]: ...
@overload
def normalize_names(
    prgm: LogicNode, bindings: dict[Alias, T]
) -> tuple[LogicNode, dict[Alias, T]]: ...
def normalize_names(
    prgm: LogicNode, bindings: dict[Alias, T]
) -> tuple[LogicNode, dict[Alias, T]]:
    """
    Normalizes names of aliases and fields in the logic program to avoid conflicts.
    The fields an aggregate reduces are its own, so they are renamed apart from
    any field of the same name outside it, unless an enclosing Fuse loops over
    them.
    """
    if bindings is None:
        bindings = {}
    spc = Namespace()
    renames: dict[str, str] = {}

    def rule_0(node: LogicNode) -> LogicNode | None:
        match node:
            case FusedAlias(alias, n):
                renamed = rule_0(alias)
                assert isinstance(renamed, HardAlias)
                return FusedAlias(renamed, n)
            case ProtocolAlias(alias, protocols):
                renamed = rule_0(alias)
                assert isinstance(renamed, HardAlias | FusedAlias)
                return ProtocolAlias(renamed, protocols)
            case Alias(name):
                if name in renames:
                    return HardAlias(renames[name])
                new_name = spc.freshen("A")
                renames[name] = new_name
                return HardAlias(new_name)
            case Field(name):
                if name in renames:
                    return Field(renames[name])
                new_name = spc.freshen("i")
                renames[name] = new_name
                return Field(new_name)
            case _:
                return None

    def rename(node: LogicNode, scope: dict[str, str], looped=frozenset()):
        match node:
            case Fuse(idx, body):
                return Fuse(
                    cast(Field, rename(idx, scope, looped)),
                    cast(LogicStatement, rename(body, scope, looped | {idx.name})),
                )
            case Aggregate(op, init, arg, idxs):
                inner = dict(scope)
                for idx in idxs:
                    if idx.name not in looped:
                        inner[idx.name] = spc.freshen("i")
                return Aggregate(
                    op,
                    cast(LogicExpression, rename(init, scope, looped)),
                    cast(LogicExpression, rename(arg, inner, looped)),
                    tuple(cast(Field, rename(idx, inner, looped)) for idx in idxs),
                )
            case Field(name) if name in scope:
                return Field(scope[name])
            case Alias() | Field():
                return Rewrite(rule_0)(node)
            case LogicTree():
                children = (rename(child, scope, looped) for child in node.children)
                return node.make_term(node.head(), *children)
            case _:
                return node

    bindings = {Rewrite(rule_0)(var): tns for var, tns in bindings.items()}
    root = rename(prgm, {})

    return root, bindings


def inline_constant_scalars(prgm: LogicNode) -> LogicNode:
    """
    Replace each reference to a bound `ConstantScalar` with its value.
    """
    constants: dict[Alias, Any] = {}
    produced: set[Alias] = set()

    def gather(node: LogicNode) -> None:
        match node:
            case Query(Table(Alias() as lhs, ()), Table(Literal(val), ())) if (
                isinstance(val, ConstantScalar)
            ):
                constants[lhs] = val.val
            case Produces(args):
                produced.update(a for a in args if isinstance(a, Alias))
        return

    Rewrite(PostWalk(gather))(prgm)
    if not constants:
        return prgm

    def constant_ref(node: LogicNode) -> Alias | None:
        """The constant `node` refers to, if it is exactly such a reference."""
        match node:
            case (
                Table(Alias() as tns, ())
                | Reorder(Table(Alias() as tns, ()), ())
                | Relabel(Table(Alias() as tns, ()), ())
            ) if tns in constants:
                return tns
            case _:
                return None

    def inline(node: LogicNode) -> LogicNode | None:
        match node:
            case Table(Alias() as tns, ()) if tns in constants:
                return Literal(constants[tns])
            case _:
                return None

    # Constants that keep their binding because inlining could not reach them.
    live: set[Alias] = set()

    def inline_stmt(stmt: LogicStatement) -> LogicStatement:
        match stmt:
            case Plan(bodies):
                return Plan(tuple(inline_stmt(body) for body in bodies))
            case Query(Table(Alias() as lhs, _), rhs) if lhs in produced and (
                ref := constant_ref(rhs)
            ):
                # `Query(Table(a, ()), Literal(v))` names no tensor to produce, so a
                # produced query that is *only* a constant reference is left
                # alone -- and the binding it reads has to stay.
                live.add(ref)
                return stmt
            case Query(lhs, rhs):
                return Query(lhs, cast(LogicExpression, Rewrite(PostWalk(inline))(rhs)))
            case QueryInto(lhs, op, rhs):
                return QueryInto(
                    lhs, op, cast(LogicExpression, Rewrite(PostWalk(inline))(rhs))
                )
            case _:
                return stmt

    inlined = inline_stmt(cast(LogicStatement, prgm))

    def drop(node: LogicNode) -> LogicNode | None:
        match node:
            case Plan(bodies):
                kept = tuple(
                    body
                    for body in bodies
                    if not (
                        isinstance(body, Query)
                        and body.lhs.tns in constants
                        and body.lhs.tns not in live
                    )
                )
                return Plan(kept) if kept != bodies else None
            case _:
                return None

    return Rewrite(PostWalk(drop))(inlined)


class LogicNormalizer(UnvalidatedForm, LogicEvaluator):
    def __init__(self, ctx: LogicEvaluator):
        self.ctx: LogicEvaluator = ctx

    def lower(self, prgm: LogicNode, bindings: dict[Alias, Tensor] | None = None):
        root, bindings = normalize_names(prgm, bindings or {})
        root = inline_constant_scalars(root)
        logger.debug(root)
        return self.ctx(root, bindings)
