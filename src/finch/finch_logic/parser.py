"""
Parser for FinchLogic programs as LogicNode.__str__ prints them, so that
parse_logic(str(prgm)) == prgm whenever the literals of prgm print as values
(see format_literal). Literals are read as a small subset of Python, whose
names are looked up in a namespace of operators, numpy scalar types, finch
dtypes and fills, rather than evaluated.
"""

import ast
import inspect
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from lark import Lark, Token, Tree
from lark.indenter import Indenter

from finch.algebra import (
    DynamicFill,
    FinchOperator,
    FType,
    StaticFill,
    ffuncs,
    ftypes,
)

from . import nodes as lgc


class LogicIndenter(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types = ["LPAR", "LSQB"]
    CLOSE_PAREN_types = ["RPAR", "RSQB"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


# Statements and expressions are parsed by shape, and read by position, so
# the heads of nodes, such as MapJoin, are names rather than keywords.
logic_parser = Lark(
    r"""
    start: _NL? stmt*                           -> program
         | _NL? expr _NL?                       -> expression

    stmt: expr "=" expr _NL                     -> query
        | expr "<<" expr ">>=" expr _NL         -> query_into
        | expr ":" _NL block?                   -> header
        | "return" (expr ("," expr)*)? _NL      -> produces
    block: _INDENT stmt* _DEDENT

    ?expr: unary
         | expr "+" unary                       -> add
         | expr "-" unary                       -> sub
    ?unary: atom
          | "-" unary                           -> neg
    ?atom: name
         | atom "." NAME                        -> attr
         | atom "(" args ")"                    -> call
         | atom "[" args "]"                    -> index
         | NUMBER
         | STRING
         | "(" expr ")"
         | "(" ")"                              -> tuple
         | "(" expr ("," expr)* "," ")"         -> tuple
         | "(" expr ("," expr)+ ")"             -> tuple
    args: (expr ("," expr)* ","?)?
    name: NAME

    NAME: /[A-Za-z_#][A-Za-z0-9_#]*/
    NUMBER: /(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?[jJ]?/
    STRING: /'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"/
    _NL: /(\r?\n[\t ]*)+/

    %ignore /[\t \f]+/
    %declare _INDENT _DEDENT
    """,
    parser="lalr",
    postlex=LogicIndenter(),
)


def _literal_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "None": None,
        "True": True,
        "False": False,
        "inf": math.inf,
        "nan": math.nan,
        "np.True_": np.True_,
        "np.False_": np.False_,
        "StaticFill": StaticFill,
        "DynamicFill": DynamicFill,
    }
    for name, value in vars(np).items():
        if isinstance(value, type) and issubclass(value, np.generic):
            namespace[f"np.{name}"] = value
    for value in vars(ftypes).values():
        if isinstance(value, FType):
            namespace[repr(value)] = value
    # Parameterized operators print as calls of the functions which make them.
    namespace.update(
        (name, value)
        for name, value in vars(ffuncs).items()
        if inspect.isfunction(value) and value.__module__ == ffuncs.__name__
    )
    for value in vars(ffuncs).values():
        if isinstance(value, FinchOperator):
            namespace[repr(value)] = value
    return namespace


LITERALS = _literal_namespace()


class _LogicReader:
    def __init__(self, namespace: Mapping[str, Any]):
        self.namespace = namespace

    def block(self, stmts: list[Tree | Token]) -> lgc.LogicStatement:
        # A block of one statement is that statement, as in the printer.
        match stmts:
            case [stmt]:
                return self.statement(stmt)
            case _:
                return lgc.Plan(tuple(self.statement(stmt) for stmt in stmts))

    def statement(self, t: Tree | Token) -> lgc.LogicStatement:
        match t:
            case Tree("query", [lhs, rhs]):
                return lgc.Query(self.table(lhs), self.expression(rhs))
            case Tree("query_into", [lhs, op, rhs]):
                return lgc.QueryInto(
                    self.table(lhs), lgc.Literal(self.value(op)), self.expression(rhs)
                )
            case Tree(
                "header", [Tree("call", [Tree("name", ["fuse"]), Tree("args", [idx])])]
            ):
                return lgc.Fuse(self.field(idx), lgc.Plan())
            case Tree(
                "header",
                [
                    Tree("call", [Tree("name", ["fuse"]), Tree("args", [idx])]),
                    Tree("block", stmts),
                ],
            ):
                return lgc.Fuse(self.field(idx), self.block(stmts))
            case Tree("header", [Tree("name", ["plan"])]):
                return lgc.Plan()
            case Tree("header", [Tree("name", ["plan"]), Tree("block", stmts)]):
                return lgc.Plan(tuple(self.statement(stmt) for stmt in stmts))
            case Tree("produces", args):
                return lgc.Produces(tuple(self.result(arg) for arg in args))
            case _:
                raise ValueError(f"Expected a statement, got {t}")

    def expression(self, t: Tree | Token) -> lgc.LogicExpression:
        match t:
            case Tree("index", _) | Tree("call", [Tree("name", ["Table"]), _]):
                return self.table(t)
            case Tree("call", [Tree("name", ["MapJoin"]), Tree("args", [op, *args])]):
                return lgc.MapJoin(
                    lgc.Literal(self.value(op)),
                    tuple(self.expression(arg) for arg in args),
                )
            case Tree(
                "call",
                [Tree("name", ["Aggregate"]), Tree("args", [op, init, arg, *idxs])],
            ):
                return lgc.Aggregate(
                    lgc.Literal(self.value(op)),
                    self.expression(init),
                    self.expression(arg),
                    self.fields(idxs),
                )
            case Tree("call", [Tree("name", ["Reorder"]), Tree("args", [arg, *idxs])]):
                return lgc.Reorder(self.expression(arg), self.fields(idxs))
            case Tree("call", [Tree("name", ["Relabel"]), Tree("args", [arg, *idxs])]):
                return lgc.Relabel(self.expression(arg), self.fields(idxs))
            case Tree("call", [Tree("name", ["FillTable"]), Tree("args", [tns])]):
                return lgc.FillTable(self.alias(tns))
            case _:
                return lgc.Literal(self.value(t))

    def table(self, t: Tree) -> lgc.Table:
        match t:
            case Tree("index", [Tree("name", [name]), Tree("args", idxs)]):
                return lgc.Table(lgc.HardAlias(str(name)), self.fields(idxs))
            case Tree(
                "index",
                [
                    Tree("call", [Tree("name", [name]), Tree("args", fused)]),
                    Tree("args", idxs),
                ],
            ):
                alias = lgc.FusedAlias(lgc.HardAlias(str(name)), len(fused))
                return lgc.Table(alias, self.fields([*fused, *idxs]))
            case Tree("call", [Tree("name", ["Table"]), Tree("args", [tns, *idxs])]):
                return lgc.Table(lgc.Literal(self.value(tns)), self.fields(idxs))
            case _:
                raise ValueError(f"Expected a table, got {t}")

    def result(self, t: Tree) -> lgc.Alias | lgc.LogicExpression:
        match t:
            case Tree("name", _) | Tree("call", [Tree("name", ["FusedAlias"]), _]):
                return self.alias(t)
            case _:
                return self.expression(t)

    def alias(self, t: Tree) -> lgc.Alias:
        match t:
            case Tree("name", [name]):
                return lgc.HardAlias(str(name))
            case Tree("call", [Tree("name", ["FusedAlias"]), Tree("args", [tns, n])]):
                alias = self.alias(tns)
                assert isinstance(alias, lgc.HardAlias)
                return lgc.FusedAlias(alias, self.value(n))
            case _:
                raise ValueError(f"Expected an alias, got {t}")

    def fields(self, ts: list[Tree]) -> tuple[lgc.Field, ...]:
        return tuple(self.field(t) for t in ts)

    def field(self, t: Tree) -> lgc.Field:
        match t:
            case Tree("name", [name]):
                return lgc.Field(str(name))
            case _:
                raise ValueError(f"Expected a field, got {t}")

    def value(self, t: Tree | Token) -> Any:
        match t:
            case Token(type="NUMBER"):
                if t[-1] in "jJ":
                    return complex(t)
                if any(c in t for c in ".eE"):
                    return float(t)
                return int(t)
            case Token(type="STRING"):
                return ast.literal_eval(t)
            case Tree("neg", [arg]):
                return -self.value(arg)
            case Tree("add", [a, b]):
                return self.value(a) + self.value(b)
            case Tree("sub", [a, b]):
                return self.value(a) - self.value(b)
            case Tree("tuple", items):
                return tuple(self.value(item) for item in items)
            case Tree("name", _) | Tree("attr", _):
                name = self.dotted(t)
                if name not in self.namespace:
                    raise ValueError(f"Unknown literal {name!r}")
                return self.namespace[name]
            case Tree("call", [head, Tree("args", args)]):
                return self.value(head)(*(self.value(arg) for arg in args))
            case _:
                raise ValueError(f"Expected a literal, got {t}")

    def dotted(self, t: Tree) -> str:
        match t:
            case Tree("name", [name]):
                return str(name)
            case Tree("attr", [head, name]):
                return f"{self.dotted(head)}.{name}"
            case _:
                raise ValueError(f"Expected a name, got {t}")


def parse_logic(text: str, namespace: Mapping[str, Any] | None = None) -> lgc.LogicNode:
    """
    Parse a program, or an expression, as LogicNode.__str__ prints it. Names in
    literals are looked up in LITERALS, updated with namespace if given, which
    can name values that don't print as text, such as tensors.
    """
    tree = logic_parser.parse(text if text.endswith("\n") else text + "\n")
    reader = _LogicReader({**LITERALS, **(namespace or {})})
    match tree:
        case Tree("program", stmts):
            return reader.block(stmts)
        case Tree("expression", [expr]):
            return reader.expression(expr)
        case _:
            raise ValueError(f"Expected a program or an expression, got {tree}")
