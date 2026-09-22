"""
Parser for Finch Assembly in Python's multiline strings.

Supports constructing assembly nodes objects from parsed strings.

There is also a dedicated VS Code extension for proper highlighting
of tagged strings: https://github.com/finch-tensor/vscode-finch-assembly.
The extension is not yet available on VS Code marketplace. You can find
installation file here: https://github.com/finch-tensor/vscode-finch-assembly/releases
"""

import numpy as np

from lark import Lark, Token, Tree

from finch.algebra import ffuncs

from . import nodes as asm

assembly_parser = Lark(
    """
    %import common.CNAME
    %import common.INT
    %import common.DECIMAL
    %import common.CPP_COMMENT
    %import common.C_COMMENT
    %import common.NEWLINE
    %import common.WS_INLINE
    %ignore WS_INLINE

    _FINCH: "finch" | "finch-asm"
    _NEWLINE: NEWLINE
    _COMMENT: C_COMMENT | CPP_COMMENT
    INFIX_OP: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>"
      | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!="
    OP: "min" | "max" | "add" | "sub" | "mul"

    start: _FINCH _NEWLINE+ block
    block: (_stmt _NEWLINE+)* _stmt
    _stmt: assign
         | increment
         | for_loop
         | while_loop
         | if
         | if_else
         | resize
         | _COMMENT
    ?access_expr: access_expr INFIX_OP access_expr | reference | INT
    ?reference: CNAME | attribute
    attribute: reference "." CNAME
    access: reference "[" access_expr "]"
    call: CNAME "(" expr ("," expr)* ")"
    ?expr: reference | INT | DECIMAL | access | scansearch | bin_op | call | expr INFIX_OP expr
    ?lhs: reference | access
    assign: lhs "=" expr
    increment: lhs INFIX_OP "=" expr
    resize: "resize" "(" reference "," expr ")"
    scansearch: "scansearch" "(" CNAME "," expr "," expr "," expr ")"
    bin_op: OP "(" expr "," expr ")"
    for_loop: "for" "(" CNAME "in" access_expr ":" access_expr ")" _NEWLINE+ block _NEWLINE+ "end"
    while_loop: "while" "(" expr ")" _NEWLINE+ block _NEWLINE+ "end"
    if: "if" "(" expr ")" _NEWLINE+ block _NEWLINE+ "end"
    if_else: "if" "(" expr ")" _NEWLINE+ block _NEWLINE+ "else" _NEWLINE+ block _NEWLINE+ "end"
"""  # noqa: E501
)

_OPS = {
    "+": ffuncs.add,
    "-": ffuncs.sub,
    "*": ffuncs.mul,
    "/": ffuncs.truediv,
    "<": ffuncs.lt,
    "<=": ffuncs.le,
    ">": ffuncs.gt,
    ">=": ffuncs.ge,
    "==": ffuncs.eq,
    "!=": ffuncs.ne,
    "min": ffuncs.min,
    "max": ffuncs.max,
    "add": ffuncs.add,
    "sub": ffuncs.sub,
    "mul": ffuncs.mul,
}


def parse_assembly(
    code: str, vars: dict[str, asm.AssemblyExpression], position_type: type = np.intp
) -> asm.AssemblyStatement:
    """
    Parse Finch Assembly code and convert it to assembly node objects.

    Takes a string containing Finch Assembly code and transforms it into a structured
    representation using assembly nodes. The parser supports assignments, increments,
    for/while loops, if/if-else statements, buffer and field accesses, and calls.

    Args:
        code: The Finch Assembly code to parse. Should start with "finch" or "finch-asm"
            followed by assembly statements. Comments (C/C++ style) are supported.
        vars: Dictionary mapping names to Assembly expressions.
            Used to resolve variable references in the assembly code.
        position_type: NumPy integer type to use for integer literals. Affects the dtype
            of parsed integer constants. (default: np.intp)

    Returns:
        A Finch Assembly Block representing the parsed code.

    Raises:
        Exception: If the parser encounters unrecognized syntax or tree nodes.

    Example:
        >>> from finch.finch_assembly import nodes as asm
        >>> vars = {"i": asm.Variable("i", int), "arr": asm.Variable("arr", np.ndarray)}
        >>> code = '''finch
        ... arr[i] = 42
        ... '''
        >>> stmt = parse_assembly(code, vars)
    """
    tree = assembly_parser.parse(code.strip())

    def ctx(tree: Tree):
        match tree:
            case Token("CNAME", "true"):
                return asm.Literal(True)
            case Token("CNAME", "false"):
                return asm.Literal(False)
            case Token("CNAME", val):
                return vars[val]
            case Token("OP" | "INFIX_OP", val):
                return _OPS[val]
            case Token("INT", val):
                return asm.Literal(position_type(val))
            case Token("DECIMAL", val):
                return asm.Literal(float(val))
            case Tree("start", [Tree("block", bodies)]):
                return asm.Block(tuple(ctx(b) for b in bodies))
            case Tree("for_loop", [i, start, stop, Tree("block", bodies)]):
                return asm.ForLoop(
                    ctx(i),
                    ctx(start),
                    ctx(stop),
                    asm.Block(tuple(ctx(b) for b in bodies)),
                )
            case Tree("if", [cond, Tree("block", bodies)]):
                return asm.If(ctx(cond), asm.Block(tuple(ctx(b) for b in bodies)))
            case Tree("while_loop", [cond, Tree("block", bodies)]):
                return asm.WhileLoop(
                    ctx(cond), asm.Block(tuple(ctx(b) for b in bodies))
                )
            case Tree(
                "if_else", [cond, Tree("block", bodies), Tree("block", else_bodies)]
            ):
                return asm.IfElse(
                    ctx(cond),
                    asm.Block(tuple(ctx(b) for b in bodies)),
                    asm.Block(tuple(ctx(b) for b in else_bodies)),
                )
            case Tree("resize", [arr, size]):
                return asm.Resize(ctx(arr), ctx(size))
            case Tree("attribute", [obj, Token("CNAME", name)]):
                return asm.GetAttr(ctx(obj), asm.Literal(name))
            case Tree("call", [op, *args]):
                return asm.Call(ctx(op), tuple(ctx(arg) for arg in args))
            case Tree("scansearch", [arr, x, lo, hi]):
                return asm.Call(
                    asm.Literal(ffuncs.scansearch), (ctx(arr), ctx(x), ctx(lo), ctx(hi))
                )
            case Tree("bin_op", [op, expr1, expr2]):
                return asm.Call(asm.Literal(ctx(op)), (ctx(expr1), ctx(expr2)))
            case Tree("assign", [lhs, expr]):
                return assign(ctx(lhs), ctx(expr))
            case Tree("access", [tns, access_expr]):
                return asm.Load(ctx(tns), ctx(access_expr))
            case Tree("increment", [lhs, op, expr]):
                lhs_e = ctx(lhs)
                return assign(lhs_e, asm.Call(asm.Literal(ctx(op)), (lhs_e, ctx(expr))))
            case Tree("expr" | "access_expr", [expr1, op, expr2]):
                return asm.Call(asm.Literal(ctx(op)), (ctx(expr1), ctx(expr2)))
            case other:
                raise Exception(f"{other} not recognized.")

    def assign(lhs, rhs):
        match lhs:
            case asm.Variable():
                return asm.Assign(lhs, rhs)
            case asm.Load(buffer, index):
                return asm.Store(buffer, index, rhs)
            case asm.GetAttr(obj, attr):
                return asm.SetAttr(obj, attr, rhs)
            case _:
                raise TypeError(f"Invalid assembly assignment target: {lhs}")

    return ctx(tree)
