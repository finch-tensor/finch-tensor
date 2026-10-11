"""
global value numbering (gvn) for finch assembly.

the analysis maps each variable to a value number (vn), an int interned from a
key that says how the value was computed:

- ("const", type, bits): a literal, by exact type and bits (`Literal` compares
  by value only, so `0.0 == -0.0`)
- ("call", op, args): a pure `ffuncs` operator applied to vns
- ("def", sid): the value statement `sid` assigns when its rhs has no vn, e.g.
  the result of an impure call
- ("phi", block, key): the value of `key` entering `block`, when it differs
  between the incoming paths
- ("unknown", sid, key): the value of `key` when `sid` read it without a vn,
  e.g. function arguments

variables with the same vn hold the same value on every path. a variable that is
undefined on some path is dropped at the join.

the cfg comes from structured code, so loops are only entered through their
header, and values named inside a loop are dropped there before they go stale.
joins only drop facts and leaves are interned per statement or block, so the
analysis terminates. never key a phi on its incoming vns instead, or
`j = add(j, 1)` diverges.
"""

from collections.abc import Iterable

import numpy as np

from finch.algebra import FType, SingletonOperatorFType, StructFType, ffuncs
from finch.algebra.ftypes import FDType
from finch.symbolic import BasicBlock, ControlFlowGraph, PostOrderDFS
from finch.util import qual_str

from .cfg_builder import NumberedStatement
from .dataflow import AbstractAssemblyDataflow
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyTree,
    Assign,
    Call,
    Literal,
    Variable,
)


def _const_key(val) -> tuple | None:
    match val:
        case bool() | int():
            return ("const", type(val), val)
        case float() | complex() | np.number() | np.bool_():
            return ("const", type(val), np.asarray(val).tobytes())
    return None


def _op_key(op) -> object | None:
    """key of a pure operator, or None if calling `op` may have effects"""
    # user-defined finch operators can mutate their arguments, `ffuncs` ones don't
    if type(op).__module__ != ffuncs.__name__:
        return None
    match op.ftype:
        case SingletonOperatorFType() as op_type:
            return op_type
    # e.g. `choose(0.0) == choose(-0.0)`, so key by identity
    return ("id", id(op))


def _is_scalar(type_: FType | None) -> bool:
    match type_:
        case StructFType():
            return False
        case FDType():
            return True
    return False


def _is_impure(stmt: AssemblyNode) -> bool:
    for node in PostOrderDFS(stmt):
        match node:
            case Call(Literal(op), _) if _op_key(op) is not None:
                pass
            case Call():
                return True
    return False


class AssemblyGlobalValueNumbering(AbstractAssemblyDataflow):
    """
    gvn analysis, see the module docstring. a state maps ("var", name) to the vn
    of the variable, `table` interns vn keys and `labels` prints them.
    """

    def __init__(self, cfg: ControlFlowGraph):
        super().__init__(cfg)
        self.table: dict[tuple, int] = {}
        self.labels: list[str] = []
        # result types by `id`, keeping the node alive so its id stays unique
        self._types: dict[int, tuple[AssemblyExpression, FType | None]] = {}

    def __str__(self) -> str:
        table = "\n".join(f"%{vn} = {label}" for vn, label in enumerate(self.labels))
        return f"{super().__str__()}\nvalue numbers:\n{table}\n"

    def direction(self) -> str:
        return "forward"

    def join(self, state_1: dict, state_2: dict) -> dict:
        return {key: vn for key, vn in state_1.items() if state_2.get(key) == vn}

    def join_states(self, block: BasicBlock, states: list[dict]) -> dict:
        joined = {}
        for key, vn in states[0].items():
            vns = [state.get(key) for state in states[1:]]
            if None in vns:
                continue
            if all(other == vn for other in vns):
                joined[key] = vn
            else:
                label = f"{key[1]}@{block.id}"
                joined[key] = self.number(("phi", block.id, key), label)
        return joined

    def transfer(self, stmts: Iterable[AssemblyNode], state: dict) -> dict:
        state = dict(state)
        for stmt in stmts:
            self.step(stmt, state)
        return state

    def step(
        self, stmt: AssemblyNode, state: dict, vns: dict[int, int] | None = None
    ) -> None:
        """apply a statement to `state` in place, recording vns by `id` in `vns`"""
        match stmt:
            case NumberedStatement(stmt, sid):
                pass
            case _:
                sid = None
        if _is_impure(stmt):
            # nothing is numbered in a statement with unknown effects
            match stmt:
                case Assign(Variable(name, _), _):
                    self.define(state, name, sid)
            return
        match stmt:
            case Assign(Variable(name, _), rhs):
                vn = self.eval(rhs, state, sid, vns)
                if vn is None:
                    self.define(state, name, sid)
                else:
                    state[("var", name)] = vn
            case AssemblyExpression():
                self.eval(stmt, state, sid, vns)
            case AssemblyTree():
                for child in stmt.children:
                    match child:
                        case AssemblyExpression():
                            self.eval(child, state, sid, vns)

    def eval(
        self,
        node: AssemblyExpression,
        state: dict,
        sid: int | None,
        vns: dict[int, int] | None,
    ) -> int | None:
        """vn of `node`, naming unknown leaves in `state` if needed"""
        vn = None
        match node:
            case Literal(val):
                key = _const_key(val)
                if key is not None:
                    vn = self.number(key, repr(val))
            case Variable(name, _):
                vn = self.read(state, ("var", name), sid)
            case Call(Literal(op), args):
                arg_vns = [self.eval(arg, state, sid, vns) for arg in args]
                op_key = _op_key(op)
                if (
                    op_key is not None
                    and None not in arg_vns
                    and all(_is_scalar(self.result_type(e)) for e in (*args, node))
                ):
                    label = f"{qual_str(op)}({', '.join(f'%{x}' for x in arg_vns)})"
                    vn = self.number(("call", op_key, tuple(arg_vns)), label)
            case AssemblyTree():
                for child in node.children:
                    match child:
                        case AssemblyExpression():
                            self.eval(child, state, sid, vns)
        if vns is not None and vn is not None:
            vns[id(node)] = vn
        return vn

    def define(self, state: dict, name: str, sid: int | None) -> None:
        """bind `name` to the value assigned by statement `sid`"""
        # unnumbered statements (loop counters) never create names
        if sid is None:
            state.pop(("var", name), None)
        else:
            state[("var", name)] = self.number(("def", sid), f"{name}@{sid}")

    def read(self, state: dict, key: tuple, sid: int | None) -> int | None:
        vn = state.get(key)
        # unnumbered statements (loop counters) never create names
        if vn is None and sid is not None:
            vn = self.number(("unknown", sid, key), f"{key[1]}@{sid}")
            state[key] = vn
        return vn

    def number(self, key: tuple, label: str) -> int:
        vn = self.table.get(key)
        if vn is None:
            vn = self.table[key] = len(self.labels)
            self.labels.append(label)
        return vn

    def result_type(self, node: AssemblyExpression) -> FType | None:
        if id(node) not in self._types:
            try:
                type_ = node.result_type
            except (AssertionError, AttributeError, KeyError, TypeError):
                type_ = None
            self._types[id(node)] = (node, type_)
        return self._types[id(node)][1]

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        """assignments show the vn they bind, other statements their operand vns"""
        state = dict(state)
        vns: dict[int, int] = {}
        self.step(stmt, state, vns)
        match stmt:
            case NumberedStatement(Assign(Variable(name, _), _), _) | Assign(
                Variable(name, _), _
            ):
                vn = state.get(("var", name))
                return [(name, "?" if vn is None else f"%{vn}")]
            case NumberedStatement(AssemblyTree() as tree, _) | (
                AssemblyTree() as tree
            ):
                return [
                    (str(child), f"%{vns[id(child)]}")
                    for child in tree.children
                    if id(child) in vns
                ]
        return []
