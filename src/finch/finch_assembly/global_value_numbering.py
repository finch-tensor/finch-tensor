"""
global value numbering (gvn) for finch assembly.

`assembly_global_value_numbering` replaces a redundant pure expression with a
variable in scope that already holds its value. assignments are kept, copy
propagation and dce can clean up after it.

the analysis maps each variable to a value number (vn), an int interned from a
key that says how the value was computed:

- ("const", type, bits): a literal, by exact type and bits (`Literal` compares
  by value only, so `0.0 == -0.0`)
- ("call", op, args): a pure `ffuncs` operator applied to vns
- ("load", buf, idx, heap), ("length", buf, heap), ("getattr", obj, attr,
  struct): memory reads, keyed by the current version of the heap or the struct
  fields
- ("def", sid, key): a new value statement `sid` gives to `key`, e.g. a variable
  assigned an impure call, the slot of an unpack, or memory after a write
- ("phi", block, key): the value of `key` entering `block`, when it differs
  between the incoming paths
- ("unknown", sid, key): the value of `key` when `sid` read it without a vn,
  e.g. function arguments

variables with the same vn hold the same value on every path. a variable that is
undefined on some path is dropped at the join.

writes start new memory versions: store and resize for the heap, setattr and
unpacking a struct slot for structs, repack and impure calls for both. buffer
writes never touch struct fields since buffers are handles. setattr also gives
the root of its target a new value, since mlir rebinds it, and every unpack gives
its slot a new value.

the cfg comes from structured code, so loops are only entered through their
header, and values named inside a loop are dropped there before they go stale.
joins only drop facts and leaves are interned per statement or block, so the
analysis terminates. never key a phi on its incoming vns instead, or
`j = add(j, 1)` diverges.
"""

from collections.abc import Iterable, Sequence

import numpy as np

from finch.algebra import FType, SingletonOperatorFType, StructFType, ffuncs
from finch.algebra.ftypes import FDType
from finch.symbolic import BasicBlock, ControlFlowGraph, PostOrderDFS, UnvalidatedForm
from finch.util import qual_str

from .cfg_builder import NumberedStatement
from .dataflow import AbstractAssemblyDataflow, assembly_dataflow_run
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyStatement,
    AssemblyTree,
    Assert,
    Assign,
    Block,
    BufferLoop,
    Call,
    ForLoop,
    Function,
    GetAttr,
    If,
    IfElse,
    Length,
    Literal,
    Load,
    Module,
    Repack,
    Resize,
    Return,
    SetAttr,
    Slot,
    Store,
    Unpack,
    Variable,
    WhileLoop,
)
from .stages import AssemblyTransform


def assembly_global_value_numbering(node: AssemblyNode) -> AssemblyNode:
    """apply global value numbering to a finch assembly node"""
    return assembly_dataflow_run(
        node,
        AssemblyGlobalValueNumbering,
        lambda pre_node, ctx: _GVNRewriter(ctx)(pre_node),
    )


class AssemblyGlobalValueNumberingTransform(UnvalidatedForm, AssemblyTransform):
    def lower(self, term: Module) -> Module:
        result = assembly_global_value_numbering(term)
        assert isinstance(result, Module)
        return result


_HEAP = ("mem", "heap")
_STRUCT = ("mem", "struct")
_MEMORY = (_HEAP, _STRUCT)


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


def _root_key(node: AssemblyExpression) -> tuple | None:
    """state key of the variable or slot under a getattr chain"""
    match node:
        case GetAttr(obj, _):
            return _root_key(obj)
        case Variable(name, _):
            return ("var", name)
        case Slot(name, _):
            return ("slot", name)
    return None


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
    gvn analysis, see the module docstring. a state maps ("var", name),
    ("slot", name) and the memory versions ("mem", "heap"), ("mem", "struct")
    to vns, `table` interns vn keys and `labels` prints them.
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

    def boundary_state(self) -> dict:
        # the memory a function starts with
        return {key: self.number(("entry", key), f"{key[1]}@entry") for key in _MEMORY}

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
                    self.define(state, ("var", name), sid)
            for key in _MEMORY:
                self.define(state, key, sid)
            return
        match stmt:
            case Assign(Variable(name, _), rhs):
                vn = self.eval(rhs, state, sid, vns)
                if vn is None:
                    self.define(state, ("var", name), sid)
                else:
                    state[("var", name)] = vn
                return
            case AssemblyExpression():
                self.eval(stmt, state, sid, vns)
                return
            case Unpack(_, rhs):
                # the slot is defined, not read
                self.eval(rhs, state, sid, vns)
            case AssemblyTree():
                for child in stmt.children:
                    match child:
                        case AssemblyExpression():
                            self.eval(child, state, sid, vns)
        # effects, after the operands are read
        match stmt:
            case Store() | Resize():
                self.define(state, _HEAP, sid)
            case SetAttr(obj, _, _):
                self.define(state, _STRUCT, sid)
                root = _root_key(obj)
                if root is not None:
                    self.define(state, root, sid)
            case Unpack(Slot(name, type_), _):
                self.define(state, ("slot", name), sid)
                match type_:
                    case StructFType():
                        self.define(state, _STRUCT, sid)
            case Repack():
                for key in _MEMORY:
                    self.define(state, key, sid)

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
            case Slot(name, _):
                vn = self.read(state, ("slot", name), sid)
            case Call(Literal(op), args):
                arg_vns = [self.eval(arg, state, sid, vns) for arg in args]
                op_key = _op_key(op)
                if (
                    op_key is not None
                    and None not in arg_vns
                    and _is_scalar(self.result_type(node))
                ):
                    key: tuple = ("call", op_key, tuple(arg_vns))
                    label = f"{qual_str(op)}({', '.join(f'%{x}' for x in arg_vns)})"
                    if not all(_is_scalar(self.result_type(arg)) for arg in args):
                        # e.g. `scansearch` reads its buffer argument
                        key += (self.read(state, _HEAP, sid),)
                        key += (self.read(state, _STRUCT, sid),)
                        label += f" @%{key[-2]}, %{key[-1]}"
                    if None not in key:
                        vn = self.number(key, label)
            case Load(buf, idx):
                buf_vn = self.eval(buf, state, sid, vns)
                idx_vn = self.eval(idx, state, sid, vns)
                heap = self.read(state, _HEAP, sid)
                if None not in (buf_vn, idx_vn, heap):
                    label = f"load(%{buf_vn}, %{idx_vn}) @%{heap}"
                    vn = self.number(("load", buf_vn, idx_vn, heap), label)
            case Length(buf):
                buf_vn = self.eval(buf, state, sid, vns)
                heap = self.read(state, _HEAP, sid)
                if None not in (buf_vn, heap):
                    label = f"length(%{buf_vn}) @%{heap}"
                    vn = self.number(("length", buf_vn, heap), label)
            case GetAttr(obj, Literal(attr)):
                obj_vn = self.eval(obj, state, sid, vns)
                struct = self.read(state, _STRUCT, sid)
                if None not in (obj_vn, struct):
                    label = f"%{obj_vn}.{attr} @%{struct}"
                    vn = self.number(("getattr", obj_vn, attr, struct), label)
            case AssemblyTree():
                for child in node.children:
                    match child:
                        case AssemblyExpression():
                            self.eval(child, state, sid, vns)
        if vns is not None and vn is not None:
            vns[id(node)] = vn
        return vn

    def define(self, state: dict, key: tuple, sid: int | None) -> None:
        """bind `key` to a new value defined by statement `sid`"""
        # unnumbered statements (loop counters) never create names
        if sid is None:
            state.pop(key, None)
        else:
            state[key] = self.number(("def", sid, key), f"{key[1]}@{sid}")

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


class _GVNRewriter:
    """
    walks the numbered program tracking the variables in scope, and rewrites
    each statement with the analysis results.
    """

    def __init__(self, ctx: AssemblyGlobalValueNumbering):
        self.ctx = ctx
        self.states = {stmt.sid: state for stmt, state in ctx.statement_states()}
        self.scopes: list[dict[str, Variable]] = []

    def __call__(self, node: AssemblyNode) -> AssemblyNode:
        match node:
            case Module(funcs):
                return Module(tuple(self.function(func) for func in funcs))
            case Function():
                return self.function(node)
            case AssemblyStatement():
                return self.statement(node)
        return node

    def function(self, func: Function) -> Function:
        return Function(func.name, func.args, self.scoped(func.body, func.args))

    def scoped(
        self, body: AssemblyStatement, decls: Sequence[Variable] = ()
    ) -> AssemblyStatement:
        """rewrite `body` in a new scope declaring `decls`"""
        # the type checker scopes every block, the interpreter every body
        self.scopes.append({var.name: var for var in decls})
        try:
            match body:
                case Block(bodies):
                    return Block(tuple(self.statement(stmt) for stmt in bodies))
            result = self.statement(body)
            assert isinstance(result, AssemblyStatement)
            return result
        finally:
            self.scopes.pop()

    def statement(
        self, node: AssemblyStatement | AssemblyExpression
    ) -> AssemblyStatement | AssemblyExpression:
        match node:
            case Block():
                return self.scoped(node)
            case If(cond, body):
                return If(cond, self.scoped(body))
            case IfElse(cond, body, else_body):
                return IfElse(cond, self.scoped(body), self.scoped(else_body))
            case WhileLoop(cond, body):
                return WhileLoop(cond, self.scoped(body))
            case ForLoop(var, start, end, body):
                return ForLoop(var, start, end, self.scoped(body, (var,)))
            case BufferLoop(buf, var, body):
                return BufferLoop(buf, var, self.scoped(body, (var,)))
            case NumberedStatement(stmt, sid):
                result = NumberedStatement(self.rewrite(stmt, sid), sid)
                # an assignment declares its variable only after it runs
                match stmt:
                    case Assign(Variable(name, _) as var, _) if not any(
                        name in scope for scope in self.scopes
                    ):
                        self.scopes[-1][name] = var
                return result
        return node

    def rewrite(self, stmt: AssemblyStatement, sid: int) -> AssemblyStatement:
        state = self.states.get(sid)
        if state is None or _is_impure(stmt):
            return stmt
        vns: dict[int, int] = {}
        self.ctx.step(NumberedStatement(stmt, sid), dict(state), vns)

        # visible scalar variables by vn, outer and earlier ones first
        candidates: dict[int, list[Variable]] = {}
        for scope in self.scopes:
            for name, var in scope.items():
                vn = state.get(("var", name))
                if vn is not None and _is_scalar(var.result_type):
                    candidates.setdefault(vn, []).append(var)

        def replace(node, exclude=None):
            match node:
                case Call() | Load() | Length() | GetAttr():
                    for var in candidates.get(vns.get(id(node), -1), ()):
                        if var != exclude and var.type == self.ctx.result_type(node):
                            return var
            # never rewrite buffer operands or getattr objects
            match node:
                case Call(op, args):
                    new_args = tuple(replace(arg) for arg in args)
                    if any(
                        new is not old for new, old in zip(new_args, args, strict=True)
                    ):
                        return Call(op, new_args)
                case Load(buf, idx):
                    new_idx = replace(idx)
                    if new_idx is not idx:
                        return Load(buf, new_idx)
            return node

        match stmt:
            case Assign(lhs, rhs):
                return Assign(lhs, replace(rhs, exclude=lhs))
            case Store(buf, idx, val):
                return Store(buf, replace(idx), replace(val))
            case Resize(buf, size):
                return Resize(buf, replace(size))
            case SetAttr(obj, attr, val):
                return SetAttr(obj, attr, replace(val))
            case Return(arg):
                return Return(replace(arg))
            case Assert(exp):
                return Assert(replace(exp))
        return stmt
