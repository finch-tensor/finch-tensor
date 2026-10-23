"""
global value numbering (gvn) for finch assembly.

`assembly_global_value_numbering` replaces a redundant pure expression with a
variable in scope that already holds its value. assignments are kept, copy
propagation and dce can clean up after it.

the analysis collects the values a program computes into a `ValueUniverse`:
variables and slots are leaves, scalar literals are constants by exact type and
bits (`Literal` compares by value only, so `0.0 == -0.0`), and pure `ffuncs`
calls on scalars are applies. memory reads are applies too: loads and lengths
read the heap leaf and struct field reads the struct leaf. a state partitions
the universe into classes of values that are equal on every path to that point,
and the class of a value is its value number (vn). a join keeps the equalities
all incoming paths share.

an assignment writes its variable: the values that read it change, and it joins
the class of its right-hand side. a statement with unknown effects numbers
nothing and gives its target an unknown value.

other writes give memory an unknown value: store and resize write the heap,
setattr and unpacking a struct slot the struct fields, repack and impure calls
both. buffer writes never touch struct fields since buffers are handles. setattr
also writes the root of its target, since mlir rebinds it, and unpack its slot.

the universe is finite, so the analysis terminates, loops included, and merged
values need no phis.
"""

from collections.abc import Iterable

from finch.algebra import FType, StructFType
from finch.symbolic import (
    Apply,
    Constant,
    ControlFlowGraph,
    Leaf,
    PostOrderDFS,
    UnvalidatedForm,
    Value,
    ValueUniverse,
)
from finch.util import qual_str

from .cfg_builder import NumberedStatement
from .dataflow import (
    AbstractAssemblyDataflow,
    ScopedRewriter,
    assembly_dataflow_run,
    has_effects,
    is_scalar_type,
    literal_key,
    pure_op_key,
    root_of,
)
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyStatement,
    AssemblyTree,
    Assert,
    Assign,
    Call,
    GetAttr,
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


_HEAP = Leaf(("mem", "heap"))
_STRUCT = Leaf(("mem", "struct"))


class AssemblyGlobalValueNumbering(AbstractAssemblyDataflow):
    """
    gvn analysis, see the module docstring. a state is {"classes": classes}, the
    partition of `universe`.
    """

    def __init__(self, cfg: ControlFlowGraph):
        super().__init__(cfg)
        # results by `id`, keeping the node alive so its id stays unique
        self._types: dict[int, tuple[AssemblyExpression, FType | None]] = {}
        self._values: dict[int, tuple[AssemblyNode, Value | None]] = {}
        self.op_names: dict[object, str] = {}
        values: list[Value] = []
        for block in cfg.blocks.values():
            for stmt in block.statements:
                values.extend(self.values_of(stmt))
        self.universe = ValueUniverse(values)

    def __str__(self) -> str:
        values = enumerate(self.universe.values)
        table = "\n".join(f"%{i} = {self.label(value)}" for i, value in values)
        return f"{super().__str__()}\nvalue numbers:\n{table}\n"

    def direction(self) -> str:
        return "forward"

    def boundary_state(self) -> dict:
        return {"classes": self.universe.empty()}

    def join(self, state_1: dict, state_2: dict) -> dict:
        return {"classes": self.universe.meet(state_1["classes"], state_2["classes"])}

    def transfer(self, stmts: Iterable[AssemblyNode], state: dict) -> dict:
        classes = state["classes"]
        for stmt in stmts:
            classes = self.step(stmt, classes)
        return {"classes": classes}

    def step(self, stmt: AssemblyNode, classes: tuple[int, ...]) -> tuple[int, ...]:
        """the classes after `stmt`"""
        match stmt:
            case NumberedStatement(stmt, _):
                pass
        assign = self.universe.assign
        if has_effects(stmt):
            classes = assign(assign(classes, _HEAP), _STRUCT)
            match stmt:
                case Assign(Variable(name, _), _):
                    return assign(classes, Leaf(("var", name)))
        else:
            match stmt:
                case Assign(Variable(name, _), rhs):
                    return assign(classes, Leaf(("var", name)), self.value(rhs))
        match stmt:
            case Store() | Resize():
                classes = assign(classes, _HEAP)
            case SetAttr(obj, _, _):
                classes = assign(classes, _STRUCT)
                root = self.value(root_of(obj))
                match root:
                    case Leaf():
                        classes = assign(classes, root)
            case Unpack(Slot(name, type_), _):
                classes = assign(classes, Leaf(("slot", name)))
                match type_:
                    case StructFType():
                        classes = assign(classes, _STRUCT)
            case Repack():
                classes = assign(assign(classes, _HEAP), _STRUCT)
        return classes

    def values_of(self, stmt: AssemblyNode) -> list[Value]:
        """the values `stmt` computes, then the variable it assigns"""
        match stmt:
            case NumberedStatement(stmt, _):
                pass
        match stmt:
            case Assign(Variable(name, _), rhs):
                computed, written = rhs, [Leaf(("var", name))]
            case _:
                computed, written = stmt, []
        values = []
        if not has_effects(stmt):
            for node in PostOrderDFS(computed):
                match node:
                    case AssemblyExpression():
                        value = self.value(node)
                        if value is not None:
                            values.append(value)
        return values + written

    def value(self, node: AssemblyNode) -> Value | None:
        """the value `node` computes, or None if it isn't numbered"""
        if id(node) not in self._values:
            self._values[id(node)] = (node, self._value(node))
        return self._values[id(node)][1]

    def _value(self, node: AssemblyNode) -> Value | None:
        match node:
            case Literal(val) if literal_key(val) is not None:
                return Constant(val)
            case Variable(name, _):
                return Leaf(("var", name))
            case Slot(name, _):
                return Leaf(("slot", name))
            case Call(Literal(op), args):
                op_key = pure_op_key(op)
                if op_key is None or not is_scalar_type(self.result_type(node)):
                    return None
                arg_values = self.arg_values(args)
                if arg_values is None:
                    return None
                if not all(is_scalar_type(self.result_type(arg)) for arg in args):
                    # e.g. `scansearch` reads its buffer argument
                    arg_values += (_HEAP, _STRUCT)
                self.op_names[op_key] = qual_str(op)
                return Apply(op_key, arg_values)
            case Load(buf, idx):
                return self.read("load", (buf, idx), _HEAP)
            case Length(buf):
                return self.read("length", (buf,), _HEAP)
            case GetAttr(obj, Literal(attr)):
                return self.read(("getattr", attr), (obj,), _STRUCT)
        return None

    def read(
        self, op: object, args: tuple[AssemblyNode, ...], memory: Leaf
    ) -> Value | None:
        """a read of `memory` by `op`"""
        arg_values = self.arg_values(args)
        if arg_values is None:
            return None
        match op:
            case ("getattr", attr):
                self.op_names[op] = f"getattr.{attr}"
            case _:
                self.op_names[op] = str(op)
        return Apply(op, (*arg_values, memory))

    def arg_values(self, args: Iterable[AssemblyNode]) -> tuple[Value, ...] | None:
        """the values of `args`, or None if one isn't numbered"""
        values = []
        for arg in args:
            value = self.value(arg)
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    def value_number(self, classes: tuple[int, ...], node: AssemblyNode) -> int | None:
        value = self.value(node)
        return None if value is None else self.universe.number(classes, value)

    def result_type(self, node: AssemblyExpression) -> FType | None:
        if id(node) not in self._types:
            try:
                type_ = node.result_type
            except (AssertionError, AttributeError, KeyError, TypeError):
                type_ = None
            self._types[id(node)] = (node, type_)
        return self._types[id(node)][1]

    def label(self, value: Value) -> str:
        match value:
            case Leaf((_, name)):
                return str(name)
            case Constant(val):
                return repr(val)
            case Apply(op, args):
                indices = ", ".join(f"%{self.universe.indices[arg]}" for arg in args)
                return f"{self.op_names[op]}({indices})"
        return str(value)

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        """assignments show the vn they give, other statements their operand vns"""
        classes = state["classes"]
        match stmt:
            case NumberedStatement(stmt, _):
                pass
        match stmt:
            case Assign(Variable(name, _) as var, _):
                return [(name, f"%{self.value_number(self.step(stmt, classes), var)}")]
            case AssemblyTree():
                annotations: list[tuple[str, object]] = []
                for child in stmt.children:
                    vn = self.value_number(classes, child)
                    if vn is not None:
                        annotations.append((str(child), f"%{vn}"))
                return annotations
        return []


class _GVNRewriter(ScopedRewriter):
    """rewrites each statement with the analysis results"""

    def __init__(self, ctx: AssemblyGlobalValueNumbering):
        super().__init__()
        self.ctx = ctx
        self.states = {stmt.sid: state for stmt, state in ctx.statement_states()}

    def rewrite(self, stmt: AssemblyStatement, sid: int) -> AssemblyStatement:
        state = self.states.get(sid)
        if state is None or has_effects(stmt):
            return stmt
        classes = state["classes"]

        # visible scalar variables by vn, outer and earlier ones first
        candidates: dict[int, list[Variable]] = {}
        for scope in self.scopes:
            for var in scope.values():
                vn = self.ctx.value_number(classes, var)
                if vn is not None and is_scalar_type(var.result_type):
                    candidates.setdefault(vn, []).append(var)

        def replace(node, exclude=None):
            match node:
                case Call() | Load() | Length() | GetAttr():
                    vn = self.ctx.value_number(classes, node)
                    for var in candidates.get(-1 if vn is None else vn, ()):
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
