"""
global value numbering (gvn) for finch assembly.

the analysis collects the values a program computes into a `ValueUniverse`:
variables are leaves, scalar literals are constants by exact type and bits
(`Literal` compares by value only, so `0.0 == -0.0`), and pure `ffuncs` calls on
scalars are applies. a state partitions the universe into classes of values that
are equal on every path to that point, and the class of a value is its value
number (vn). a join keeps the equalities all incoming paths share.

an assignment writes its variable: the values that read it change, and it joins
the class of its right-hand side. a statement with unknown effects numbers
nothing and gives its target an unknown value.

the universe is finite, so the analysis terminates, loops included, and merged
values need no phis.
"""

from collections.abc import Iterable

from finch.algebra import FType
from finch.symbolic import (
    Apply,
    Constant,
    ControlFlowGraph,
    Leaf,
    PostOrderDFS,
    Value,
    ValueUniverse,
)
from finch.util import qual_str

from .cfg_builder import NumberedStatement
from .dataflow import (
    AbstractAssemblyDataflow,
    has_effects,
    is_scalar_type,
    literal_constant,
    pure_op_key,
)
from .nodes import (
    AssemblyExpression,
    AssemblyNode,
    AssemblyTree,
    Assign,
    Call,
    Literal,
    Variable,
)


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
        match stmt:
            case Assign(Variable(name, _), rhs):
                value = None if has_effects(stmt) else self.value(rhs)
                return self.universe.assign(classes, Leaf(("var", name)), value)
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
            case Literal(val):
                return literal_constant(val)
            case Variable(name, _):
                return Leaf(("var", name))
            case Call(Literal(op), args):
                op_key = pure_op_key(op)
                if op_key is None or not is_scalar_type(self.result_type(node)):
                    return None
                arg_values = self.arg_values(args)
                if arg_values is None:
                    return None
                self.op_names[op_key] = qual_str(op)
                return Apply(op_key, arg_values)
        return None

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
