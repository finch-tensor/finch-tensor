"""Conservative non-SSA GVN over a finite universe of scalar source expressions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from finch.algebra import ffuncs
from finch.symbolic import (
    DataFlowAnalysis,
    PostOrderDFS,
    PostWalk,
    Rewrite,
    UnvalidatedForm,
)

from .cfg_builder import (
    NumberedStatement,
    notation_build_cfg,
    notation_dataflow_postprocess,
    notation_dataflow_preprocess,
)
from .nodes import (
    Assign,
    Block,
    Call,
    Function,
    If,
    IfElse,
    Literal,
    Loop,
    Module,
    NotationExpression,
    NotationNode,
    Repack,
    Return,
    Variable,
)
from .stages import NotationTransform

_OPERATORS = (ffuncs.add, ffuncs.sub, ffuncs.mul)


def _literal_bits(value) -> bool | int | bytes | None:
    scalar_type = type(value)
    if scalar_type is bool or scalar_type is int:
        return value
    if scalar_type is float or scalar_type is complex:
        return np.asarray(value).tobytes()
    if type(scalar_type) is type and issubclass(scalar_type, np.generic):
        dtype = np.dtype(scalar_type)
        if (
            scalar_type is dtype.type
            and dtype.kind in "biufc"
            and dtype.itemsize <= (16 if dtype.kind == "c" else 8)
        ):
            return np.asarray(value).tobytes()
    return None


def _supported_expression(node: NotationNode) -> bool:
    from finch.compile.lower import make_extent

    match node:
        case Variable():
            return True
        case Literal(val):
            return _literal_bits(val) is not None
        case Call(Literal(op), args):
            if op is make_extent:
                return len(args) == 2 and all(
                    _supported_expression(arg) for arg in args
                )
            return (
                any(op is candidate for candidate in _OPERATORS)
                and bool(args)
                and all(_supported_expression(arg) for arg in args)
            )
    return False


def _supported_loops(func: Function) -> bool:
    from finch.compile.lower import make_extent

    for node in PostOrderDFS(func.body):
        match node:
            case Loop(_, Call(Literal(op), _) as extent, _) if op is make_extent:
                if not _supported_expression(extent):
                    return False
            case Loop():
                # An arbitrary extent controls iteration and may mutate bindings.
                return False
    return True


def _supported_returns(node: NotationNode, terminal: bool = True) -> bool:
    # The interpreter keeps executing after Return, unlike the CFG. Restrict
    # rewrites to functions where both execution models have the same paths.
    match node:
        case Return():
            return terminal
        case Block(bodies):
            return all(
                _supported_returns(body, terminal and index == len(bodies) - 1)
                for index, body in enumerate(bodies)
            )
        case _:
            return not any(isinstance(term, Return) for term in PostOrderDFS(node))


def _unsafe_names(func: Function) -> set[str]:
    """Exclude bindings whose lifetime the flat CFG cannot represent."""
    unsafe: set[str] = set()

    def visit(node, bound, nested=False):
        match node:
            case Block(bodies):
                for body in bodies:
                    visit(body, bound, nested)
            case Assign(Variable(name, _), _) | Repack(_, Variable(name, _)):
                if nested and name not in bound:
                    unsafe.add(name)
                bound.add(name)
            case If(_, body):
                visit(body, bound.copy(), True)
            case IfElse(_, body, other):
                visit(body, bound.copy(), True)
                visit(other, bound.copy(), True)
            case Loop(Variable(name, _), _, body):
                unsafe.add(name)
                visit(body, bound.copy(), True)

    visit(func.body, {arg.name for arg in func.args})
    types: dict[str, object | None] = {}
    for term in PostOrderDFS(func.body):
        match term:
            case Variable(name, type_):
                if name in types and types[name] is not type_:
                    unsafe.add(name)
                types[name] = type_
    return unsafe


@dataclass(frozen=True)
class _NumberedValue:
    id: int = field(compare=False)
    owner: object = field(compare=False)


@dataclass(frozen=True)
class _VariableValue(_NumberedValue):
    name: str


@dataclass(frozen=True)
class _LiteralValue(_NumberedValue):
    scalar_type: type
    payload: bool | int | bytes


@dataclass(frozen=True, eq=False)
class _CallValue(_NumberedValue):
    op: object
    operands: tuple[_NumberedValue, ...]

    def __hash__(self) -> int:
        return hash((id(self.op), tuple(operand.id for operand in self.operands)))

    def __eq__(self, other: object) -> bool:
        match other:
            case _CallValue():
                return (
                    self.op is other.op
                    and len(self.operands) == len(other.operands)
                    and all(
                        left is right
                        for left, right in zip(
                            self.operands, other.operands, strict=True
                        )
                    )
                )
            case _:
                return NotImplemented


class _ValuePool:
    def __init__(self) -> None:
        self.values: dict[_NumberedValue, _NumberedValue] = {}
        self.next_id = 0
        self.owner = object()

    def intern(self, candidate: _NumberedValue) -> _NumberedValue:
        value = self.values.setdefault(candidate, candidate)
        if value is candidate:
            self.next_id += 1
        return value

    def variable(self, name: str) -> _NumberedValue:
        return self.intern(_VariableValue(self.next_id, self.owner, name))

    def literal(self, value) -> _NumberedValue | None:
        payload = _literal_bits(value)
        if payload is None:
            return None
        return self.intern(
            _LiteralValue(self.next_id, self.owner, type(value), payload)
        )

    def call(self, op: object, operands: tuple[_NumberedValue, ...]) -> _NumberedValue:
        return self.intern(_CallValue(self.next_id, self.owner, op, operands))


@dataclass(frozen=True)
class _Term:
    value: _NumberedValue
    operands: tuple[int, ...]
    reads: frozenset[str]


class _Universe:
    def __init__(self, func: Function):
        self.terms: list[_Term] = []
        self.indices: dict[_NumberedValue, int] = {}
        self.node_indices: dict[int, int | None] = {}
        self.variables: dict[str, tuple[int, Variable]] = {}
        self.unsafe = _unsafe_names(func)
        self.pool = _ValuePool()
        for node in PostOrderDFS(func):
            self.intern(node)
        self.calls = tuple(i for i, term in enumerate(self.terms) if term.operands)
        self.kills = {
            name: frozenset(
                i for i, term in enumerate(self.terms) if name in term.reads
            )
            for name in self.variables
        }

    def value(self, node) -> _NumberedValue | None:
        match node:
            case Variable(name, _) if name not in self.unsafe:
                return self.pool.variable(name)
            case Literal(val):
                return self.pool.literal(val)
            case Call(Literal(op), args) if any(op is item for item in _OPERATORS):
                operands = tuple(self.value(arg) for arg in args)
                if not operands or any(operand is None for operand in operands):
                    return None
                return self.pool.call(
                    op, tuple(operand for operand in operands if operand is not None)
                )
        return None

    def intern(self, node) -> int | None:
        node_id = id(node)
        if node_id in self.node_indices:
            return self.node_indices[node_id]
        operands: tuple[int, ...] = ()
        reads: frozenset[str] = frozenset()
        match node:
            case Variable(name, _) if name not in self.unsafe:
                value = self.pool.variable(name)
                reads = frozenset((name,))
            case Literal(val):
                literal = self.pool.literal(val)
                if literal is None:
                    self.node_indices[node_id] = None
                    return None
                value = literal
            case Call(Literal(op), args) if any(op is item for item in _OPERATORS):
                children = tuple(self.intern(arg) for arg in args)
                if not children or any(child is None for child in children):
                    self.node_indices[node_id] = None
                    return None
                operands = tuple(child for child in children if child is not None)
                value = self.pool.call(
                    op, tuple(self.terms[child].value for child in operands)
                )
                reads = frozenset().union(*(self.terms[i].reads for i in operands))
            case _:
                self.node_indices[node_id] = None
                return None
        if value in self.indices:
            index = self.indices[value]
            self.node_indices[node_id] = index
            return index
        index = len(self.terms)
        self.terms.append(_Term(value, operands, reads))
        self.indices[value] = index
        self.node_indices[node_id] = index
        match node:
            case Variable(name, _):
                self.variables[name] = (index, node)
        return index

    def lookup(self, node) -> int | None:
        if id(node) in self.node_indices:
            return self.node_indices[id(node)]
        value = self.value(node)
        return self.indices.get(value) if value is not None else None

    def close(self, labels) -> tuple[int, ...]:
        """Canonicalize class labels and close equality under supported calls."""
        parents = list(range(len(self.terms)))

        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left, right = root(left), root(right)
            if left == right:
                return False
            parents[max(left, right)] = min(left, right)
            return True

        groups: dict[object, int] = {}
        for index, label in enumerate(labels):
            union(index, groups.setdefault(label, index))
        changed = True
        while changed:
            changed = False
            signatures: dict[tuple, int] = {}
            for index in self.calls:
                term = self.terms[index]
                match term.value:
                    case _CallValue(op=op):
                        signature = (id(op), tuple(root(i) for i in term.operands))
                    case _:
                        raise AssertionError("call term must contain a call value")
                changed |= union(index, signatures.setdefault(signature, index))
        return tuple(root(index) for index in range(len(parents)))

    @property
    def empty(self) -> tuple[int, ...]:
        return tuple(range(len(self.terms)))


class NotationGlobalValueNumbering(DataFlowAnalysis):
    """Forward must-equality analysis; an empty state dictionary denotes TOP.

    The finite source-term partition deliberately sacrifices expression-pool
    precision to guarantee termination, including on loop backedges.
    """

    def __init__(
        self,
        func: Function,
        *,
        assume_pure_arithmetic: bool = False,
        assume_value_semantics: bool = False,
    ):
        self.assume_value_semantics = assume_value_semantics
        self.assume_pure_arithmetic = assume_pure_arithmetic
        self.supported = _supported_loops(func) and _supported_returns(func.body)
        self.universe = _Universe(func)
        numbered, sid = notation_dataflow_preprocess(Module((func,)))
        self.numbered = numbered
        self.original_count = sid
        super().__init__(notation_build_cfg(numbered, sid))
        self.boundary = {
            "classes": self.universe.empty,
            "bound": frozenset(
                arg.name for arg in func.args if arg.name in self.universe.variables
            ),
        }

    def direction(self) -> str:
        return "forward"

    def join(self, state_1: dict, state_2: dict) -> dict:
        if not state_1:
            return state_2
        if not state_2:
            return state_1
        return {
            "classes": self.universe.close(
                tuple(zip(state_1["classes"], state_2["classes"], strict=True))
            ),
            "bound": state_1["bound"] & state_2["bound"],
        }

    def transfer(self, stmts, state: dict) -> dict:
        if not state:
            return state
        classes, bound = state["classes"], state["bound"]
        universe = self.universe
        for entry in stmts:
            if isinstance(entry, NumberedStatement):
                # Parameters are seeded at entry; index names are outside the domain.
                if entry.sid >= self.original_count:
                    continue
                stmt = entry.stmt
            else:
                stmt = entry
            match stmt:
                case Assign(Variable(name, _), rhs):
                    supported = (
                        _supported_expression(rhs) and universe.lookup(rhs) is not None
                    )
                    if not supported:
                        classes = universe.empty
                    elif name in universe.variables:
                        target, _ = universe.variables[name]
                        source = universe.lookup(rhs)
                        killed = universe.kills[name]
                        labels = [
                            ("fresh", i) if i in killed else ("old", group)
                            for i, group in enumerate(classes)
                        ]
                        if source is not None:
                            surviving = next(
                                (
                                    i
                                    for i, group in enumerate(classes)
                                    if i not in killed and group == classes[source]
                                ),
                                None,
                            )
                            if surviving is not None:
                                labels[target] = labels[surviving]
                            if source == target:
                                continue
                        classes = universe.close(labels)
                    if name in universe.variables:
                        bound = bound | {name}
                case Return(rhs):
                    if not _supported_expression(rhs):
                        classes = universe.empty
                case NotationExpression():
                    if not _supported_expression(stmt):
                        classes = universe.empty
                case _:
                    classes = universe.empty
        return {"classes": classes, "bound": bound}

    def analyze(self) -> None:
        # Reachability and an explicit boundary distinguish unvisited TOP from entry.
        reachable = {}
        pending = [self.cfg.entry_block]
        while pending:
            block = pending.pop()
            if block.id not in reachable:
                reachable[block.id] = block
                pending.extend(reversed(block.successors))
        work = deque(reachable.values())
        queued = set(reachable)
        while work:
            block = work.popleft()
            queued.remove(block.id)
            state = self.boundary if block is self.cfg.entry_block else {}
            for pred in block.predecessors:
                if pred.id in reachable:
                    state = self.join(state, self.output_states[pred.id])
            self.input_states[block.id] = state
            result = self.transfer(block.statements, state)
            if result != self.output_states[block.id]:
                self.output_states[block.id] = result
                for successor in block.successors:
                    if successor.id not in queued:
                        queued.add(successor.id)
                        work.append(successor)

    def stmt_str(self, stmt, state: dict) -> str:
        return f"Classes: {state.get('classes', 'TOP')} | Stmt: {stmt}"

    def rewrite(self) -> Module:
        from finch.compile.lower import make_extent

        if (
            not self.supported
            or not self.assume_pure_arithmetic
            or not self.assume_value_semantics
        ):
            result = notation_dataflow_postprocess(self.numbered)
            assert isinstance(result, Module)
            return result

        replacements = {}
        universe = self.universe
        for block in self.cfg.blocks.values():
            state = self.input_states[block.id]
            for entry in block.statements:
                if not state:
                    break
                match entry:
                    case Call(Literal(op), _) if op is make_extent:
                        if not _supported_expression(entry):
                            result = notation_dataflow_postprocess(self.numbered)
                            assert isinstance(result, Module)
                            return result
                    case NumberedStatement(Assign(lhs, Call() as rhs), sid) if (
                        sid < self.original_count
                    ):
                        source = universe.lookup(rhs)
                        if (
                            source is not None
                            and _supported_expression(rhs)
                            and universe.terms[source].reads <= state["bound"]
                        ):
                            for name, (index, var) in universe.variables.items():
                                if (
                                    name in state["bound"]
                                    and state["classes"][index]
                                    == state["classes"][source]
                                ):
                                    replacements[sid] = Assign(lhs, var)
                                    break
                state = self.transfer([entry], state)

        def replace(node):
            match node:
                case NumberedStatement(_, sid) if sid in replacements:
                    return NumberedStatement(replacements[sid], sid)
            return None

        result = notation_dataflow_postprocess(
            Rewrite(PostWalk(replace))(self.numbered)
        )
        assert isinstance(result, Module)
        return result


def notation_global_value_numbering(
    node: NotationNode,
    *,
    assume_pure_arithmetic: bool = False,
    assume_value_semantics: bool = False,
) -> NotationNode:
    """Replace redundant pure scalar RHS calls with definitely available variables.

    Analyze functions independently. Tensor accesses and unknown effects are
    conservative barriers; branch-local names and loop indices are not numbered.
    Assignments remain in place so that lexical binding creation is preserved.
    Rewriting requires both contracts. ``assume_pure_arithmetic`` guarantees that
    supported arithmetic and implicit runtime hooks are pure, deterministic and
    well-typed. ``assume_value_semantics`` guarantees that scalar object identity
    is unobservable, including through opaque consumers and returned values.
    Without both guarantees the original IR is returned.
    """
    match node:
        case Module(funcs):
            updated = []
            for func in funcs:
                result = notation_global_value_numbering(
                    func,
                    assume_pure_arithmetic=assume_pure_arithmetic,
                    assume_value_semantics=assume_value_semantics,
                )
                assert isinstance(result, Function)
                updated.append(result)
            return Module(tuple(updated))
        case Function():
            if not assume_pure_arithmetic or not assume_value_semantics:
                return node
            if not _supported_loops(node) or not _supported_returns(node.body):
                return node
            analysis = NotationGlobalValueNumbering(
                node,
                assume_pure_arithmetic=assume_pure_arithmetic,
                assume_value_semantics=assume_value_semantics,
            )
            if not analysis.universe.calls:
                return node
            analysis.analyze()
            return analysis.rewrite().funcs[0]
        case _:
            raise TypeError(
                "Global value numbering expects a Notation Module or Function"
            )


class NotationGlobalValueNumberingTransform(UnvalidatedForm, NotationTransform):
    def __init__(
        self,
        *,
        assume_pure_arithmetic: bool = False,
        assume_value_semantics: bool = False,
    ):
        self.assume_pure_arithmetic = assume_pure_arithmetic
        self.assume_value_semantics = assume_value_semantics

    def lower(self, term: Module) -> Module:
        result = notation_global_value_numbering(
            term,
            assume_pure_arithmetic=self.assume_pure_arithmetic,
            assume_value_semantics=self.assume_value_semantics,
        )
        assert isinstance(result, Module)
        return result
