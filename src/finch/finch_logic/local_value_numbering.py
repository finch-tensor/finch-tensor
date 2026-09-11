"""Local value-numbering analysis for sequential, aliased Finch Logic plans.

Adapted from the hash-based local value-numbering approach in Steven S. Muchnick,
Advanced Compiler Design and Implementation (1997), section 12.4.1, pp. 344-348.
Unlike its Remove routine (Figure 12.14, p. 347), this analysis versions changed
operands rather than deleting historical expression keys. The table records value
equivalence, not available replacement candidates. It does not implement the
textbook's rewrites, commutative matching, or global algorithm in section 12.4.2.

This analysis records values immediately after definitions, not the current
contents of all tensors ever assigned those values. Equal value numbers do not
prove that a replacement is available or that two outputs may share storage.
Branches, loops, reductions and general alias/effect analysis are not supported.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from finch.algebra import FType, TensorFType, ffuncs, ftype, return_type
from finch.symbolic import PostOrderDFS

from .nodes import (
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)

_NUMPY_SCALARS = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)
_SCALARS = (int, float, complex, *_NUMPY_SCALARS)
_NUMPY_TYPES = tuple(ftype(t) for t in _NUMPY_SCALARS)
_TYPES = tuple(ftype(t) for t in _SCALARS)


@dataclass(frozen=True)
class LogicDefinition:
    """An input, query occurrence, or storage invalidation in one analysis run.

    Query IDs are nonnegative and follow execution order. Negative IDs denote
    inputs or synthetic versions; ``invalidated_by`` identifies the writing query.
    """

    sid: int
    alias: Alias
    query: Query | None = None
    invalidated_by: int | None = None


@dataclass
class LogicLocalValueNumberingResult:
    """Definition snapshots and stored-value equivalence within a single plan.

    ``uses`` records query-entry definitions referenced by the RHS. For opaque
    expressions it does not resolve reads after effects nested inside that RHS.
    """

    definitions: dict[int, LogicDefinition]
    uses: dict[int, tuple[int, ...]]
    value_numbers: dict[int, int]
    current_definitions: dict[Alias, int]

    @property
    def equivalence_classes(self) -> dict[int, tuple[int, ...]]:
        """Group query definitions by their stored value, including singletons."""
        groups: dict[int, list[int]] = {}
        for sid, definition in self.definitions.items():
            if definition.query is not None:
                vn = self.value_numbers[sid]
                groups.setdefault(vn, []).append(sid)
        return {vn: tuple(sids) for vn, sids in groups.items()}


@dataclass(frozen=True)
class _Value:
    number: int
    dtype: FType | None
    ndim: int | None
    materialized: bool = False


@dataclass(frozen=True)
class _Expression:
    value: _Value
    fields: tuple[Field, ...]


def _statements(node: LogicStatement) -> Iterator[Query | Produces]:
    match node:
        case Plan(bodies):
            for body in bodies:
                for stmt in _statements(body):
                    yield stmt
                    if isinstance(stmt, Produces):
                        return
        case Query() | Produces():
            yield node
        case _:
            raise TypeError(f"Unsupported Logic statement: {type(node).__name__}")


class _LogicLocalValueNumbering:
    def __init__(self, bindings: dict[Alias, TensorFType]):
        self.result = LogicLocalValueNumberingResult({}, {}, {}, {})
        self.values: dict[Alias, _Value] = {}
        self.external = tuple(bindings)
        self.table: dict[tuple, int] = {}
        self.next_value = 0
        self.next_version = -1
        for alias, tp in bindings.items():
            dtype: FType | None = tp.element_type
            dtype = dtype if any(dtype is t for t in _TYPES) else None
            value = _Value(self.fresh(), dtype, tp.ndim)
            self.version(alias, value)

    def fresh(self) -> int:
        number = self.next_value
        self.next_value += 1
        return number

    def intern(self, key: tuple) -> int:
        if key not in self.table:
            self.table[key] = self.fresh()
        return self.table[key]

    def record(self, definition: LogicDefinition, value: _Value) -> None:
        self.result.definitions[definition.sid] = definition
        self.result.value_numbers[definition.sid] = value.number
        self.result.current_definitions[definition.alias] = definition.sid
        self.values[definition.alias] = value

    def version(self, alias: Alias, value: _Value, sid: int | None = None) -> None:
        definition = LogicDefinition(self.next_version, alias, invalidated_by=sid)
        self.next_version -= 1
        self.record(definition, value)

    def invalidate(self, aliases, sid: int, *, unknown_effect: bool = False) -> None:
        for alias in aliases:
            old = self.values[alias]
            value = _Value(
                self.fresh(),
                None if unknown_effect else old.dtype,
                None if unknown_effect else old.ndim,
            )
            self.version(alias, value, sid)

    def reads(self, expr) -> tuple[int, ...]:
        uses = []
        for node in PostOrderDFS(expr):
            match node:
                case Alias():
                    if node not in self.values:
                        raise ValueError(f"Undefined tensor alias: {node.name}")
                    uses.append(self.result.current_definitions[node])
                case Table(Literal(), _):
                    raise ValueError("Extract tensor literals before value numbering")
                case Table(Alias() as alias, fields):
                    ndim = self.values[alias].ndim
                    if len(set(fields)) != len(fields):
                        raise ValueError("Table fields must be distinct")
                    if ndim is not None and len(fields) != ndim:
                        raise ValueError(
                            f"Table rank does not match alias {alias.name}"
                        )
        return tuple(uses)

    def expression(self, expr: LogicExpression) -> _Expression | None:
        match expr:
            case Table(Alias() as alias, fields):
                value = self.values[alias]
                if value.dtype is not None:
                    return _Expression(value, fields)
            case Literal(val) if type(val) in _SCALARS:
                # Literal/StaticFill equality collapses signed zero. Exact scalar
                # types and bytes also preserve complex signs and NaN payloads.
                bits: Any = val if type(val) is int else np.asarray(val).tobytes()
                number = self.intern((Literal, type(val), bits))
                return _Expression(_Value(number, ftype(val), 0), ())
            case MapJoin(Literal(op), args) if (
                any(op is f for f in (ffuncs.add, ffuncs.sub, ffuncs.mul))
                and len(args) == 2
            ):
                operands = [self.expression(arg) for arg in args]
                if any(arg is None for arg in operands):
                    return None
                known = [arg for arg in operands if arg is not None]
                fields = tuple(dict.fromkeys(f for arg in known for f in arg.fields))
                if any(arg.fields and set(arg.fields) != set(fields) for arg in known):
                    return None  # Broadcasting between non-scalar arrays.
                types = [
                    arg.value.dtype for arg in known if arg.value.dtype is not None
                ]
                if len(types) != len(known):
                    return None
                # Only trusted operators on exact, standard numeric types reach
                # Finch's inference, which evaluates operators on sample scalars.
                try:
                    dtype = return_type(op, *types)
                except (TypeError, ValueError, NotImplementedError):
                    return None
                if not any(dtype is t for t in _TYPES):
                    return None
                key = (
                    MapJoin,
                    op,
                    tuple(
                        (arg.value.number, tuple(fields.index(f) for f in arg.fields))
                        for arg in known
                    ),
                    dtype,
                )
                return _Expression(_Value(self.intern(key), dtype, len(fields)), fields)
            case Reorder(arg, fields):
                inner = self.expression(arg)
                if inner is None:
                    return None
                if len(set(fields)) != len(fields):
                    raise ValueError("Reorder fields must be distinct")
                if set(fields) != set(inner.fields):
                    return None  # Inserting/dropping axes needs extent information.
                if fields == inner.fields:
                    return inner
                permutation = tuple(inner.fields.index(f) for f in fields)
                number = self.intern((Reorder, inner.value.number, permutation))
                return _Expression(
                    _Value(number, inner.value.dtype, len(fields)), fields
                )
        return None

    def materialize(self, value: _Value) -> _Value:
        # Keep the Query boundary: literal and external tensor metadata need not
        # survive allocation unchanged. Copies of known numeric local results are
        # idempotent. Operand VNs carry shape/fill provenance without comparing
        # DynamicFill objects (whose equality intentionally ignores their values).
        if value.materialized:
            return value
        dtype = value.dtype if any(value.dtype is t for t in _NUMPY_TYPES) else None
        return _Value(self.intern((Query, value.number)), dtype, value.ndim, True)

    def analyze(self, plan: Plan) -> LogicLocalValueNumberingResult:
        sid = 0
        for stmt in _statements(plan):
            match stmt:
                case Produces():
                    self.reads(stmt)
                    break
                case Query(lhs, rhs):
                    self.result.uses[sid] = self.reads(rhs)
                    expr = self.expression(rhs)
                    old = self.values.get(lhs)
                    if expr is None:
                        self.invalidate(tuple(self.values), sid, unknown_effect=True)
                        value = _Value(self.fresh(), None, None)
                    else:
                        value = self.materialize(expr.value)
                    if lhs in self.external:
                        self.invalidate(self.external, sid)
                        value = _Value(
                            self.fresh(),
                            old.dtype if old and expr is not None else None,
                            old.ndim if old and expr is not None else None,
                        )
                    elif old is not None and old.number != value.number:
                        # Equal rank/type is not proof of equal extents or fill.
                        # Preserve storage metadata only for modeled writes.
                        value = _Value(
                            self.fresh(),
                            old.dtype if expr is not None else None,
                            old.ndim if expr is not None else None,
                        )
                    self.record(LogicDefinition(sid, lhs, stmt), value)
                    sid += 1
        return self.result


def logic_local_value_numbering(
    plan: Plan, bindings: dict[Alias, TensorFType]
) -> LogicLocalValueNumberingResult:
    """Locally number query definitions and equivalent stored values without rewriting.

    Inputs must be aliased Logic with defined RHS reads. Query IDs start at zero;
    ``uses[sid]`` lists query-entry definition/version IDs in RHS traversal order,
    including repeated references. For an opaque RHS, these are not per-read facts
    across nested effects. IDs and value numbers are local to this invocation.

    Recognizes ordered numeric add/subtract/multiply, scalar literals, copies and
    rank-preserving reorders. External writes invalidate every external binding;
    unsupported expressions conservatively invalidate all current tensor facts.
    Materialization boundaries are retained, so a copy of an external tensor or a
    literal need not share its source's number. Historical equivalence is not CSE
    availability, storage identity, or equivalence across separate plan executions.
    """
    if not isinstance(plan, Plan):
        raise TypeError("Value numbering requires a Logic Plan")
    return _LogicLocalValueNumbering(bindings).analyze(plan)
