"""Local value-numbering analysis for sequential, aliased Finch Logic plans.

Adapted from the hash-based local value-numbering approach in Steven S. Muchnick,
Advanced Compiler Design and Implementation (1997), section 12.4.1, pp. 344-348.
Changed operands receive fresh versions; historical expression keys are retained.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from finch.algebra import ffuncs
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


class _RankedTensor(Protocol):
    @property
    def ndim(self) -> int: ...


def _literal_bits(value: Any) -> int | bytes | None:
    scalar_type = type(value)
    if scalar_type is int:
        return value
    if scalar_type is float or scalar_type is complex:
        return np.asarray(value).tobytes()
    if type(scalar_type) is type and issubclass(scalar_type, np.generic):
        dtype = np.dtype(scalar_type)
        # Exclude custom scalars and extended formats with non-value padding.
        if (
            scalar_type is dtype.type
            and dtype.kind in "iufc"
            and dtype.itemsize <= (16 if dtype.kind == "c" else 8)
        ):
            return np.asarray(value).tobytes()
    return None


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

    ``uses[sid]`` lists query-entry definition IDs in RHS traversal order,
    including repeats, not per-read facts across nested effects in opaque RHSs.
    All IDs and value numbers are local to this analysis run.
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


@dataclass(frozen=True, eq=False)
class _NumberedValue:
    id: int = field(compare=False)
    _owner: object = field(compare=False)


@dataclass(frozen=True)
class _LiteralValue(_NumberedValue):
    scalar_type: type
    payload: int | bytes


@dataclass(frozen=True, eq=False)
class _DerivedValue(_NumberedValue):
    kind: type
    operands: tuple[_NumberedValue, ...]
    op: object | None = None
    metadata: tuple = ()

    def __hash__(self) -> int:
        # Canonical operand IDs keep hashing shallow, even for deep graphs.
        return hash(
            (
                self.kind,
                id(self.op),
                tuple(operand.id for operand in self.operands),
                self.metadata,
            )
        )

    def __eq__(self, other: object) -> bool:
        match other:
            case _DerivedValue():
                return (
                    self.kind is other.kind
                    and self.op is other.op
                    and self.metadata == other.metadata
                    and len(self.operands) == len(other.operands)
                    and all(
                        a is b
                        for a, b in zip(self.operands, other.operands, strict=True)
                    )
                )
            case _:
                return NotImplemented


class _ValuePool:
    def __init__(self) -> None:
        self._values: dict[_NumberedValue, _NumberedValue] = {}
        self._next_id = 0
        self._owner = object()

    def fresh(self) -> _NumberedValue:
        value = _NumberedValue(self._next_id, self._owner)
        self._next_id += 1
        return value

    def literal(self, scalar_type: type, payload: int | bytes) -> _NumberedValue:
        return self._intern(
            _LiteralValue(self._next_id, self._owner, scalar_type, payload)
        )

    def derived(
        self,
        kind: type,
        operands: tuple[_NumberedValue, ...],
        *,
        op: object | None = None,
        metadata: tuple = (),
    ) -> _NumberedValue:
        for operand in operands:
            if operand._owner is not self._owner:
                raise ValueError("Cannot derive values from a different pool")
        return self._intern(
            _DerivedValue(self._next_id, self._owner, kind, operands, op, metadata)
        )

    def _intern(self, candidate: _NumberedValue) -> _NumberedValue:
        value = self._values.setdefault(candidate, candidate)
        if value is candidate:
            self._next_id += 1
        return value


@dataclass(frozen=True)
class _Value:
    node: _NumberedValue
    ndim: int | None
    materialized: bool = False

    @property
    def number(self) -> int:
        return self.node.id


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
    def __init__(self, bindings: Mapping[Alias, _RankedTensor]):
        self.result = LogicLocalValueNumberingResult({}, {}, {}, {})
        self.values: dict[Alias, _Value] = {}
        self.external = tuple(bindings)
        self.pool = _ValuePool()
        self.next_version = -1
        for alias, tp in bindings.items():
            value = _Value(self.pool.fresh(), tp.ndim)
            self.version(alias, value)

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
                self.pool.fresh(),
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
                if value.ndim is not None:
                    return _Expression(value, fields)
            case Literal(val):
                # Exact type/bits preserve signed zero and NaN payloads.
                bits = _literal_bits(val)
                if bits is None:
                    return None
                node = self.pool.literal(type(val), bits)
                return _Expression(_Value(node, 0), ())
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
                axis_mappings = tuple(
                    tuple(fields.index(f) for f in arg.fields) for arg in known
                )
                node = self.pool.derived(
                    MapJoin,
                    tuple(arg.value.node for arg in known),
                    op=op,
                    metadata=axis_mappings,
                )
                return _Expression(_Value(node, len(fields)), fields)
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
                node = self.pool.derived(
                    Reorder, (inner.value.node,), metadata=(permutation,)
                )
                return _Expression(_Value(node, len(fields)), fields)
        return None

    def materialize(self, value: _Value) -> _Value:
        # First allocation may change metadata; later copies preserve it.
        if value.materialized:
            return value
        return _Value(self.pool.derived(Query, (value.node,)), value.ndim, True)

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
                        value = _Value(self.pool.fresh(), None)
                    else:
                        value = self.materialize(expr.value)
                    if lhs in self.external:
                        self.invalidate(self.external, sid)
                        value = _Value(
                            self.pool.fresh(),
                            old.ndim if old and expr is not None else None,
                        )
                    elif old is not None and old.number != value.number:
                        # Equal rank does not imply equal extents, dtype or fill.
                        value = _Value(
                            self.pool.fresh(),
                            old.ndim if expr is not None else None,
                        )
                    self.record(LogicDefinition(sid, lhs, stmt), value)
                    sid += 1
        return self.result


def logic_local_value_numbering(
    plan: Plan, bindings: Mapping[Alias, _RankedTensor]
) -> LogicLocalValueNumberingResult:
    """Locally number query definitions and equivalent stored values without rewriting.

    Requires aliased Logic with defined RHS reads and bindings providing ``ndim``.
    Recognized arithmetic, including overloaded scalar operations, must be pure,
    deterministic and well-typed; no purity checks or dtype inference are performed.
    New Query destinations must have independent storage; copies of stored results
    must preserve contents, shape, dtype and fill.

    Recognizes ordered add/subtract/multiply, exactly encodable numeric literals,
    copies and rank-preserving reorders, but not branches, loops or reductions.
    External writes invalidate every external binding; unsupported expressions
    conservatively invalidate all current tensor facts.
    First materialization retains a distinct number from its source. Equivalence
    describes definition snapshots, not current contents, CSE availability or
    shared storage.
    """
    if not isinstance(plan, Plan):
        raise TypeError("Value numbering requires a Logic Plan")
    return _LogicLocalValueNumbering(bindings).analyze(plan)
