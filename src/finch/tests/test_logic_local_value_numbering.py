from dataclasses import FrozenInstanceError, dataclass
from typing import Any, cast

import pytest

import numpy as np

import finch
from finch.algebra import DynamicFill, ffuncs, ftype
from finch.autoschedule.executor import extract_tensors
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicInterpreter,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
    logic_local_value_numbering,
)
from finch.finch_logic.local_value_numbering import (
    _LogicLocalValueNumbering,
    _ValuePool,
)
from finch.tensor import BufferizedNDArray

A, B, C, D, X, Y, Z = map(Alias, ("A", "B", "C", "D", "X", "Y", "Z"))
i, j = Field("i"), Field("j")


def table(alias, fields=(i,)):
    return Table(alias, fields)


def add(a=A, b=B, fields=(i,)):
    return MapJoin(Literal(ffuncs.add), (table(a, fields), table(b, fields)))


@pytest.fixture
def bindings():
    tp = ftype(BufferizedNDArray.from_numpy(np.zeros(3, dtype=np.int64)))
    return {A: tp, B: tp, C: tp}


def analyze(*stmts, bindings):
    return logic_local_value_numbering(Plan(stmts), bindings)


def test_bindings_need_only_tensor_rank():
    @dataclass(frozen=True)
    class Rank:
        ndim: int

    result = analyze(
        Query(X, add()), Query(Y, add()), bindings={A: Rank(1), B: Rank(1)}
    )

    assert result.value_numbers[0] == result.value_numbers[1]
    assert len(set(result.uses[0])) == 2
    with pytest.raises(ValueError, match="Table rank"):
        analyze(Query(X, table(A, ())), bindings={A: Rank(1)})


@pytest.mark.parametrize("value,expected", [(1, 3), (1.5, 3.5), (1 + 2j, 3 + 2j)])
def test_materialized_builtin_scalar_results_can_be_reused(value, expected):
    expr = MapJoin(Literal(ffuncs.add), (table(X, ()), Literal(2)))
    plan = Plan(
        (Query(X, Literal(value)), Query(Y, expr), Query(Z, expr), Produces((Y, Z)))
    )
    result = logic_local_value_numbering(plan, {})

    assert result.value_numbers[1] == result.value_numbers[2]
    assert result.uses[1] == result.uses[2] == (0,)
    y, z = LogicInterpreter(make_tensor=make_tensor)(plan)
    np.testing.assert_equal(y.to_numpy(), expected)
    np.testing.assert_equal(z.to_numpy(), expected)
    assert y is not z


def test_analysis_does_not_execute_arithmetic_or_type_inference(bindings, monkeypatch):
    def unexpected(*args):
        raise AssertionError("Unexpected arithmetic or type inference")

    monkeypatch.setattr(ffuncs.add, "return_type", unexpected)
    monkeypatch.setattr(type(ffuncs.add), "__call__", unexpected)

    result = analyze(Query(X, add()), Query(Y, add()), bindings=bindings)

    assert result.value_numbers[0] == result.value_numbers[1]


def test_value_pool_reuses_equal_literals_without_consuming_ids():
    pool = _ValuePool()
    first = pool.literal(int, 1)
    second = pool.literal(int, 1)
    fresh = pool.fresh()

    assert second is first
    assert (first.id, fresh.id) == (0, 1)


def test_value_pool_distinguishes_literal_type_and_colliding_payloads():
    pool = _ValuePool()
    int_one = pool.literal(int, 1)
    numpy_one = pool.literal(np.int64, np.asarray(np.int64(1)).tobytes())
    same_payload_unsigned = pool.literal(np.uint64, np.asarray(np.int64(1)).tobytes())
    minus_one = pool.literal(int, -1)
    minus_two = pool.literal(int, -2)

    assert (
        len(
            {
                node.id
                for node in (
                    int_one,
                    numpy_one,
                    same_payload_unsigned,
                    minus_one,
                    minus_two,
                )
            }
        )
        == 5
    )
    assert same_payload_unsigned is not numpy_one
    assert hash(minus_one) == hash(minus_two)
    assert pool.literal(int, -1) is minus_one
    assert pool.literal(int, -2) is minus_two


def test_value_pool_distinguishes_nan_payloads():
    pool = _ValuePool()
    bits_a = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)[0]
    bits_b = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)[0]

    first = pool.literal(np.float64, np.asarray(bits_a).tobytes())
    repeat = pool.literal(np.float64, np.asarray(bits_a).tobytes())
    changed = pool.literal(np.float64, np.asarray(bits_b).tobytes())

    assert repeat is first
    assert changed is not first


def test_value_pool_derived_key_parts_are_significant():
    pool = _ValuePool()
    left = pool.fresh()
    right = pool.fresh()

    base = pool.derived(
        MapJoin,
        (left, right),
        op=ffuncs.add,
        metadata=((0,), (0,)),
    )
    assert (
        pool.derived(
            MapJoin,
            (left, right),
            op=ffuncs.add,
            metadata=((0,), (0,)),
        )
        is base
    )
    fresh_after_lookup = pool.fresh()
    assert fresh_after_lookup.id == base.id + 1
    variants = (
        pool.derived(Query, (left, right), op=ffuncs.add),
        pool.derived(MapJoin, (left, right), op=ffuncs.sub),
        pool.derived(MapJoin, (right, left), op=ffuncs.add),
        pool.derived(
            MapJoin,
            (left, right),
            op=ffuncs.add,
            metadata=((0,), (1,)),
        ),
        pool.derived(
            MapJoin,
            (left,),
            op=ffuncs.add,
            metadata=((0,), (0,)),
        ),
        pool.derived(Reorder, (left,), metadata=((1, 0),)),
        pool.derived(Reorder, (left,), metadata=((0, 1),)),
        pool.derived(Query, (left,)),
    )

    assert all(node is not base for node in variants)
    assert len({node.id for node in (base, *variants)}) == 9


def test_value_pool_rejects_foreign_operands_with_matching_ids():
    pool = _ValuePool()
    other = _ValuePool()
    local = pool.fresh()
    foreign = other.fresh()

    assert local.id == foreign.id
    with pytest.raises(ValueError, match="different pool"):
        pool.derived(Query, (foreign,))
    local_literal = pool.literal(int, 1)
    foreign_literal = other.literal(int, 1)
    local_derived = pool.derived(Query, (local_literal,))
    foreign_derived = other.derived(Query, (foreign_literal,))

    assert local_literal is not foreign_literal
    assert local_derived is not foreign_derived


def test_value_pool_nodes_are_frozen_and_fresh_nodes_are_distinct():
    pool = _ValuePool()
    first = cast(Any, pool.fresh())
    second = pool.fresh()
    literal = cast(Any, pool.literal(int, 1))
    derived = cast(Any, pool.derived(Query, (first,), metadata=("base",)))

    assert first is not second
    assert first != second
    with pytest.raises(FrozenInstanceError):
        first.id = 99
    with pytest.raises(FrozenInstanceError):
        literal.payload = 2
    with pytest.raises(FrozenInstanceError):
        derived.metadata = ("changed",)


def test_value_pool_derived_nodes_do_not_recurse_through_deep_graph():
    pool = _ValuePool()
    node = pool.fresh()
    for depth in range(1500):
        node = pool.derived(Query, (node,), metadata=(depth,))

    first = pool.derived(Query, (node,))
    second = pool.derived(Query, (node,))

    assert second is first


def test_analysis_reuses_expression_and_materialization_nodes(bindings):
    analysis = _LogicLocalValueNumbering(bindings)
    result = analysis.analyze(
        Plan((Query(X, add()), Query(Y, add()), Query(Z, table(X))))
    )

    assert result.value_numbers[0] == result.value_numbers[1] == result.value_numbers[2]
    assert analysis.values[X].node is analysis.values[Y].node is analysis.values[Z].node
    assert result.definitions[0].query is not result.definitions[1].query


def test_analysis_keeps_inputs_and_invalidations_fresh(bindings):
    analysis = _LogicLocalValueNumbering(bindings)
    input_a = analysis.values[A].node
    input_b = analysis.values[B].node
    result = analysis.analyze(
        Plan((Query(X, add()), Query(A, table(C)), Query(Y, add())))
    )

    invalidated_a = analysis.values[A].node
    assert input_a is not input_b
    assert invalidated_a is not input_a
    assert invalidated_a.id == result.value_numbers[1]
    assert result.value_numbers[0] != result.value_numbers[2]


def test_equivalent_definitions_and_copies(bindings):
    result = analyze(
        Query(X, add()), Query(Y, table(X)), Query(Z, add()), bindings=bindings
    )
    assert result.value_numbers[0] == result.value_numbers[1] == result.value_numbers[2]
    assert result.equivalence_classes[result.value_numbers[0]] == (0, 1, 2)
    assert result.uses[1] == (0,)
    assert result.definitions[0].query.lhs == X
    assert result.definitions[1].query.lhs == Y


def test_number_occurrences_not_node_objects(bindings):
    stmt = Query(X, add())
    result = analyze(stmt, Plan((stmt,)), bindings=bindings)
    assert result.definitions[0].query is result.definitions[1].query
    assert result.value_numbers[0] == result.value_numbers[1]
    assert result.current_definitions[X] == 1


def test_rhs_reads_precede_redefinition(bindings):
    result = analyze(
        Query(X, add()), Query(A, add(A, A)), Query(Y, add()), bindings=bindings
    )
    a_input, b_input = result.uses[0]
    assert a_input < 0 and b_input < 0
    assert result.uses[1] == (a_input, a_input)
    assert result.uses[2][0] == 1
    assert result.uses[2][1] != b_input
    assert result.value_numbers[0] != result.value_numbers[2]
    assert result.definitions[result.uses[2][1]].invalidated_by == 1


def test_copy_redefinition_kills_equivalence(bindings):
    result = analyze(
        Query(X, add()), Query(A, table(C)), Query(Y, add()), bindings=bindings
    )
    assert result.value_numbers[0] != result.value_numbers[2]


def test_independent_local_write_preserves_equivalence(bindings):
    result = analyze(
        Query(D, table(C)),
        Query(X, add()),
        Query(D, add(D, D)),
        Query(Y, add()),
        bindings=bindings,
    )
    assert result.value_numbers[1] == result.value_numbers[3]
    assert result.value_numbers[0] != result.value_numbers[2]


def test_copies_are_independent_snapshots(bindings):
    result = analyze(
        Query(X, add()),
        Query(Y, table(X)),
        Query(X, table(C)),
        Query(Z, table(Y)),
        bindings=bindings,
    )
    assert result.value_numbers[0] == result.value_numbers[1] == result.value_numbers[3]
    assert result.value_numbers[2] != result.value_numbers[3]
    assert result.uses[3] == (1,)


def test_external_writes_version_all_possible_aliases(bindings):
    result = analyze(
        Query(X, add()),
        Query(C, table(A)),
        Query(Y, add()),
        Query(C, table(A)),
        Query(Z, add()),
        bindings=bindings,
    )
    assert len({result.value_numbers[sid] for sid in (0, 2, 4)}) == 3
    for sid in (0, 2, 4):
        assert len(set(result.uses[sid])) == 2
    assert len({result.uses[sid][0] for sid in (0, 2, 4)}) == 3


@pytest.mark.parametrize("nested", [False, True])
def test_unknown_calls_are_not_executed_and_invalidate_locals(bindings, nested):
    def unknown():
        raise AssertionError("The analysis must not execute this operator")

    call = MapJoin(Literal(unknown), ())
    rhs = MapJoin(Literal(ffuncs.add), (call, table(A))) if nested else call
    result = analyze(
        Query(D, table(A)),
        Query(X, add(D, B)),
        Query(C, rhs),
        Query(Y, add(D, B)),
        bindings=bindings,
    )
    assert result.value_numbers[1] != result.value_numbers[3]
    assert result.uses[3][0] != 0
    assert result.definitions[result.uses[3][0]].invalidated_by == 2


def test_scalar_subclasses_are_opaque():
    class IntWithEffects(int):
        def __add__(self, other):
            raise AssertionError("custom numeric methods must not run")

    class NumpyIntWithEffects(np.int64):
        def __add__(self, other):
            raise AssertionError("custom numpy scalar methods must not run")

        def __eq__(self, other):
            raise AssertionError("custom numpy scalar equality must not run")

        def __hash__(self):
            raise AssertionError("custom numpy scalar hash must not run")

    expressions = (
        MapJoin(Literal(ffuncs.add), (Literal(IntWithEffects(1)), Literal(2))),
        MapJoin(Literal(ffuncs.add), (Literal(NumpyIntWithEffects(1)), Literal(2))),
    )
    for expr in expressions:
        result = analyze(Query(X, expr), Query(Y, expr), bindings={})
        assert result.value_numbers[0] != result.value_numbers[1]


def test_literal_encoding_does_not_call_scalar_metaclass_hooks():
    class ScalarMeta(type):
        def __hash__(cls):
            raise AssertionError("custom scalar class hashing must not run")

    class CustomScalar(np.int64, metaclass=ScalarMeta):
        pass

    expr = Literal(CustomScalar(1))
    result = analyze(Query(X, expr), Query(Y, expr), bindings={})

    assert result.value_numbers[0] != result.value_numbers[1]


def test_literal_encoding_does_not_inspect_opaque_instance_class():
    class Opaque:
        def __getattribute__(self, name):
            if name == "__class__":
                raise AssertionError("custom instance class lookup must not run")
            return super().__getattribute__(name)

    expr = Literal(Opaque())
    result = analyze(Query(X, expr), Query(Y, expr), bindings={})

    assert result.value_numbers[0] != result.value_numbers[1]


def test_field_alpha_renaming_and_frontend_reorders(bindings):
    k = Field("fresh")
    wrapped = Reorder(
        MapJoin(Literal(ffuncs.add), (Reorder(table(A, (k,)), (k,)), table(B, (k,)))),
        (k,),
    )
    result = analyze(Query(X, add()), Query(Y, wrapped), bindings=bindings)
    assert result.value_numbers[0] == result.value_numbers[1]


def test_axis_mapping_is_part_of_expression_key():
    tp = ftype(BufferizedNDArray.from_numpy(np.zeros((2, 2), dtype=np.int64)))
    expr = add(fields=(i, j))
    transposed_b = MapJoin(Literal(ffuncs.add), (table(A, (i, j)), table(B, (j, i))))
    result = analyze(
        Query(X, expr),
        Query(Y, transposed_b),
        Query(Z, Reorder(expr, (j, i))),
        bindings={A: tp, B: tp},
    )
    assert len({result.value_numbers[sid] for sid in (0, 1, 2)}) == 3


@pytest.mark.parametrize("op", [ffuncs.sub, ffuncs.mul])
def test_operator_identity_is_part_of_key(bindings, op):
    result = analyze(
        Query(X, add()),
        Query(Y, MapJoin(Literal(op), (table(A), table(B)))),
        bindings=bindings,
    )
    assert result.value_numbers[0] != result.value_numbers[1]


def test_operand_order_is_preserved(bindings):
    result = analyze(Query(X, add()), Query(Y, add(B, A)), bindings=bindings)
    assert result.value_numbers[0] != result.value_numbers[1]


@pytest.mark.parametrize(
    "a,b",
    [
        (1, np.int64(1)),
        (np.float32(1), np.float64(1)),
        (0.0, -0.0),
        (np.float32(0), np.float32(-0.0)),
        (complex(0.0, 0.0), complex(0.0, -0.0)),
    ],
)
def test_literal_type_and_bits_are_preserved(a, b):
    result = analyze(Query(X, Literal(a)), Query(Y, Literal(b)), bindings={})
    assert result.value_numbers[0] != result.value_numbers[1]


@pytest.mark.parametrize(
    "dtype",
    ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16"],
)
def test_fixed_width_numpy_literals_are_canonical(dtype):
    scalar = np.dtype(dtype).type
    result = analyze(
        Query(X, Literal(scalar(1))),
        Query(Y, Literal(scalar(1))),
        Query(Z, Literal(scalar(2))),
        bindings={},
    )

    assert result.value_numbers[0] == result.value_numbers[1]
    assert result.value_numbers[0] != result.value_numbers[2]


def test_literal_materialization_has_its_own_value():
    expr = MapJoin(Literal(ffuncs.add), (Literal(1), Literal(2)))
    result = analyze(
        Query(X, Literal(1)),
        Query(Y, expr),
        Query(Z, MapJoin(Literal(ffuncs.add), (table(X, ()), Literal(2)))),
        bindings={},
    )
    assert result.value_numbers[1] != result.value_numbers[2]


@pytest.mark.parametrize(
    "fills",
    [
        (DynamicFill(np.float64(1)), DynamicFill(np.float64(2))),
        (0.0, -0.0),
    ],
)
def test_equal_ftypes_do_not_identify_input_contents_or_fills(fills):
    a, b = (BufferizedNDArray.from_numpy(np.zeros(3), fill_value=f) for f in fills)
    result = analyze(
        Query(X, table(A)), Query(Y, table(B)), bindings={A: ftype(a), B: ftype(b)}
    )
    assert result.value_numbers[0] != result.value_numbers[1]


def test_unsupported_reduction_and_broadcast_are_opaque(bindings):
    reduction = Aggregate(Literal(ffuncs.add), Literal(np.int64(0)), table(A), (i,))
    broadcast = Reorder(table(A), (i, j))
    for expr in (reduction, broadcast):
        result = analyze(Query(X, expr), Query(Y, expr), bindings=bindings)
        assert result.value_numbers[0] != result.value_numbers[1]


@pytest.mark.parametrize(
    "expr,message",
    [
        (table(X), "Undefined tensor alias"),
        (table(A, ()), "Table rank"),
        (table(A, (i, i)), "fields must be distinct"),
        (Table(Literal(np.zeros(3)), (i,)), "Extract tensor literals"),
    ],
)
def test_invalid_reads_rejected_before_lhs_definition(bindings, expr, message):
    with pytest.raises(ValueError, match=message):
        analyze(Query(X, expr), bindings=bindings)


def test_produces_stops_nested_plan(bindings):
    result = analyze(
        Plan((Query(X, add()), Produces((X,)))),
        Query(Y, table(Z)),
        bindings=bindings,
    )
    assert tuple(result.uses) == (0,)
    with pytest.raises(ValueError, match="Undefined tensor alias"):
        analyze(Produces((Z,)), bindings=bindings)


def test_analysis_does_not_change_ir_or_bindings(bindings):
    plan = Plan((Query(X, add()), Query(A, add(A, A)), Produces((X,))))
    original_bindings = bindings.copy()
    before = repr(plan)
    result = logic_local_value_numbering(plan, bindings)
    assert repr(plan) == before
    assert bindings == original_bindings
    assert logic_local_value_numbering(plan, bindings) == result


def make_tensor(shape, fill_value, *, dtype):
    # The default interpreter allocator currently drops nonzero fill metadata.
    return BufferizedNDArray.from_numpy(
        np.full(shape, fill_value, dtype=np.asarray(dtype(np.int64(0))).dtype),
        fill_value=fill_value,
    )


def test_copy_and_inplace_semantics_against_interpreter():
    a = BufferizedNDArray.from_numpy(np.array([1, 2, 3]))
    b = BufferizedNDArray.from_numpy(np.array([4, 5, 6]))
    plan = Plan(
        (
            Query(X, add()),
            Query(Y, table(X)),
            Query(A, add(A, A)),
            Query(Z, add()),
            Produces((X, Y, Z)),
        )
    )
    result = logic_local_value_numbering(plan, {A: ftype(a), B: ftype(b)})
    x, y, z = LogicInterpreter(make_tensor=make_tensor)(plan, {A: a, B: b})
    np.testing.assert_array_equal(x.to_numpy(), [5, 7, 9])
    np.testing.assert_array_equal(y.to_numpy(), x.to_numpy())
    np.testing.assert_array_equal(z.to_numpy(), [6, 9, 12])
    assert x is not y
    assert result.value_numbers[0] == result.value_numbers[1]
    assert result.value_numbers[0] != result.value_numbers[3]


def test_external_alias_mutation_against_interpreter():
    shared = BufferizedNDArray.from_numpy(np.array([1, 2, 3]))
    plan = Plan(
        (Query(X, add()), Query(A, add(A, A)), Query(Y, add()), Produces((X, Y)))
    )
    result = logic_local_value_numbering(plan, {A: ftype(shared), B: ftype(shared)})
    x, y = LogicInterpreter(make_tensor=make_tensor)(plan, {A: shared, B: shared})
    np.testing.assert_array_equal(x.to_numpy(), [2, 4, 6])
    np.testing.assert_array_equal(y.to_numpy(), [4, 8, 12])
    assert result.value_numbers[0] != result.value_numbers[2]


@pytest.mark.parametrize("source", [np.array([9]), np.array([1.5, 2.5, 3.5])])
def test_existing_destination_does_not_adopt_rhs_value(source):
    a = BufferizedNDArray.from_numpy(np.array([1, 2, 3]))
    b = BufferizedNDArray.from_numpy(source)
    plan = Plan(
        (Query(X, table(A)), Query(Y, table(B)), Query(X, table(B)), Produces((X, Y)))
    )
    result = logic_local_value_numbering(plan, {A: ftype(a), B: ftype(b)})
    x, y = LogicInterpreter(make_tensor=make_tensor)(plan, {A: a, B: b})
    assert result.value_numbers[1] != result.value_numbers[2]
    assert not np.array_equal(x.to_numpy(), y.to_numpy())


def test_external_write_does_not_destroy_local_copy(bindings):
    result = analyze(
        Query(X, table(A)), Query(A, table(C)), Query(Y, table(X)), bindings=bindings
    )
    assert result.value_numbers[0] == result.value_numbers[2]


def test_repeated_computations_after_external_write(bindings):
    result = analyze(
        Query(X, add()),
        Query(A, table(C)),
        Query(Y, add()),
        Query(Z, add()),
        bindings=bindings,
    )
    assert result.value_numbers[0] != result.value_numbers[2]
    assert result.value_numbers[2] == result.value_numbers[3]


def test_frontend_generated_elementwise_queries():
    a = finch.defer(BufferizedNDArray.from_numpy(np.arange(3)))
    b = finch.defer(BufferizedNDArray.from_numpy(np.arange(3)))
    x = a + b
    y = a + b
    a = a + a
    z = a + b
    plan = Plan(
        x.ctx.join(y.ctx, z.ctx).trace() + (Produces((x.data, y.data, z.data)),)
    )
    plan, tensors = extract_tensors(plan, {})
    assert isinstance(plan, Plan)
    result = logic_local_value_numbering(
        plan, {alias: ftype(tns) for alias, tns in tensors.items()}
    )
    vns = {
        definition.alias: result.value_numbers[sid]
        for sid, definition in result.definitions.items()
        if definition.query is not None
    }
    assert vns[x.data] == vns[y.data]
    assert vns[x.data] != vns[z.data]
