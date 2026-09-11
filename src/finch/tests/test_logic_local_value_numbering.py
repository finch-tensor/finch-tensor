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

    expr = MapJoin(Literal(ffuncs.add), (Literal(IntWithEffects(1)), Literal(2)))
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
        np.full(shape, fill_value, dtype=dtype(np.int64(0)).dtype),
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
