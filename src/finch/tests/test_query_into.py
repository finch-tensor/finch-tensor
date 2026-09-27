import pytest

import numpy as np

from finch.algebra import ffuncs
from finch.autoschedule import (
    INTERPRET_LOGIC,
    INTERPRET_NOTATION,
    INTERPRET_NOTATION_GALLEY,
    DefaultLogicFormatter,
    DefaultLoopOrderer,
    LogicCapture,
    LogicCompiler,
)
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.stages import LoopOrderedForm, SingleAggregateForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.util import propagate_copy_queries, resugar_query_into
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
)
from finch.tensor import BufferizedNDArray

from .conftest import finch_assert_equal

i, j, k = Field("i"), Field("j"), Field("k")
A, B, C, D, V = Alias("A"), Alias("B"), Alias("C"), Alias("D"), Alias("V")

A_DATA = np.array([[1.0, 2.0], [3.0, 4.0]])
B_DATA = np.array([[5.0, 6.0], [7.0, 8.0]])
B3_DATA = np.arange(12.0).reshape(2, 2, 3)

SCHEDULERS = [INTERPRET_LOGIC, INTERPRET_NOTATION, INTERPRET_NOTATION_GALLEY]


def bindings():
    return {
        A: BufferizedNDArray.from_numpy(A_DATA.copy()),
        B: BufferizedNDArray.from_numpy(B_DATA.copy()),
    }


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_query_into_updates_produced_tensor(scheduler):
    plan = Plan(
        (
            QueryInto(Table(A, (i, j)), Literal(ffuncs.add), Table(B, (i, j))),
            Produces((A,)),
        )
    )
    (result,) = scheduler(plan, bindings())
    finch_assert_equal(result, A_DATA + B_DATA)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_query_into_update_is_visible_to_later_queries(scheduler):
    # A is updated in place but not produced, so the update must survive until
    # C reads it.
    plan = Plan(
        (
            QueryInto(
                Table(A, (i,)),
                Literal(ffuncs.add),
                Aggregate(Literal(ffuncs.add), Literal(0.0), Table(B, (i, j)), (j,)),
            ),
            Query(
                Table(C, (i,)),
                MapJoin(Literal(ffuncs.mul), (Table(A, (i,)), Literal(2.0))),
            ),
            Produces((C,)),
        )
    )
    binds = {
        A: BufferizedNDArray.from_numpy(np.array([1.0, 2.0])),
        B: BufferizedNDArray.from_numpy(B_DATA.copy()),
    }
    (result,) = scheduler(plan, binds)
    finch_assert_equal(result, 2 * (np.array([1.0, 2.0]) + B_DATA.sum(axis=1)))


def test_query_into_printer():
    stmt = QueryInto(Table(A, (i, j)), Literal(ffuncs.add), Table(B, (j, i)))
    assert str(stmt) == "A[i, j] <<add>>= Table(B, j, i)"


def test_query_into_as_query():
    stmt = QueryInto(Table(A, (i, j)), Literal(ffuncs.add), Table(B, (j, i)))
    assert stmt.as_query() == Query(
        Table(A, (i, j)),
        Aggregate(Literal(ffuncs.add), Table(A, (i, j)), Table(B, (j, i)), ()),
    )


def test_query_into_as_query_reduces_fields_not_in_lhs():
    stmt = QueryInto(Table(A, (i,)), Literal(ffuncs.add), Table(B, (j, k, i)))
    assert stmt.as_query() == Query(
        Table(A, (i,)),
        Aggregate(Literal(ffuncs.add), Table(A, (i,)), Table(B, (j, k, i)), (j, k)),
    )


@pytest.mark.parametrize(
    "rhs, valid",
    [
        (Table(B, (j, i)), True),
        (MapJoin(Literal(ffuncs.mul), (Table(B, (i, j)), Literal(2.0))), True),
        # The field k is not in the output, so it is reduced with add.
        (Table(B, (i, k, j)), True),
        # A QueryInto is itself a reduction, so it holds no aggregates.
        (
            Aggregate(Literal(ffuncs.add), Literal(0.0), Table(B, (i, k, j)), (k,)),
            False,
        ),
        (Aggregate(Literal(ffuncs.mul), Literal(1.0), Table(B, (i, j)), ()), False),
    ],
)
def test_single_aggregate_form_query_into(rhs, valid):
    plan = Plan((QueryInto(Table(A, (i, j)), Literal(ffuncs.add), rhs), Produces((A,))))
    ftypes = {var: tns.ftype for var, tns in bindings().items()}
    if valid:
        SingleAggregateForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    else:
        with pytest.raises(ValueError):
            SingleAggregateForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_single_aggregate_form_rejects_tensor_init():
    plan = Plan(
        (
            Query(
                Table(C, (i, j)),
                Aggregate(Literal(ffuncs.add), Table(A, (i, j)), Table(B, (i, j)), ()),
            ),
            Produces((C,)),
        )
    )
    ftypes = {var: tns.ftype for var, tns in bindings().items()}
    with pytest.raises(ValueError, match="literal"):
        SingleAggregateForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_query_into_compiles_without_factorizing():
    # An in-place update is already in the normal form of the loop orderer. It
    # loops in the order of the updated table, and B is transposed to match.
    plan = Plan(
        (
            QueryInto(Table(A, (i, j)), Literal(ffuncs.add), Table(B, (j, i))),
            Produces((A,)),
        )
    )
    capture = LogicCapture()
    stats_factory = DenseStatsFactory()
    ftypes = {var: tns.ftype for var, tns in bindings().items()}
    DefaultLoopOrderer(capture)(plan, ftypes, {}, stats_factory)
    ordered = capture.last_prgm
    assert isinstance(ordered, Plan)
    LoopOrderedForm.validate_inputs(ordered, capture.last_bindings, {}, stats_factory)

    scheduler = LogicExecutor(
        DefaultLoopOrderer(DefaultLogicFormatter(LogicCompiler()))
    )
    (result,) = scheduler(plan, bindings())
    finch_assert_equal(result, A_DATA + B_DATA.T)


def test_query_into_reduction_compiles_without_factorizing():
    # A[i] += B[i, j], which reduces j directly into A.
    plan = Plan(
        (
            QueryInto(Table(A, (i,)), Literal(ffuncs.add), Table(B, (j, i))),
            Produces((A,)),
        )
    )
    capture = LogicCapture()
    stats_factory = DenseStatsFactory()
    ftypes = {
        A: BufferizedNDArray.from_numpy(np.zeros(2)).ftype,
        B: bindings()[B].ftype,
    }
    DefaultLoopOrderer(capture)(plan, ftypes, {}, stats_factory)
    ordered = capture.last_prgm
    assert isinstance(ordered, Plan)
    LoopOrderedForm.validate_inputs(ordered, capture.last_bindings, {}, stats_factory)

    scheduler = LogicExecutor(
        DefaultLoopOrderer(DefaultLogicFormatter(LogicCompiler()))
    )
    binds = {
        A: BufferizedNDArray.from_numpy(np.array([1.0, 2.0])),
        B: BufferizedNDArray.from_numpy(B_DATA.copy()),
    }
    (result,) = scheduler(plan, binds)
    finch_assert_equal(result, np.array([1.0, 2.0]) + B_DATA.T.sum(axis=1))


def test_copy_propagation_keeps_copies_updated_in_place():
    # C starts as a copy of A, but is then updated, so it needs its own storage.
    plan = Plan(
        (
            Query(Table(C, (i, j)), Table(A, (i, j))),
            QueryInto(Table(C, (i, j)), Literal(ffuncs.add), Table(B, (i, j))),
            Produces((C,)),
        )
    )
    assert propagate_copy_queries(plan, {A: None, B: None}) == plan


@pytest.mark.parametrize("scheduler", SCHEDULERS)
@pytest.mark.parametrize(
    "op, expected",
    [
        (ffuncs.add, np.array([1.0, 2.0]) + B_DATA.sum(axis=1)),
        (ffuncs.mul, np.array([1.0, 2.0]) * B_DATA.prod(axis=1)),
        (ffuncs.max, np.maximum(np.array([1.0, 2.0]), B_DATA.max(axis=1))),
    ],
)
def test_query_into_reduces_fields_not_in_lhs(scheduler, op, expected):
    plan = Plan(
        (
            QueryInto(Table(A, (i,)), Literal(op), Table(B, (i, j))),
            Produces((A,)),
        )
    )
    binds = {
        A: BufferizedNDArray.from_numpy(np.array([1.0, 2.0])),
        B: BufferizedNDArray.from_numpy(B_DATA.copy()),
    }
    (result,) = scheduler(plan, binds)
    finch_assert_equal(result, expected)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_aggregate_with_tensor_init(scheduler):
    # D[i] = sum_j A[i, j] + sum_j B[j, i], where the second sum starts from
    # the first.
    plan = Plan(
        (
            Query(
                Table(C, (i,)),
                Aggregate(Literal(ffuncs.add), Literal(0.0), Table(A, (i, j)), (j,)),
            ),
            Query(
                Table(D, (i,)),
                Aggregate(Literal(ffuncs.add), Table(C, (i,)), Table(B, (j, i)), (j,)),
            ),
            Produces((D,)),
        )
    )
    (result,) = scheduler(plan, bindings())
    finch_assert_equal(result, A_DATA.sum(axis=1) + B_DATA.sum(axis=0))


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_aggregate_broadcasts_tensor_init(scheduler):
    # The init has fewer fields than the result, so it is broadcast over j.
    plan = Plan(
        (
            Query(
                Table(C, (i, j)),
                Aggregate(
                    Literal(ffuncs.add), Table(V, (i,)), Table(B, (i, j, k)), (k,)
                ),
            ),
            Produces((C,)),
        )
    )
    v = np.array([10.0, 20.0])
    binds = {
        V: BufferizedNDArray.from_numpy(v),
        B: BufferizedNDArray.from_numpy(B3_DATA.copy()),
    }
    (result,) = scheduler(plan, binds)
    finch_assert_equal(result, v[:, None] + B3_DATA.sum(axis=2))


def test_resugar_query_into():
    agg = Aggregate(Literal(ffuncs.add), Table(A, (i,)), Table(B, (i, j)), (j,))
    # An aggregate which starts from the table it writes is an in-place update.
    plan = Plan((Query(Table(A, (i,)), agg), Produces((A,))))
    assert resugar_query_into(plan) == Plan(
        (
            QueryInto(Table(A, (i,)), Literal(ffuncs.add), Table(B, (i, j))),
            Produces((A,)),
        )
    )
    # Otherwise, the init is copied into the output first.
    plan = Plan((Query(Table(C, (i,)), agg), Produces((C,))))
    assert resugar_query_into(plan) == Plan(
        (
            Plan(
                (
                    Query(Table(C, (i,)), Table(A, (i,))),
                    QueryInto(Table(C, (i,)), Literal(ffuncs.add), Table(B, (i, j))),
                )
            ),
            Produces((C,)),
        )
    )


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_query_drops_unit_dims(scheduler):
    plan = Plan((Query(Table(C, (j,)), Table(B, (i, j))), Produces((C,))))
    binds = {B: BufferizedNDArray.from_numpy(B_DATA[:1].copy())}
    (result,) = scheduler(plan, binds)
    finch_assert_equal(result, B_DATA[0])
