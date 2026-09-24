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
from finch.autoschedule.util import propagate_copy_queries
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
    Reorder,
    Table,
)
from finch.tensor import BufferizedNDArray

from .conftest import finch_assert_equal

i, j, k = Field("i"), Field("j"), Field("k")
A, B, C = Alias("A"), Alias("B"), Alias("C")

A_DATA = np.array([[1.0, 2.0], [3.0, 4.0]])
B_DATA = np.array([[5.0, 6.0], [7.0, 8.0]])

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
        MapJoin(
            Literal(ffuncs.add),
            (Table(A, (i, j)), Reorder(Table(B, (j, i)), (i, j))),
        ),
    )


@pytest.mark.parametrize(
    "rhs, valid",
    [
        (Table(B, (j, i)), True),
        (MapJoin(Literal(ffuncs.mul), (Table(B, (i, j)), Literal(2.0))), True),
        # A reduction folded into the output from the identity of add.
        (
            Aggregate(Literal(ffuncs.add), Literal(0.0), Table(B, (i, k, j)), (k,)),
            True,
        ),
        # A reduction which doesn't start from an identity of add.
        (Aggregate(Literal(ffuncs.add), Literal(7.0), Table(B, (i, j)), ()), False),
        # A reduction with a different operator than the update.
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
    # A[i] += sum_j B[i, j], folded directly into A.
    plan = Plan(
        (
            QueryInto(
                Table(A, (i,)),
                Literal(ffuncs.add),
                Aggregate(Literal(ffuncs.add), Literal(0.0), Table(B, (i, j)), (j,)),
            ),
            Produces((A,)),
        )
    )
    scheduler = LogicExecutor(
        DefaultLoopOrderer(DefaultLogicFormatter(LogicCompiler()))
    )
    binds = {
        A: BufferizedNDArray.from_numpy(np.array([1.0, 2.0])),
        B: BufferizedNDArray.from_numpy(B_DATA.copy()),
    }
    (result,) = scheduler(plan, binds)
    finch_assert_equal(result, np.array([1.0, 2.0]) + B_DATA.sum(axis=1))


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
