import pytest

import numpy as np

import finch.finch_notation as ntn
from finch.algebra import ffuncs
from finch.autoschedule import NotationGenerator
from finch.autoschedule.compiler import to_compiler_form
from finch.autoschedule.stages import CompilerForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
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
from finch.symbolic import PostOrderDFS
from finch.tensor import BufferizedNDArray

from .conftest import finch_assert_equal

i, j = Field("i"), Field("j")
A, B, C, D = Alias("A"), Alias("B"), Alias("C"), Alias("D")
OVERWRITE, ADD, MUL = (
    Literal(ffuncs.overwrite),
    Literal(ffuncs.add),
    Literal(ffuncs.mul),
)

A_DATA = np.array([[1.0, 2.0], [3.0, 4.0]])
B_DATA = np.array([[5.0, 6.0], [7.0, 8.0]])


def bindings(c=(0.0, 0.0), d=(0.0, 0.0)):
    return {
        A: BufferizedNDArray.from_numpy(A_DATA.copy()),
        B: BufferizedNDArray.from_numpy(B_DATA.copy()),
        C: BufferizedNDArray.from_numpy(np.array(c)),
        D: BufferizedNDArray.from_numpy(np.array(d)),
    }


def lower(plan, binds):
    return NotationGenerator()(
        plan, {var: tns.ftype for var, tns in binds.items()}, {}, None
    )


def lifecycle(program):
    """The declarations, thaws, and freezes in `program`, in order."""
    events = []
    for node in PostOrderDFS(program):
        match node:
            case (
                ntn.Declare(ntn.Slot(name, _), _, ntn.Literal(op), _)
                | ntn.Thaw(ntn.Slot(name, _), ntn.Literal(op))
                | ntn.Freeze(ntn.Slot(name, _), ntn.Literal(op))
            ):
                var = name.split("#")[1].removeprefix("_")
                events.append((var, type(node).__name__, op))
    return events


def run(program, binds):
    return ntn.NotationInterpreter()(program).main(*binds.values())[0].to_numpy()


def test_folds_with_the_same_op_share_a_declaration():
    plan = Plan(
        (
            QueryInto(Table(C, (i,)), OVERWRITE, Literal(0.0)),
            QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i, j))),
            QueryInto(Table(C, (i,)), ADD, Reorder(Table(B, (i, j)), (i, j))),
            Produces((C,)),
        )
    )
    binds = bindings()
    program = lower(plan, binds)
    assert lifecycle(program) == [
        ("C", "Declare", ffuncs.add),
        ("C", "Freeze", ffuncs.add),
    ]
    finch_assert_equal(run(program, binds), A_DATA.sum(axis=1) + B_DATA.sum(axis=1))


def test_tensor_is_frozen_before_it_is_read():
    plan = Plan(
        (
            QueryInto(Table(C, (i,)), OVERWRITE, Literal(0.0)),
            QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i, j))),
            QueryInto(
                Table(D, (i,)),
                OVERWRITE,
                Reorder(MapJoin(MUL, (Table(C, (i,)), Literal(2.0))), (i,)),
            ),
            Produces((D,)),
        )
    )
    binds = bindings()
    program = lower(plan, binds)
    assert lifecycle(program) == [
        ("C", "Declare", ffuncs.add),
        ("C", "Freeze", ffuncs.add),
        ("D", "Declare", ffuncs.init_write(0.0)),
        ("D", "Freeze", ffuncs.init_write(0.0)),
    ]
    finch_assert_equal(run(program, binds), 2 * A_DATA.sum(axis=1))


def test_changing_op_refreezes_and_thaws():
    plan = Plan(
        (
            QueryInto(Table(C, (i,)), OVERWRITE, Literal(1.0)),
            QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i, j))),
            QueryInto(Table(C, (i,)), MUL, Reorder(Table(B, (i, j)), (i, j))),
            Produces((C,)),
        )
    )
    binds = bindings()
    program = lower(plan, binds)
    assert lifecycle(program) == [
        ("C", "Declare", ffuncs.add),
        ("C", "Freeze", ffuncs.add),
        ("C", "Thaw", ffuncs.mul),
        ("C", "Freeze", ffuncs.mul),
    ]
    finch_assert_equal(
        run(program, binds), (1.0 + A_DATA.sum(axis=1)) * B_DATA.prod(axis=1)
    )


def test_unused_initialization_is_declared_when_produced():
    plan = Plan((QueryInto(Table(C, (i,)), OVERWRITE, Literal(3.0)), Produces((C,))))
    binds = bindings()
    program = lower(plan, binds)
    assert lifecycle(program) == [
        ("C", "Declare", ffuncs.overwrite),
        ("C", "Freeze", ffuncs.overwrite),
    ]
    finch_assert_equal(run(program, binds), np.array([3.0, 3.0]))


def test_folding_into_a_tensor_without_initializing_it_thaws_it():
    plan = Plan(
        (
            QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i, j))),
            Produces((C,)),
        )
    )
    binds = bindings(c=(1.0, 2.0))
    program = lower(plan, binds)
    assert lifecycle(program) == [
        ("C", "Thaw", ffuncs.add),
        ("C", "Freeze", ffuncs.add),
    ]
    finch_assert_equal(run(program, binds), np.array([1.0, 2.0]) + A_DATA.sum(axis=1))


@pytest.mark.parametrize(
    "stmt, valid",
    [
        (QueryInto(Table(C, (i,)), OVERWRITE, Literal(0.0)), True),
        (QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i, j))), True),
        # A transpose loops over the table in the order it is stored.
        (QueryInto(Table(C, (i,)), OVERWRITE, Table(A, (j, i))), True),
        (QueryInto(Table(C, (i,)), ADD, Table(A, (i, j))), True),
        # Queries must be split into an initialization and a fold.
        (
            Query(
                Table(C, (i,)),
                Aggregate(ADD, Literal(0.0), Reorder(Table(A, (i, j)), (i, j)), (j,)),
            ),
            False,
        ),
        # Initializations overwrite.
        (QueryInto(Table(C, (i,)), ADD, Literal(1.0)), False),
        # A tensor is either being read or being written.
        (
            QueryInto(
                Table(C, (i,)),
                ADD,
                Reorder(MapJoin(ADD, (Table(C, (i,)), Table(A, (i, j)))), (i, j)),
            ),
            False,
        ),
        # Folds need a loop order, which visits the output in order.
        (QueryInto(Table(C, (i,)), ADD, MapJoin(ADD, (Table(A, (i, j)),))), False),
        (QueryInto(Table(D, (j, i)), ADD, Reorder(Table(A, (i, j)), (i, j))), False),
    ],
)
def test_compiler_form(stmt, valid):
    binds = {
        **bindings(),
        D: BufferizedNDArray.from_numpy(np.zeros((2, 2))),
    }
    plan = Plan((stmt, Produces((C,))))
    ftypes = {var: tns.ftype for var, tns in binds.items()}
    if valid:
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    else:
        with pytest.raises(ValueError):
            CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_to_compiler_form():
    arg = Reorder(Table(A, (i, j)), (i, j))
    plan = Plan(
        (
            Query(Table(C, (i,)), Aggregate(ADD, Literal(0.0), arg, (j,))),
            Query(Table(D, (j, i)), Table(A, (i, j))),
            Produces((C, D)),
        )
    )
    assert to_compiler_form(plan) == Plan(
        (
            QueryInto(Table(C, (i,)), OVERWRITE, Literal(0.0)),
            QueryInto(Table(C, (i,)), ADD, arg),
            QueryInto(Table(D, (j, i)), OVERWRITE, Table(A, (i, j))),
            Produces((C, D)),
        )
    )
