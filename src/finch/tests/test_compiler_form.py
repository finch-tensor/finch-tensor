from itertools import permutations

import pytest

import numpy as np

import finch as fl
import finch.finch_notation as ntn
from finch.algebra import ffuncs
from finch.autoschedule import (
    CompilerFormLowerer,
    DefaultLoopOrderer,
    LogicCapture,
    NotationGenerator,
)
from finch.autoschedule.stages import CompilerForm, FormattedForm, LoopOrderedForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.compile import NotationCompiler
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
        # A transpose has an explicit loop order, including the lhs in order.
        (
            QueryInto(Table(D, (j, i)), OVERWRITE, Reorder(Table(A, (i, j)), (j, i))),
            True,
        ),
        (QueryInto(Table(C, (i,)), OVERWRITE, Table(A, (j, i))), False),
        (QueryInto(Table(C, (i,)), ADD, Table(A, (i, j))), False),
        # Fields cannot be omitted or revisited by the explicit loop order.
        (QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i,))), False),
        (QueryInto(Table(C, (i,)), ADD, Reorder(Table(A, (i, j)), (i, j, j))), False),
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


def test_compiler_form_lowerer():
    arg = Reorder(Table(A, (i, j)), (i, j))
    plan = Plan(
        (
            Query(Table(C, (i,)), Aggregate(ADD, Literal(0.0), arg, (j,))),
            Query(Table(D, (j, i)), Table(A, (i, j))),
            Produces((C, D)),
        )
    )
    capture = LogicCapture()
    binds = bindings(d=np.zeros((2, 2)))
    _, _, _, original = CompilerFormLowerer(capture)(
        plan, {var: tns.ftype for var, tns in binds.items()}, {}, DenseStatsFactory()
    )
    assert original == plan
    assert capture.last_prgm == Plan(
        (
            QueryInto(Table(C, (i,)), OVERWRITE, Literal(0.0)),
            QueryInto(Table(C, (i,)), ADD, arg),
            QueryInto(Table(D, (j, i)), OVERWRITE, Reorder(Table(A, (i, j)), (j, i))),
            Produces((C, D)),
        )
    )


@pytest.mark.parametrize("axes", list(permutations(range(3))))
@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
def test_transpose_keeps_output_in_loop_order(axes, compiler):
    data = np.arange(24.0).reshape(2, 3, 4)
    data[data % 3 != 0] = 0
    expected = data.transpose(axes)
    source = BufferizedNDArray.from_numpy(data)
    output = BufferizedNDArray.from_numpy(np.zeros_like(expected))
    fields = (i, j, Field("k"))
    lhs_idxs = tuple(fields[axis] for axis in axes)
    plan = Plan((Query(Table(C, lhs_idxs), Table(A, fields)), Produces((C,))))
    capture = LogicCapture()
    CompilerFormLowerer(capture)(
        plan, {A: source.ftype, C: output.ftype}, {}, DenseStatsFactory()
    )
    match capture.last_prgm:
        case Plan((QueryInto(Table(_, idxs), _, Reorder(_, loop_order)), Produces())):
            assert tuple(idx for idx in loop_order if idx in idxs) == idxs
        case _:
            pytest.fail("Expected a fold with an explicit loop order")
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    result = compiler()(program).main(source, output)
    finch_assert_equal(result[0].to_numpy(), expected)


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
def test_transposed_fold_broadcasts_and_reduces(compiler):
    k = Field("k")
    data = np.arange(6.0).reshape(2, 3)
    source = BufferizedNDArray.from_numpy(data)
    output = BufferizedNDArray.from_numpy(np.ones((3, 4)))
    plan = Plan(
        (
            QueryInto(Table(C, (j, k)), ADD, Reorder(Table(A, (i, j)), (j, k, i))),
            Produces((C,)),
        )
    )
    program = lower(plan, {A: source, C: output})
    result = compiler()(program).main(source, output)
    finch_assert_equal(
        result[0].to_numpy(), 1 + np.broadcast_to(data.sum(axis=0)[:, None], (3, 4))
    )


def test_sparse_transpose():
    import scipy.sparse as sps

    data = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    source = fl.FiberTensor.from_scipy_csr(sps.csr_array(data))
    output = fl.FiberTensor.from_scipy_csr(sps.csr_array(np.zeros(data.T.shape)))
    capture = LogicCapture()
    plan = Plan((Query(Table(C, (j, i)), Table(A, (i, j))), Produces((C,))))
    CompilerFormLowerer(capture)(plan, {A: source.ftype, C: output.ftype}, {}, None)
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    result = NotationCompiler()(program).main(source, output)
    finch_assert_equal(result[0].to_scipy().toarray(), data.T)


@pytest.mark.parametrize("form", [LoopOrderedForm, FormattedForm])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "lhs_idxs, valid",
    [
        ((i,), True),
        ((j,), True),
        ((i, j), True),
        ((), True),
        ((j, i), False),
        ((i, Field("k")), False),
    ],
)
def test_loop_ordered_form_checks_lhs(form, inplace, lhs_idxs, valid):
    rhs = Reorder(Table(A, (i, j)), (i, j))
    lhs = Table(C, lhs_idxs)
    stmt = (
        QueryInto(lhs, ADD, rhs)
        if inplace
        else Query(
            lhs,
            Aggregate(
                ADD,
                Literal(0.0),
                rhs,
                tuple(idx for idx in rhs.fields() if idx not in lhs_idxs),
            ),
        )
    )
    plan = Plan((stmt, Produces((C,))))
    ftypes = {var: tns.ftype for var, tns in bindings().items()}
    if valid:
        form.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    else:
        with pytest.raises(
            ValueError, match="Table index order does not match loop order"
        ):
            form.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


@pytest.mark.parametrize("lhs_idxs", [(i, Field("k")), (Field("k"), i), (Field("k"),)])
def test_loop_orderer_includes_output_only_fields(lhs_idxs):
    plan = Plan(
        (
            Query(
                Table(C, lhs_idxs),
                Aggregate(
                    ADD,
                    Literal(0.0),
                    Table(A, (i, j)),
                    tuple(idx for idx in (i, j) if idx not in lhs_idxs),
                ),
            ),
            Produces((C,)),
        )
    )
    capture = LogicCapture()
    factory = DenseStatsFactory()
    DefaultLoopOrderer(capture)(plan, {A: bindings()[A].ftype}, {}, factory)
    assert isinstance(capture.last_prgm, Plan)
    LoopOrderedForm.validate_inputs(
        capture.last_prgm, capture.last_bindings, {}, factory
    )
    match capture.last_prgm:
        case Plan(
            (
                *_,
                Query(Table(_, idxs), Aggregate(_, _, Reorder(_, order), _)),
                Produces(),
            )
        ):
            assert idxs == lhs_idxs
            assert tuple(idx for idx in order if idx in lhs_idxs) == lhs_idxs
        case _:
            pytest.fail("Expected a loop-ordered aggregate")
