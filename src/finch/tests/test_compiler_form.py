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
from finch.autoschedule.util import fuse_loop_order
from finch.compile import NotationCompiler
from finch.compile_jl.analyze import reset_argument_positions
from finch.finch_logic import (
    Aggregate,
    Field,
    Fuse,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
)
from finch.symbolic import PostOrderDFS
from finch.tensor import BufferizedNDArray

from .conftest import finch_assert_equal

i, j = Field("i"), Field("j")
A, B, C, D = HardAlias("A"), HardAlias("B"), HardAlias("C"), HardAlias("D")
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


def sum_rows(lhs, arg, init=0.0):
    rhs = Aggregate(ADD, Literal(init), Table(arg, (i, j)), (j,))
    return fuse_loop_order(Query(Table(lhs, (i,)), rhs), (i, j))


def fold(lhs, op, rhs, loop_order):
    return fuse_loop_order(QueryInto(lhs, op, rhs), loop_order)


def test_folds_with_the_same_op_share_a_declaration():
    plan = Plan(
        (
            sum_rows(C, A),
            fold(Table(C, (i,)), ADD, Table(B, (i, j)), (i, j)),
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
            sum_rows(C, A),
            fold(
                Table(D, (i,)),
                OVERWRITE,
                MapJoin(MUL, (Table(C, (i,)), Literal(2.0))),
                (i,),
            ),
            Produces((D,)),
        )
    )
    binds = bindings()
    program = lower(plan, binds)
    assert lifecycle(program) == [
        ("C", "Declare", ffuncs.add),
        ("C", "Freeze", ffuncs.add),
        ("D", "Declare", ffuncs.init_write(np.float64(0.0))),
        ("D", "Freeze", ffuncs.init_write(np.float64(0.0))),
    ]
    finch_assert_equal(run(program, binds), 2 * A_DATA.sum(axis=1))


def test_changing_op_refreezes_and_thaws():
    plan = Plan(
        (
            sum_rows(C, A, init=1.0),
            fold(Table(C, (i,)), MUL, Table(B, (i, j)), (i, j)),
            Produces((C,)),
        )
    )
    binds = bindings()
    # A declaration can't change a static fill, so C starts from the init.
    binds[C] = BufferizedNDArray.from_numpy(np.zeros(2), fill_value=1.0)
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


def test_initialization_is_declared_by_overwriting():
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
            fold(Table(C, (i,)), ADD, Table(A, (i, j)), (i, j)),
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
        (fold(Table(C, (i,)), ADD, Table(A, (i, j)), (i, j)), True),
        # A transpose has a loop order, which visits the lhs in order.
        (fold(Table(D, (j, i)), OVERWRITE, Table(A, (i, j)), (j, i)), True),
        (QueryInto(Table(C, (i,)), OVERWRITE, Table(A, (j, i))), False),
        (QueryInto(Table(C, (i,)), ADD, Table(A, (i, j))), False),
        # Fields cannot be omitted or revisited by the loop order.
        (fold(Table(C, (i,)), ADD, Table(A, (i, j)), (i,)), False),
        (fold(Table(C, (i,)), ADD, Table(A, (i, j)), (i, j, j)), False),
        # Definitions are queries, which copy a table or reduce from a literal.
        (sum_rows(C, A), True),
        (fuse_loop_order(Query(Table(D, (j, i)), Table(A, (i, j))), (j, i)), True),
        (Query(Table(D, (j, i)), Table(A, (i, j))), False),
        (
            fuse_loop_order(
                Query(
                    Table(C, (i,)),
                    Aggregate(ADD, Table(C, (i,)), Table(A, (i, j)), (j,)),
                ),
                (i, j),
            ),
            False,
        ),
        # Initializations overwrite.
        (QueryInto(Table(C, (i,)), ADD, Literal(1.0)), False),
        # A tensor is either being read or being written.
        (
            fold(
                Table(C, (i,)),
                ADD,
                MapJoin(ADD, (Table(C, (i,)), Table(A, (i, j)))),
                (i, j),
            ),
            False,
        ),
        (
            fuse_loop_order(
                Query(
                    Table(C, (i,)),
                    Aggregate(
                        ADD,
                        Literal(0.0),
                        MapJoin(ADD, (Table(C, (i,)), Table(A, (i, j)))),
                        (j,),
                    ),
                ),
                (i, j),
            ),
            False,
        ),
        # Folds need a loop order, which visits the output in order.
        (QueryInto(Table(C, (i,)), ADD, MapJoin(ADD, (Table(A, (i, j)),))), False),
        (fold(Table(D, (j, i)), ADD, Table(A, (i, j)), (i, j)), False),
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


@pytest.mark.parametrize("form", [LoopOrderedForm, FormattedForm])
@pytest.mark.parametrize(
    "stmt",
    [
        QueryInto(Table(C, (i,)), OVERWRITE, Literal(0.0)),
        fold(Table(D, (j, i)), OVERWRITE, Table(A, (i, j)), (j, i)),
    ],
)
def test_loop_ordered_form_accepts_compiler_form(form, stmt):
    binds = {
        **bindings(),
        D: BufferizedNDArray.from_numpy(np.zeros((2, 2))),
    }
    plan = Plan((stmt, Produces((C,))))
    ftypes = {var: tns.ftype for var, tns in binds.items()}
    form.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_compiler_form_lowerer_keeps_definitions():
    transpose = fuse_loop_order(Query(Table(D, (j, i)), Table(A, (i, j))), (j, i))
    plan = Plan((sum_rows(C, A), transpose, Produces((C, D))))
    capture = LogicCapture()
    binds = bindings(d=np.zeros((2, 2)))
    _, result_bindings = CompilerFormLowerer(capture)(
        plan, {var: tns.ftype for var, tns in binds.items()}, {}, DenseStatsFactory()
    )
    assert result_bindings == capture.last_bindings
    assert capture.last_prgm == plan


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
    transpose = fuse_loop_order(Query(Table(C, lhs_idxs), Table(A, fields)), lhs_idxs)
    plan = Plan((transpose, Produces((C,))))
    capture = LogicCapture()
    CompilerFormLowerer(capture)(
        plan, {A: source.ftype, C: output.ftype}, {}, DenseStatsFactory()
    )
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
            fold(Table(C, (j, k)), ADD, Table(A, (i, j)), (j, k, i)),
            Produces((C,)),
        )
    )
    program = lower(plan, {A: source, C: output})
    result = compiler()(program).main(source, output)
    finch_assert_equal(
        result[0].to_numpy(), 1 + np.broadcast_to(data.sum(axis=0)[:, None], (3, 4))
    )


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize("n", [0, 3])
def test_overwriting_with_an_empty_reduction_keeps_values(compiler, n):
    data = np.arange(2.0 * n).reshape(2, n)
    source = BufferizedNDArray.from_numpy(data)
    output = BufferizedNDArray.from_numpy(np.array([1.0, 2.0]))
    plan = Plan(
        (
            fold(Table(C, (i,)), OVERWRITE, Table(A, (i, j)), (i, j)),
            Produces((C,)),
        )
    )
    program = lower(plan, {A: source, C: output})
    result = compiler()(program).main(source, output)
    # Overwriting along j keeps the last value, and an empty j writes nothing.
    expected = data[:, -1] if n else np.array([1.0, 2.0])
    finch_assert_equal(result[0].to_numpy(), expected)


def test_guarded_overwrite_keeps_previous_values_for_julia():
    def resets(rhs, loop_order):
        plan = Plan((fold(Table(C, (i,)), OVERWRITE, rhs, loop_order), Produces((C,))))
        func = lower(plan, bindings()).funcs[0]
        positions = reset_argument_positions(func)
        return {func.args[pos].name.split("#")[1] for pos in positions}

    # Overwriting without a reduction resets C, so the Julia runtime needn't
    # pass its previous values, but a reduction which may be empty might not.
    assert resets(Table(D, (i,)), (i,)) == {"C"}
    assert resets(Table(A, (i, j)), (i, j)) == set()


def test_sparse_transpose():
    import scipy.sparse as sps

    data = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    source = fl.FiberTensor.from_scipy_csr(sps.csr_array(data))
    output = fl.FiberTensor.from_scipy_csr(sps.csr_array(np.zeros(data.T.shape)))
    capture = LogicCapture()
    transpose = fuse_loop_order(Query(Table(C, (j, i)), Table(A, (i, j))), (j, i))
    plan = Plan((transpose, Produces((C,))))
    CompilerFormLowerer(capture)(plan, {A: source.ftype, C: output.ftype}, {}, None)
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    result = NotationCompiler()(program).main(source, output)
    finch_assert_equal(result[0].to_scipy().toarray(), data.T)


@pytest.mark.parametrize("form", [LoopOrderedForm, FormattedForm])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "lhs_idxs, error",
    [
        ((i,), None),
        ((j,), None),
        ((i, j), None),
        ((), None),
        ((j, i), "Table index order does not match loop order"),
        ((i, Field("k")), "k need Fuse nodes"),
    ],
)
def test_loop_ordered_form_checks_lhs(form, inplace, lhs_idxs, error):
    rhs = Table(A, (i, j))
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
    plan = Plan((fuse_loop_order(stmt, (i, j)), Produces((C,))))
    ftypes = {var: tns.ftype for var, tns in bindings().items()}
    if error is None:
        form.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    else:
        with pytest.raises(ValueError, match=error):
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
        case Plan((*_, stmt, Produces())):
            order = []
            while isinstance(stmt, Fuse):
                order.append(stmt.idx)
                stmt = stmt.body
            match stmt:
                case Query(Table(_, idxs), Aggregate()):
                    assert idxs == lhs_idxs
                    assert tuple(idx for idx in order if idx in lhs_idxs) == lhs_idxs
                case _:
                    pytest.fail("Expected a loop-ordered aggregate")
        case _:
            pytest.fail("Expected a loop-ordered aggregate")
