import numpy as np

from finch import ffuncs
from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import ftype
from finch.algebra.fill import DynamicFillFType
from finch.autoschedule import (
    CompilerFormLowerer,
    LogicCapture,
    LogicFillDropper,
    NotationGenerator,
    drop_fills,
)
from finch.autoschedule.drop_fills import dropping_fill
from finch.autoschedule.util import fuse_loop_order
from finch.compile import NotationCompiler
from finch.finch_logic import (
    Aggregate,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.symbolic import PostOrderDFS
from finch.tensor import dense, element, fiber_tensor, sparse_list

from .test_sparse_update import dense_values, run_copy, tensors
from .utils import with_workspaces

DENSE = fiber_tensor(dense(element(0.0)))
LIST = fiber_tensor(sparse_list(element(0.0), ftype(np.intp)))
DENSE_ROWS = fiber_tensor(sparse_list(dense(element(0.0)), ftype(np.intp)))
# The fill of the arguments, as their formats type it.
FILL = LIST.fill_type.value

i, j = Field("i"), Field("j")
A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")


def product(*tables):
    return MapJoin(Literal(ffuncs.mul), tables)


def elementwise_plan():
    # C[i] = A[i] * B[i]
    arg = product(Table(A, (i,)), Table(B, (i,)))
    query = Query(
        Table(C, (i,)), Aggregate(Literal(ffuncs.overwrite), Literal(0.0), arg, ())
    )
    return Plan((fuse_loop_order(query, (i,)), Produces((C,)))), arg


def test_a_dense_loop_drops_the_fills_it_computes():
    plan, arg = elementwise_plan()
    query = Query(
        Table(C, (i,)),
        Aggregate(
            Literal(ffuncs.overwrite), Literal(0.0), dropping_fill(arg, FILL), ()
        ),
    )
    dropped = drop_fills(plan, {A: DENSE, B: LIST, C: LIST})
    assert dropped == Plan((fuse_loop_order(query, (i,)), Produces((C,))))


def test_a_sparse_loop_is_left_alone():
    plan, _ = elementwise_plan()
    assert drop_fills(plan, {A: LIST, B: LIST, C: LIST}) == plan


def test_a_dense_output_is_left_alone():
    # A dense level stores every slot whatever is written to it, so there are
    # no fills to drop.
    plan, _ = elementwise_plan()
    assert drop_fills(plan, {A: DENSE, B: LIST, C: DENSE}) == plan


def test_an_output_with_any_sparse_level_drops_fills():
    # C[i, j] = A[i, j] * B[i, j], into a C of sparse rows of dense columns.
    arg = product(Table(A, (i, j)), Table(B, (i, j)))
    query = Query(
        Table(C, (i, j)), Aggregate(Literal(ffuncs.overwrite), Literal(0.0), arg, ())
    )
    plan = Plan((fuse_loop_order(query, (i, j)), Produces((C,))))
    all_dense = fiber_tensor(dense(dense(element(0.0))))
    assert drop_fills(plan, {A: all_dense, B: all_dense, C: DENSE_ROWS}) != plan
    assert drop_fills(plan, {A: all_dense, B: all_dense, C: all_dense}) == plan


def test_a_reduction_drops_fills_within_its_loop():
    # C[i] = sum_j A[i, j] * B[j], whose rows of A are dense, so the loop over j
    # visits every index of them.
    arg = product(Table(A, (i, j)), Table(B, (j,)))
    query = Query(
        Table(C, (i,)), Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (j,))
    )
    plan = Plan((fuse_loop_order(query, (i, j)), Produces((C,))))
    dropped = Query(
        Table(C, (i,)),
        Aggregate(Literal(ffuncs.add), Literal(0.0), dropping_fill(arg, FILL), (j,)),
    )
    assert drop_fills(plan, {A: DENSE_ROWS, B: LIST, C: LIST}) == Plan(
        (fuse_loop_order(dropped, (i, j)), Produces((C,)))
    )


def test_any_init_drops_the_fills_of_the_argument():
    # C[i] = 5 + sum_j A[i, j] * B[j]: the fills of the argument are 0, which add
    # nothing, whatever the init.
    arg = product(Table(A, (i, j)), Table(B, (j,)))
    query = Query(
        Table(C, (i,)), Aggregate(Literal(ffuncs.add), Literal(5.0), arg, (j,))
    )
    plan = Plan((fuse_loop_order(query, (i, j)), Produces((C,))))
    dropped = Query(
        Table(C, (i,)),
        Aggregate(Literal(ffuncs.add), Literal(5.0), dropping_fill(arg, FILL), (j,)),
    )
    assert drop_fills(plan, {A: DENSE_ROWS, B: LIST, C: LIST}) == Plan(
        (fuse_loop_order(dropped, (i, j)), Produces((C,)))
    )


def test_a_dynamic_fill_is_left_alone():
    # The fills of A and B, and so of their product, are only known when it
    # runs. A static 0 of either would make the product's fill a static 0.
    plan, _ = elementwise_plan()
    dynamic = fiber_tensor(dense(element(DynamicFillFType(np.float64))))
    assert drop_fills(plan, {A: dynamic, B: dynamic, C: LIST}) == plan


def test_a_fill_which_changes_the_output_is_left_alone():
    # C[i] = sum_j (A[i, j] + 1) adds the fill 1 of its argument, and so does
    # C[i] = prod_j A[i, j] * B[j] multiply by its fill 0, so neither may skip
    # those terms. C[i] = A[i] * B[i] overwrites C with the fill 0 of its
    # argument, which isn't the fill 1 of C, as an entry left unwritten is.
    plus_one = MapJoin(Literal(ffuncs.add), (Table(A, (i, j)), Literal(1.0)))
    for op, arg in (
        (ffuncs.add, plus_one),
        (ffuncs.mul, product(Table(A, (i, j)), Table(B, (j,)))),
    ):
        query = Query(Table(C, (i,)), Aggregate(Literal(op), Literal(0.0), arg, (j,)))
        plan = Plan((fuse_loop_order(query, (i, j)), Produces((C,))))
        assert drop_fills(plan, {A: DENSE_ROWS, B: LIST, C: LIST}) == plan
    plan, _ = elementwise_plan()
    ones = fiber_tensor(sparse_list(element(1.0), ftype(np.intp)))
    assert drop_fills(plan, {A: DENSE, B: LIST, C: ones}) == plan


def test_dropping_fills_keeps_what_a_program_computes_and_stores():
    # c[i] = x[i] * y[i], where the dense x and y have zeros, into a sparse list
    # c, which the dense loop over i writes at every index. The notation
    # backend drops the fills a sparse list is written anyway, so the program
    # computes and stores the same when it writes only where the value isn't 0.
    loader = asm.AssemblyInterpreter()
    data_x = np.array([0, 2, 0, 4, 5, 0, 7, 0, 0, 9])
    data_y = np.array([1, 0, 3, 4, 5, 0, 7, 8, 0, 9])
    x, c = HardAlias("x"), HardAlias("c")
    y = HardAlias("y")
    inputs = {
        x: run_copy(loader, *tensors("D", data_x)),
        y: run_copy(loader, *tensors("D", data_y)),
    }
    _, output = tensors("S", data_x)
    fill = output.ftype.fill_type.value
    arg = product(Table(x, (i,)), Table(y, (i,)))
    query = Query(
        Table(c, (i,)), Aggregate(Literal(ffuncs.overwrite), Literal(fill), arg, ())
    )
    plan = Plan((fuse_loop_order(query, (i,)), Produces((c,))))
    bindings = {**{var: tns.ftype for var, tns in inputs.items()}, c: output.ftype}

    def run(stage):
        capture = LogicCapture()
        stage(CompilerFormLowerer(capture))(plan, dict(bindings), {}, None)
        prgm, formats = capture.last_prgm, capture.last_bindings
        program = NotationGenerator()(prgm, formats, {}, None)
        conditional = False
        for node in PostOrderDFS(program):
            match node:
                case ntn.If():
                    conditional = True
        target = formats[c].construct((10,))
        (result,) = NotationCompiler(loader)(program).main(
            *with_workspaces(program, *inputs.values(), target)
        )
        np.testing.assert_array_equal(dense_values(result.lvl), data_x * data_y)
        stored = result.lvl.ptr.load(1) - result.lvl.ptr.load(0)
        assert stored == np.count_nonzero(data_x * data_y)
        return conditional

    assert not run(lambda ctx: ctx)
    assert run(LogicFillDropper)
