import pytest

import numpy as np
import scipy.sparse as sps

import finch as fl
from finch import ffuncs
from finch import finch_assembly as asm
from finch.algebra import TensorFType
from finch.autoschedule import (
    CompilerFormLowerer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    NotationGenerator,
    default_schedulers,
)
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.factorizer.optimize import DefaultLogicFactorizer
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.fuse_merger import LogicFuseMerger, merge_sibling_fuses
from finch.autoschedule.loop_orderer import DefaultLoopOrderer
from finch.autoschedule.stages import CompilerForm
from finch.autoschedule.tensor_stats import (
    DCStatsFactory,
    DenseStatsFactory,
    FDStatsFactory,
)
from finch.autoschedule.tensor_stats.util import get_lp_norms
from finch.autoschedule.util import fuse_loop_order
from finch.codegen import CCompiler, NumbaCompiler
from finch.compile import NotationCompiler
from finch.finch_assembly import AssemblySimplify, LowerPackedStructSlots
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Fuse,
    FusedAlias,
    HardAlias,
    Literal,
    LogicSimplify,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
    TensorStats,
)
from finch.symbolic import PostOrderDFS, Reflector


def _merged(queries, outputs, tensors):
    """The program of queries, each with its own loops, as LogicFuseMerger
    passes it on, given the tensors of tensors."""
    sf = DCStatsFactory()
    bindings: dict[Alias, TensorFType] = {}
    stats: dict[Alias, TensorStats] = {}
    for alias, (tensor, fields) in tensors.items():
        bindings[alias] = tensor.ftype
        stats[alias] = sf(tensor, fields)
    capture = LogicCapture()
    plan = Plan((*queries, Produces(outputs)))
    LogicFuseMerger(capture)(plan, bindings, stats, sf)
    return capture.last_prgm


def _total(op, init, arg, idx):
    return Aggregate(Literal(op), Literal(init), arg, (idx,))


def test_norms_of_a_vector_run_in_one_loop():
    x = HardAlias("x")
    i1, i2, i3 = Field("i1"), Field("i2"), Field("i3")
    outs = tuple(HardAlias(f"n{k}") for k in range(3))
    queries = (
        fuse_loop_order(
            Query(Table(outs[0], ()), _total(ffuncs.add, 0.0, Table(x, (i1,)), i1)),
            (i1,),
        ),
        fuse_loop_order(
            Query(
                Table(outs[1], ()),
                _total(
                    ffuncs.add,
                    0.0,
                    MapJoin(Literal(ffuncs.mul), (Table(x, (i2,)), Table(x, (i2,)))),
                    i2,
                ),
            ),
            (i2,),
        ),
        fuse_loop_order(
            Query(Table(outs[2], ()), _total(ffuncs.max, 0.0, Table(x, (i3,)), i3)),
            (i3,),
        ),
    )
    vector = fl.asarray(np.array([0.0, 2.0, 0.0, 3.0]))
    prgm = _merged(queries, outs, {x: (vector, (Field("i"),))})

    # The three loops over x run as one, over the field of the first.
    match prgm:
        case Plan((Fuse(idx, Plan(bodies)), Produces())):
            assert idx == i1
            assert [body.lhs.tns for body in bodies] == list(outs)
            assert {i2, i3}.isdisjoint(
                node for body in bodies for node in body.rhs.fields()
            )
        case _:
            raise AssertionError(f"Expected one loop over x: {prgm}")


def test_loops_over_other_tensors_or_positions_stay_apart():
    x, y, a = HardAlias("x"), HardAlias("y"), HardAlias("a")
    i1, i2, j1, j2 = Field("i1"), Field("i2"), Field("j1"), Field("j2")
    s, t, u, v = (HardAlias(name) for name in "stuv")
    vector = fl.asarray(np.ones(4))
    matrix = fl.asarray(np.ones((4, 4)))
    queries = (
        fuse_loop_order(
            Query(Table(s, ()), _total(ffuncs.add, 0.0, Table(x, (i1,)), i1)), (i1,)
        ),
        fuse_loop_order(
            Query(Table(t, ()), _total(ffuncs.add, 0.0, Table(y, (i2,)), i2)), (i2,)
        ),
        # Rows of a, then columns of a, so the outer fields read a at
        # different positions.
        fuse_loop_order(
            Query(Table(u, (i1,)), _total(ffuncs.add, 0.0, Table(a, (i1, j1)), j1)),
            (i1, j1),
        ),
        fuse_loop_order(
            Query(Table(v, (j2,)), _total(ffuncs.add, 0.0, Table(a, (i2, j2)), i2)),
            (j2, i2),
        ),
    )
    tensors = {
        x: (vector, (Field("i"),)),
        y: (vector, (Field("i"),)),
        a: (matrix, (Field("i"), Field("j"))),
    }
    plan = Plan((*queries, Produces((s, t, u, v))))
    assert _merged(queries, (s, t, u, v), tensors) == plan


def test_loops_which_copy_their_accumulators_merge():
    # Each row sum of a accumulates in a workspace fused over the rows, which
    # isn't read with the row field, so the loops over the rows of a merge, as
    # do the loops over each row.
    a, s, t = HardAlias("a"), HardAlias("s"), HardAlias("t")
    i1, i2, j1, j2 = Field("i1"), Field("i2"), Field("j1"), Field("j2")
    s_acc, t_acc = FusedAlias(HardAlias("s_acc"), 1), FusedAlias(HardAlias("t_acc"), 1)

    def row_sum(out, acc, i, j):
        total = _total(ffuncs.add, 0.0, Table(a, (i, j)), j)
        return Fuse(
            i,
            Plan(
                (
                    Fuse(j, Query(Table(acc, (i,)), total)),
                    Query(Table(out, (i,)), Table(acc, (i,))),
                )
            ),
        )

    queries = (row_sum(s, s_acc, i1, j1), row_sum(t, t_acc, i2, j2))
    tensors = {a: (fl.asarray(np.ones((4, 4))), (Field("i"), Field("j")))}
    prgm = _merged(queries, (s, t), tensors)
    total = _total(ffuncs.add, 0.0, Table(a, (i1, j1)), j1)
    assert prgm == Plan(
        (
            Fuse(
                i1,
                Plan(
                    (
                        Fuse(
                            j1,
                            Plan(
                                (
                                    Query(Table(s_acc, (i1,)), total),
                                    Query(Table(t_acc, (i1,)), total),
                                )
                            ),
                        ),
                        Query(Table(s, (i1,)), Table(s_acc, (i1,))),
                        Query(Table(t, (i1,)), Table(t_acc, (i1,))),
                    )
                ),
            ),
            Produces((s, t)),
        )
    )


def test_loops_which_depend_on_each_other_stay_apart():
    x, s, t = HardAlias("x"), HardAlias("s"), HardAlias("t")
    i1, i2 = Field("i1"), Field("i2")
    # t reads s, which the first loop reduces into, so it needs all of it.
    queries = (
        fuse_loop_order(
            Query(Table(s, ()), _total(ffuncs.add, 0.0, Table(x, (i1,)), i1)), (i1,)
        ),
        fuse_loop_order(
            Query(
                Table(t, (i2,)),
                _total(
                    ffuncs.add,
                    0.0,
                    MapJoin(Literal(ffuncs.mul), (Table(x, (i2,)), Table(s, ()))),
                    i2,
                ),
            ),
            (i2,),
        ),
    )
    vector = fl.asarray(np.ones(4))
    plan = Plan((*queries, Produces((t,))))
    assert _merged(queries, (t,), {x: (vector, (Field("i"),))}) == plan


def _sum_of(out, tns, idx):
    return fuse_loop_order(
        Query(Table(out, ()), _total(ffuncs.add, 0.0, Table(tns, (idx,)), idx)), (idx,)
    )


def _copy(out, tns):
    return Query(Table(out, ()), Table(tns, ()))


def test_loops_with_statements_between_them_merge():
    x, y = HardAlias("x"), HardAlias("y")
    s, t, u, r = (HardAlias(name) for name in "stur")
    i1, i2, i3 = Field("i1"), Field("i2"), Field("i3")
    vector = fl.asarray(np.ones(4))
    tensors = {x: (vector, (Field("i"),)), y: (vector, (Field("i"),))}

    # The loop over y doesn't depend on the first loop over x, so it runs first.
    queries = (_sum_of(s, x, i1), _sum_of(t, y, i2), _sum_of(u, x, i3))
    match _merged(queries, (s, t, u), tensors):
        case Plan((Fuse(idx, _) as other, Fuse(merged, Plan(bodies)), Produces())):
            assert (idx, merged) == (i2, i1)
            assert other == queries[1]
            assert [body.lhs.tns for body in bodies] == [s, u]
        case prgm:
            raise AssertionError(f"Expected one loop over x: {prgm}")

    # Copying s needs all of the first loop over x, so it runs after the merge.
    queries = (_sum_of(s, x, i1), _copy(r, s), _sum_of(u, x, i3))
    match _merged(queries, (r, u), tensors):
        case Plan((Fuse(merged, Plan(bodies)), copy, Produces())):
            assert merged == i1
            assert copy == queries[1]
            assert [body.lhs.tns for body in bodies] == [s, u]
        case prgm:
            raise AssertionError(f"Expected one loop over x: {prgm}")


def test_loops_which_depend_through_statements_between_them_stay_apart():
    x, s, r, t = HardAlias("x"), HardAlias("s"), HardAlias("r"), HardAlias("t")
    i1, i2 = Field("i1"), Field("i2")
    # The second loop reads r, a copy of s, which the first loop reduces into.
    queries = (
        _sum_of(s, x, i1),
        _copy(r, s),
        fuse_loop_order(
            Query(
                Table(t, ()),
                _total(
                    ffuncs.add,
                    0.0,
                    MapJoin(Literal(ffuncs.mul), (Table(x, (i2,)), Table(r, ()))),
                    i2,
                ),
            ),
            (i2,),
        ),
    )
    vector = fl.asarray(np.ones(4))
    plan = Plan((*queries, Produces((t,))))
    assert _merged(queries, (t,), {x: (vector, (Field("i"),))}) == plan


def test_degree_scans_of_a_matrix_run_as_one_scan():
    a, rows, cols = HardAlias("a"), HardAlias("rows"), HardAlias("cols")
    i1, j1, i2, j2 = Field("i1"), Field("j1"), Field("i2"), Field("j2")
    queries = (
        fuse_loop_order(
            Query(Table(rows, (i1,)), _total(ffuncs.add, 0.0, Table(a, (i1, j1)), j1)),
            (i1, j1),
        ),
        fuse_loop_order(
            Query(Table(cols, (j2,)), _total(ffuncs.add, 0.0, Table(a, (i2, j2)), i2)),
            (i2, j2),
        ),
    )
    matrix = fl.asarray(np.ones((3, 4)))
    prgm = _merged(queries, (rows, cols), {a: (matrix, (Field("i"), Field("j")))})

    # Both loops of the second scan merge into those of the first.
    match prgm:
        case Plan((Fuse(i, Plan((Fuse(j, Plan(bodies)),))), Produces())):
            assert (i, j) == (i1, j1)
            assert [body.lhs for body in bodies] == [
                Table(rows, (i1,)),
                Table(cols, (j1,)),
            ]
        case _:
            raise AssertionError(f"Expected one scan of a: {prgm}")


def test_lp_norms_of_degrees_are_exact():
    rng = np.random.default_rng(0)
    data = np.where(rng.random((30, 20)) < 0.2, rng.random((30, 20)), 0.0)
    norms = (0, 1, 2, float("inf"))
    i, j = Field("i"), Field("j")
    result = get_lp_norms(fl.asarray(data), (i, j), norms)
    for field, axis in ((i, 1), (j, 0)):
        degrees = np.count_nonzero(data, axis=axis)
        expected = [
            np.count_nonzero(degrees),
            degrees.sum(),
            np.sqrt((degrees**2).sum()),
            degrees.max(),
        ]
        np.testing.assert_allclose(result[field], expected)


@pytest.mark.parametrize(
    "compiler",
    [
        pytest.param(NumbaCompiler, id="numba"),
        pytest.param(CCompiler, id="c", marks=pytest.mark.c_backend),
    ],
)
def test_lp_norms_compile_as_one_scan_and_one_loop_per_axis(compiler, monkeypatch):
    # The statistics scheduler, with the backend under test.
    capture = LogicCapture(
        DefaultLogicFormatter(
            CompilerFormLowerer(
                LogicCompiler(
                    NotationCompiler(
                        compiler(),
                        ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify()),
                    )
                )
            )
        )
    )
    scheduler = LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(
                LogicSimplify(DefaultLoopOrderer(LogicFuseMerger(capture)))
            ),
            stats_factory=FDStatsFactory(),
        )
    )
    monkeypatch.setattr(default_schedulers, "NON_RECURSIVE_SCHEDULER", scheduler)
    rng = np.random.default_rng(1)
    data = np.where(rng.random((12, 9)) < 0.3, rng.random((12, 9)), 0.0)
    i, j = Field("i"), Field("j")
    result = get_lp_norms(fl.asarray(data), (i, j), (1, 2, float("inf")))
    for field, axis in ((i, 1), (j, 0)):
        degrees = np.count_nonzero(data, axis=axis)
        expected = [degrees.sum(), np.sqrt((degrees**2).sum()), degrees.max()]
        np.testing.assert_allclose(result[field], expected)

    # One scan of the matrix computes both degree sequences, and one loop over
    # each computes its norms.
    match capture.last_prgm:
        case Plan((Fuse(), Fuse(), Fuse(), Produces())):
            pass
        case prgm:
            raise AssertionError(f"Expected one scan and two norm loops: {prgm}")


def test_reads_of_one_tensor_in_a_merged_loop_share_one_walk():
    # Merging the degree scans of a CSR matrix reads it twice in one loop nest.
    a, rows, cols = HardAlias("a"), HardAlias("rows"), HardAlias("cols")
    i1, j1, i2, j2 = Field("i1"), Field("j1"), Field("i2"), Field("j2")

    def degrees(out, idx, over, order):
        nonzero = MapJoin(Literal(ffuncs.ne), (Table(a, order), Literal(0.0)))
        return Query(Table(out, (idx,)), _total(ffuncs.add, np.int64(0), nonzero, over))

    plan = Plan(
        (
            fuse_loop_order(degrees(rows, i1, j1, (i1, j1)), (i1, j1)),
            fuse_loop_order(degrees(cols, j2, i2, (i2, j2)), (i2, j2)),
            Produces((rows, cols)),
        )
    )
    plan = merge_sibling_fuses(plan, lambda _: True)
    assert isinstance(plan, Plan)
    data = np.array([[2.0, 0, 3, 0], [0, 0, 0, 0], [1, 5, 0, 7]])
    bindings = {
        a: fl.asarray(sps.csr_array(data)),
        rows: fl.asarray(np.zeros(0, dtype=np.int64)),
        cols: fl.asarray(np.zeros(0, dtype=np.int64)),
    }
    ftypes: dict[Alias, TensorFType] = {var: val.ftype for var, val in bindings.items()}
    CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    program = NotationGenerator()(plan, ftypes, {}, None)
    asm_program = NotationCompiler(Reflector())(program)  # ty: ignore[invalid-argument-type]

    # The rows of a are walked once, so the walk splits each row into its gaps
    # and its entries, rather than into the four cases of two walks.
    walks = []
    for node in PostOrderDFS(asm_program):
        match node:
            case asm.WhileLoop(_, asm.Block(bodies)) if any(
                isinstance(child, asm.ForLoop) for child in bodies
            ):
                walks.append(bodies)
    assert len(walks) == 1
    assert sum(isinstance(child, asm.ForLoop) for child in walks[0]) == 2

    row_degrees, col_degrees = NumbaCompiler()(asm_program).main(*bindings.values())
    np.testing.assert_array_equal(row_degrees.to_numpy(), np.count_nonzero(data, 1))
    np.testing.assert_array_equal(col_degrees.to_numpy(), np.count_nonzero(data, 0))
