import math
from typing import Any

import pytest

import numpy as np
import scipy.sparse as sps

import finch as fl
from finch import ffuncs
from finch.algebra import StaticFillFType, TensorFType, ftype
from finch.autoschedule import (
    CompilerFormLowerer,
    NotationGenerator,
    default_schedulers,
)
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    CostModel,
    DenseLevelOption,
    Prefix,
    SparseListLevelOption,
    WriteBasis,
    write_class,
)
from finch.autoschedule.formatter.galley_formatter import GalleyFormatter
from finch.autoschedule.formatter.smart_formatter import CostFormatter, format_ftype
from finch.autoschedule.fuse_merger import LogicFuseMerger
from finch.autoschedule.loop_orderer import BFSLoopOrderer
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DCStats,
    DCStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.bound_stats import DC
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.tensor_stats.uniform_stats import UniformStats
from finch.autoschedule.util import fuse_loop_order
from finch.autoschedule.workspacer import (
    LogicWorkspacer,
    accumulator_depth,
    accumulator_pays,
    add_1d_workspaces,
    add_accumulators,
    add_cost_effective_reformats,
    add_cost_effective_workspaces,
    add_reformat,
    add_workspace,
    galley_format_cost,
    workspace_depth,
)
from finch.codegen import NumbaCompiler, NumpyBuffer
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
    MapJoin,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
    TensorStats,
)
from finch.symbolic import PostOrderDFS
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    SparseListLevel,
    Temp0DFType,
    element,
    temp_0d,
)


def _matmul(c, a, b, loop_order=()):
    # c[i, k] = sum_j a[i, j] * b[j, k], looping over loop_order.
    i, j, k = Field("i"), Field("j"), Field("k")
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k)))),
        (j,),
    )
    query = fuse_loop_order(Query(Table(c, (i, k)), rhs), loop_order)
    return Plan((query, Produces((c,)))), rhs


def test_workspace_depth_counts_the_fields_written_in_order():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")

    # The j loop makes the writes to c scatter at k, but not at i.
    assert workspace_depth(_matmul(c, a, b, (i, j, k))[0], c) == 1
    assert workspace_depth(_matmul(c, a, b, (j, i, k))[0], c) == 0
    assert workspace_depth(_matmul(c, a, b, (i, k, j))[0], c) is None

    # An alias which is updated after its definition has no workspace.
    define, _ = _matmul(c, a, b, (i, j, k))
    update = fuse_loop_order(
        QueryInto(Table(c, (i, k)), Literal(ffuncs.add), Table(a, (i, k))), (i, k)
    )
    plan = Plan((define.bodies[0], update, Produces((c,))))
    assert workspace_depth(plan, c) is None


def test_add_workspace_copies_each_slice_after_its_query():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c, w = (HardAlias(name) for name in "abcw")
    plan, rhs = _matmul(c, a, b, (i, j, k))
    fused = FusedAlias(w, 1)

    staged = add_workspace(plan, c, 1, w)

    assert staged == Plan(
        (
            Fuse(
                i,
                Plan(
                    (
                        fuse_loop_order(Query(Table(fused, (i, k)), rhs), (j, k)),
                        Fuse(k, Query(Table(c, (i, k)), Table(fused, (i, k)))),
                    )
                ),
            ),
            Produces((c,)),
        )
    )
    tensor = fl.asarray(np.ones((2, 2)))
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype, b: tensor.ftype}
    assert isinstance(staged, Plan)
    LoopOrderedForm.validate_inputs(staged, bindings, {}, DCStatsFactory())

    # Without a field written in order, the workspace holds all of c.
    plan, rhs = _matmul(c, a, b, (j, i, k))
    full = FusedAlias(w, 0)
    assert add_workspace(plan, c, 0, w) == Plan(
        (
            fuse_loop_order(Query(Table(full, (i, k)), rhs), (j, i, k)),
            fuse_loop_order(Query(Table(c, (i, k)), Table(full, (i, k))), (i, k)),
            Produces((c,)),
        )
    )


def _random_csr(n, per_row, seed):
    """
    An n by n matrix with per_row entries in random columns of each row, as a
    dense root over sorted lists, and as SciPy CSR.
    """
    rng = np.random.default_rng(seed)
    cols = np.concatenate(
        [np.sort(rng.choice(n, per_row, replace=False)) for _ in range(n)]
    )
    vals = rng.random(n * per_row)
    ptr = np.arange(0, n * per_row + 1, per_row, dtype=np.intp)
    fmt = element(np.float64(0), ftype(np.float64), ftype(np.intp))
    lvl = SparseListLevel(
        ElementLevel(fmt, NumpyBuffer(vals)),
        np.intp(n),
        NumpyBuffer(ptr),
        NumpyBuffer(cols.astype(np.intp)),
    )
    tensor = FiberTensor(DenseLevel(lvl, np.intp(n)))
    return tensor, sps.csr_array((vals, cols, ptr), shape=(n, n))


def _csr_stats(tensor, csr, fields):
    """
    The stats DCStatsFactory gives tensor, from the degree sequences of its
    SciPy copy csr, since the stats scheduler can't read a fiber tensor: the
    rows and columns with entries, the entries, and the most in a row and in a
    column.
    """
    i, j = fields
    rows = np.diff(csr.indptr)
    cols = np.bincount(csr.indices, minlength=csr.shape[1])
    base = BaseTensorStats(
        fields,
        dict(zip(fields, map(float, csr.shape), strict=True)),
        tensor.ftype.fill_type,
        dict(zip(fields, tensor.ftype.shape_type, strict=True)),
    )
    dcs = {
        DC(frozenset(), frozenset({i}), float(np.count_nonzero(rows)), math.inf),
        DC(frozenset(), frozenset({j}), float(np.count_nonzero(cols)), math.inf),
        DC(frozenset(), frozenset(fields), float(csr.nnz), math.inf),
        DC(frozenset({i}), frozenset(fields), float(np.max(rows)), math.inf),
        DC(frozenset({j}), frozenset(fields), float(np.max(cols)), math.inf),
    }
    return DCStats(base, dcs, DCStatsFactory().ps)


def _order(a_csr, b_csr, workspaces):
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    (tensor_a, csr_a), (tensor_b, csr_b) = a_csr, b_csr
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: _csr_stats(tensor_a, csr_a, (i, j)),
        b: _csr_stats(tensor_b, csr_b, (j, k)),
    }
    capture = LogicCapture()
    plan, _ = _matmul(c, a, b)
    formatter = CostFormatter(CompilerFormLowerer(capture))
    BFSLoopOrderer(LogicWorkspacer(formatter) if workspaces else formatter)(
        plan, {a: tensor_a.ftype, b: tensor_b.ftype}, stats, sf
    )
    return capture


def test_workspacer_accumulates_rows_in_a_workspace():
    i, j, k = Field("i"), Field("j"), Field("k")
    c = HardAlias("c")
    # The output is sparse enough that a workspace for each row costs less
    # than storing the output densely.
    (tensor_a, csr_a), (tensor_b, csr_b) = (
        _random_csr(10_000, 4, 0),
        _random_csr(10_000, 4, 1),
    )

    # The search scatters into the columns of c, for each of its rows.
    capture = _order((tensor_a, csr_a), (tensor_b, csr_b), workspaces=False)
    match capture.last_prgm:
        case Plan((Fuse(i_, Fuse(j_, Fuse(k_, Query(Table(lhs, _), _)))), Produces())):
            assert (i_, j_, k_, lhs) == (i, j, k, c)
        case prgm:
            raise AssertionError(f"Expected the loop order (i, j, k): {prgm}")

    # The loop orderer propagates copies before the workspacer adds the
    # workspace, so its copy stays, and the writes scatter into a workspace for
    # each row instead.
    capture = _order((tensor_a, csr_a), (tensor_b, csr_b), workspaces=True)
    match capture.last_prgm:
        case Plan(
            (
                Fuse(
                    _,
                    Plan(
                        (
                            Fuse(_, Fuse(_, Query(Table(FusedAlias(ws, 1), _), _))),
                            Fuse(_, Query(Table(lhs, (i_, k_)), Table(view, _))),
                        )
                    ),
                ),
                Produces(_),
            )
        ):
            assert (lhs, i_, k_, view) == (c, i, k, FusedAlias(ws, 1))
        case prgm:
            raise AssertionError(f"Expected a workspace for each row: {prgm}")
    # The copy writes c in order, so its rows are sorted lists.
    fmt = capture.last_bindings[c]
    assert isinstance(fmt.lvl_t.lvl_t, fl.SparseListLevelFType)

    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    bindings = [
        tensor_a,
        tensor_b,
        capture.last_bindings[ws].construct((0,)),
        capture.last_bindings[c].construct((0, 0)),
    ]
    compiler = NotationCompiler(
        NumbaCompiler(), ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify())
    )
    (result,) = compiler(program).main(*bindings)
    assert abs(result.to_scipy() - csr_a @ csr_b).max() < 1e-12


def test_workspacer_needs_a_field_written_in_order():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(np.ones((4, 4))), (i, j)),
        b: sf(fl.asarray(np.ones((4, 4))), (j, k)),
    }
    # The loops scatter into i, so a workspace would store all of c.
    plan, _ = _matmul(c, a, b, (k, i, j))
    assert (
        add_cost_effective_workspaces(plan, {}, stats, sf, PROTOCOL_COST_MODEL) == plan
    )


@pytest.mark.parametrize(
    ("n", "per_row", "staged"),
    [
        (10_000, 4, True),
        # A product which fills most of its output costs less stored densely.
        (256, 16, False),
    ],
)
def test_workspacer_keeps_workspaces_which_cost_less(n, per_row, staged):
    capture = _order(
        _random_csr(n, per_row, 0), _random_csr(n, per_row, 1), workspaces=True
    )
    fused = []
    for node in PostOrderDFS(capture.last_prgm):
        match node:
            case FusedAlias():
                fused.append(node)
    assert bool(fused) == staged


def test_workspacer_leaves_queries_which_write_in_order():
    # An inner product writes c in order, so it needs no workspace.
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    rng = np.random.default_rng(0)
    tensor_a, tensor_b = fl.asarray(rng.random((8, 8))), fl.asarray(rng.random((8, 8)))
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(tensor_a, (i, j)),
        b: sf(tensor_b, (j, k)),
    }
    bindings: dict[Alias, TensorFType] = {a: tensor_a.ftype, b: tensor_b.ftype}
    plan, _ = _matmul(c, a, b, (i, k, j))
    assert workspace_depth(plan, c) is None
    assert (
        add_cost_effective_workspaces(plan, bindings, stats, sf, PROTOCOL_COST_MODEL)
        == plan
    )


def test_only_small_workspaces_make_scattered_writes_cheap():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = HardAlias("a"), HardAlias("b")
    fill = StaticFillFType(np.float64(0.0))
    n = 1000
    stats: dict[Alias, TensorStats] = {
        a: UniformStats(BaseTensorStats((i, j), {i: n, j: n}, fill), nnz=4 * n),
        b: UniformStats(BaseTensorStats((j, k), {j: n, k: n}, fill), nnz=4 * n),
    }
    _, rhs = _matmul(HardAlias("c"), a, b)
    machine = StatsMachine(UniformStatsFactory(), dict(stats))

    def scattered(order):
        """Whether each prefix of order which scatters writes does so into a
        region beyond the cutoff a workspace could hold."""
        regions = set()
        for n in range(1, len(order) + 1):
            prefix = Prefix(order[:n], rhs, (i, k))
            rows = WriteBasis().get_component_features(prefix, machine)
            if rows[0, 1] > 0:
                regions.add(bool(write_class(rows)[0]))
        return regions

    # Gustavson scatters within each row of c, n entries, which a workspace
    # for the row could hold.
    assert scattered((i, j, k)) == {False}
    # An outer product scatters from the outermost loop into all n * n entries.
    assert scattered((j, i, k)) == {True}


def test_scatter_workspaces_store_the_one_field_scattered_into():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    # Gustavson scatters into k alone, within each row of c.
    plan, _ = _matmul(c, a, b, (i, j, k))
    assert add_1d_workspaces(plan, {}) == add_workspace(plan, c, 1, HardAlias("c_ws"))

    # The j loop scatters into both k and m, so a workspace would store two.
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k, m)))),
        (j,),
    )
    query = fuse_loop_order(Query(Table(c, (i, k, m)), rhs), (i, j, k, m))
    plan = Plan((query, Produces((c,))))
    assert add_1d_workspaces(plan, {}) == plan


def test_accumulators_hold_each_entry_an_inner_loop_reduces_into():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    # An inner product reduces over j into c[i, k], which j leaves fixed.
    plan, rhs = _matmul(c, a, b, (i, k, j))
    assert accumulator_depth(plan, c) == 2
    acc = FusedAlias(HardAlias("c_acc"), 2)
    tensor = fl.asarray(np.ones((2, 2)))
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype, b: tensor.ftype}
    staged, formats = add_accumulators(plan, bindings)
    # The accumulator holds one element, which the kernel keeps to itself.
    assert formats == {acc.unfused: temp_0d(element(0.0, ftype(np.float64)))}
    assert staged == Plan(
        (
            Fuse(
                i,
                Fuse(
                    k,
                    Plan(
                        (
                            Fuse(j, Query(Table(acc, (i, k)), rhs)),
                            Query(Table(c, (i, k)), Table(acc, (i, k))),
                        )
                    ),
                ),
            ),
            Produces((c,)),
        )
    )
    LoopOrderedForm.validate_inputs(staged, bindings, {}, DCStatsFactory())

    # Where j scatters into c, or c is written in the innermost loop, no
    # entry stays fixed across a loop.
    for order in ((i, j, k), (j, i, k)):
        plan, _ = _matmul(c, a, b, order)
        assert accumulator_depth(plan, c) is None
        assert add_accumulators(plan, bindings) == (plan, {})
    elementwise = Plan(
        (
            fuse_loop_order(Query(Table(c, (i, j)), Table(a, (i, j))), (i, j)),
            Produces((c,)),
        )
    )
    assert accumulator_depth(elementwise, c) is None


@pytest.mark.parametrize(("nnz", "pays"), [(1000, False), (500_000, True)])
def test_accumulators_pay_where_their_loops_visit_every_point(nnz, pays):
    # c[i, j] = sum_k a[i, j] * b[i, k] * d[k, j], an SDDMM, loops over the
    # entries of the sparse rows of a, but an accumulator for each (i, j) is
    # reset and copied at every point, so it pays only where a is dense enough
    # that the reduction visits as many terms as c has points.
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, d, c = (HardAlias(name) for name in "abdc")
    n, m = 1000, 4
    fill = StaticFillFType(np.float64(0.0))
    sizes = {i: n, j: n, k: m}

    def uniform(fields, count):
        dims = {idx: sizes[idx] for idx in fields}
        return UniformStats(BaseTensorStats(fields, dims, fill), nnz=count)

    stats: dict[Alias, TensorStats] = {
        a: uniform((i, j), nnz),
        b: uniform((i, k), n * m),
        d: uniform((k, j), m * n),
    }
    shape = (ftype(np.intp),) * 2
    dense = format_ftype([DenseLevelOption()] * 2, fill, shape)
    rows = format_ftype([DenseLevelOption(), SparseListLevelOption()], fill, shape)
    bindings: dict[Alias, TensorFType] = {a: rows, b: dense, d: dense}
    terms = MapJoin(
        Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (i, k)), Table(d, (k, j)))
    )
    query = Query(
        Table(c, (i, j)), Aggregate(Literal(ffuncs.add), Literal(0.0), terms, (k,))
    )
    plan = Plan((fuse_loop_order(query, (i, j, k)), Produces((c,))))
    sf = UniformStatsFactory()
    assert accumulator_pays(plan, query, bindings, stats, sf) == pays
    # Where a stores its rows densely, the loops visit every point anyway.
    bindings[a] = dense
    assert accumulator_pays(plan, query, bindings, stats, sf)


def test_workspacer_accumulates_row_sums_which_merge_into_one_pass():
    # Two sums of each row of a, accumulated in workspaces which store nothing,
    # still merge into one loop over a, which copies each sum once per row.
    i, j, j_2 = Field("i"), Field("j"), Field("j_2")
    a, s, t = HardAlias("a"), HardAlias("s"), HardAlias("t")
    rng = np.random.default_rng(0)
    data = rng.random((6, 5)) * (rng.random((6, 5)) < 0.5)
    tensor = fl.asarray(data)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i, j))}
    sums = (
        Query(
            Table(s, (i,)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Table(a, (i, j)), (j,)),
        ),
        Query(
            Table(t, (i,)),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0.0),
                MapJoin(Literal(ffuncs.mul), (Table(a, (i, j_2)), Table(a, (i, j_2)))),
                (j_2,),
            ),
        ),
    )
    plan = Plan(
        (
            fuse_loop_order(sums[0], (i, j)),
            fuse_loop_order(sums[1], (i, j_2)),
            Produces((s, t)),
        )
    )
    capture = LogicCapture()
    scheduler = LogicWorkspacer(
        LogicFuseMerger(GalleyFormatter(CompilerFormLowerer(capture))),
        format_cost=galley_format_cost,
    )
    scheduler(plan, {a: tensor.ftype}, stats, sf)
    s_acc, t_acc = FusedAlias(HardAlias("s_acc"), 1), FusedAlias(HardAlias("t_acc"), 1)
    match capture.last_prgm:
        case Plan(
            (
                Fuse(
                    _,
                    Plan(
                        (
                            Fuse(
                                _,
                                Plan(
                                    (
                                        Query(Table(acc_1, _), _),
                                        Query(Table(acc_2, _), _),
                                    )
                                ),
                            ),
                            Query(Table(lhs_1, _), Table(src_1, _)),
                            Query(Table(lhs_2, _), Table(src_2, _)),
                        )
                    ),
                ),
                Produces(_),
            )
        ):
            assert (acc_1, acc_2, src_1, src_2) == (s_acc, t_acc, s_acc, t_acc)
            assert (lhs_1, lhs_2) == (s, t)
        case prgm:
            raise AssertionError(f"Expected one pass over a: {prgm}")
    # Each accumulator stores only its element, which the kernel keeps to itself.
    for acc in (s_acc, t_acc):
        fmt = capture.last_bindings[acc.unfused]
        assert isinstance(fmt, Temp0DFType)
        assert fmt.ndim == 0

    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    compiler = NotationCompiler(
        NumbaCompiler(), ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify())
    )
    args = [
        tensor if var == a else fmt.construct((0,) * fmt.ndim)
        for var, fmt in capture.last_bindings.items()
    ]
    result_s, result_t = compiler(program).main(*args)
    np.testing.assert_allclose(result_s.to_numpy(), data.sum(axis=1))
    np.testing.assert_allclose(result_t.to_numpy(), (data * data).sum(axis=1))


def test_add_reformat_copies_an_input_before_the_program():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c, copy = (HardAlias(name) for name in ("a", "b", "c", "a_copy"))
    plan, _ = _matmul(c, a, b, (i, k, j))
    staged = add_reformat(plan, a, (i, j), copy)

    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(copy, (i, j)), Table(b, (j, k)))),
        (j,),
    )
    assert staged == Plan(
        (
            fuse_loop_order(Query(Table(copy, (i, j)), Table(a, (i, j))), (i, j)),
            fuse_loop_order(Query(Table(c, (i, k)), rhs), (i, k, j)),
            Produces((c,)),
        )
    )
    tensor = fl.asarray(np.ones((2, 2)))
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype, b: tensor.ftype}
    LoopOrderedForm.validate_inputs(staged, bindings, {}, DCStatsFactory())


def test_reformats_copy_inputs_where_they_cost_less():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    rng = np.random.default_rng(0)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(rng.random((8, 8))), (i, j)),
        b: sf(fl.asarray(rng.random((8, 8))), (j, k)),
    }
    fill = StaticFillFType(np.float64(0.0))
    shape = (ftype(np.intp),) * 2
    bindings: dict[Alias, TensorFType] = {
        a: format_ftype([DenseLevelOption()] * 2, fill, shape),
        b: format_ftype([SparseListLevelOption()] * 2, fill, shape),
    }
    # Looping over k before j reads a once for each k, so where the visits of
    # dense levels cost, a copy of a read once into another format pays.
    plan, _ = _matmul(c, a, b, (i, k, j))
    dense_visits = CostModel(
        PROTOCOL_BASIS,
        [float(name == "dense.follow.visits") for name in PROTOCOL_BASIS.feature_names],
    )
    staged = add_cost_effective_reformats(plan, bindings, stats, sf, dense_visits)
    copy = HardAlias("a_copy")
    assert staged == add_reformat(plan, a, (i, j), copy)
    # Where nothing costs, no copy pays for itself.
    free = CostModel(PROTOCOL_BASIS, np.zeros(PROTOCOL_BASIS.n_features))
    assert add_cost_effective_reformats(plan, bindings, stats, sf, free) == plan


def test_galley_format_cost_prices_the_formats_a_galley_formatter_chooses():
    # C[t, i] = sum_j A[t, j] B[j, i], row by row through a workspace fused
    # over t: the formats galley_format_cost prices are those a GalleyFormatter
    # gives the same program.
    t, j, i = fields = (Field("t"), Field("j"), Field("i"))
    rng = np.random.default_rng(0)
    n = 30
    a_data, b_data = np.zeros((n, n)), np.zeros((n, n))
    a_data[np.arange(n), rng.integers(0, n, n)] = 1.0
    b_data[np.arange(n), rng.integers(0, n, n)] = 1.0
    a, b, c, ws = (HardAlias(name) for name in ("a", "b", "c", "ws"))
    sf = DCStatsFactory()
    stats: dict[Alias, DCStats] = {
        a: sf(fl.asarray(a_data), (t, j)),
        b: sf(fl.asarray(b_data), (j, i)),
    }
    bindings: dict[Alias, TensorFType] = {
        a: fl.asarray(a_data).ftype,
        b: fl.asarray(b_data).ftype,
    }
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(a, (t, j)), Table(b, (j, i)))),
        (j,),
    )
    plan = Plan(
        (
            fuse_loop_order(Query(Table(c, (t, i)), rhs), fields),
            Produces((c,)),
        )
    )
    staged = add_workspace(plan, c, 1, ws)
    formats, cost = galley_format_cost(staged, bindings, stats, sf, PROTOCOL_COST_MODEL)
    _, formatted = GalleyFormatter(MockLogicLoader()).lower(
        staged, dict(bindings), dict(stats), sf
    )
    assert formats == {alias: formatted[alias] for alias in (c, ws)}
    assert cost > 0


def test_compile_julia_galley_prices_workspaces_with_galley_formats():
    stage: Any = default_schedulers.COMPILE_JULIA_GALLEY
    while not isinstance(stage, LogicWorkspacer):
        stage = stage.ctx
    assert stage.format_cost is galley_format_cost
