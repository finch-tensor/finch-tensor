import pytest

import numpy as np
import scipy.sparse as sps

import finch as fl
from finch import ffuncs
from finch.algebra import StaticFillFType, TensorFType, ftype
from finch.autoschedule import CompilerFormLowerer, NotationGenerator
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import PROTOCOL_COST_MODEL, Prefix, WriteBasis
from finch.autoschedule.formatter.smart_formatter import CostFormatter
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DCStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.tensor_stats.uniform_stats import UniformStats
from finch.autoschedule.util import fuse_loop_order
from finch.autoschedule.workspacer import (
    LogicWorkspacer,
    add_scatter_workspaces,
    add_workspace,
    add_workspaces,
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
    element,
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


def _order(tensor_a, tensor_b, workspaces):
    """Format the Gustavson matmul of tensor_a and tensor_b, looping over
    (i, j, k), with or without the workspacer before the formatter."""
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(tensor_a, (i, j)),
        b: sf(tensor_b, (j, k)),
    }
    capture = LogicCapture()
    plan, _ = _matmul(c, a, b, (i, j, k))
    formatter = CostFormatter(CompilerFormLowerer(capture))
    (LogicWorkspacer(formatter) if workspaces else formatter)(
        plan, {a: tensor_a.ftype, b: tensor_b.ftype}, stats, sf
    )
    return capture


def test_workspacer_accumulates_rows_in_a_workspace():
    i, k = Field("i"), Field("k")
    c = HardAlias("c")
    # The output is sparse enough that a workspace for each row costs less
    # than storing the output densely.
    (tensor_a, _), (tensor_b, _) = (
        _random_csr(10_000, 4, 0),
        _random_csr(10_000, 4, 1),
    )

    # The loops scatter into the columns of c, for each of its rows, so without
    # a workspace its columns can't be sorted lists.
    capture = _order(tensor_a, tensor_b, workspaces=False)
    fmt = capture.last_bindings[c]
    assert not isinstance(fmt.lvl_t.lvl_t, fl.SparseListLevelFType)

    # With the workspacer, the writes scatter into a workspace for each row,
    # which is copied into c in order.
    capture = _order(tensor_a, tensor_b, workspaces=True)
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


def test_workspace_kernels_compute_the_product():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c, w = (HardAlias(name) for name in "abcw")
    (tensor_a, csr_a), (tensor_b, csr_b) = (
        _random_csr(200, 4, 0),
        _random_csr(200, 4, 1),
    )
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(tensor_a, (i, j)),
        b: sf(tensor_b, (j, k)),
    }
    # The product is dense enough that the workspacer would store c densely,
    # so the workspace for each row is staged without pricing it.
    plan, _ = _matmul(c, a, b, (i, j, k))
    staged = add_workspace(plan, c, 1, w)
    capture = LogicCapture()
    CostFormatter(CompilerFormLowerer(capture))(
        staged, {a: tensor_a.ftype, b: tensor_b.ftype}, stats, sf
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    bindings = [
        tensor_a,
        tensor_b,
        capture.last_bindings[w].construct((0,)),
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
    assert add_workspaces(plan, {}, stats, sf, PROTOCOL_COST_MODEL) == plan


@pytest.mark.parametrize(
    ("n", "per_row", "staged"),
    [
        (10_000, 4, True),
        # A product which fills most of its output costs less stored densely.
        (256, 16, False),
    ],
)
def test_workspacer_keeps_workspaces_which_cost_less(n, per_row, staged):
    (tensor_a, _), (tensor_b, _) = (
        _random_csr(n, per_row, 0),
        _random_csr(n, per_row, 1),
    )
    capture = _order(tensor_a, tensor_b, workspaces=True)
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
    assert add_workspaces(plan, bindings, stats, sf, PROTOCOL_COST_MODEL) == plan


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
    basis = WriteBasis(workspace_limit=100_000)
    machine = StatsMachine(UniformStatsFactory(), dict(stats))

    def writes(order):
        rows = [
            basis.get_component_features(Prefix(order[:n], rhs, (i, k)), machine)
            for n in range(len(order) + 1)
        ]
        totals = np.concatenate(rows).sum(axis=0)
        return dict(zip(basis.feature_names, totals > 0, strict=True))

    # Gustavson writes the rows of c in order, so a workspace for one row of n
    # entries takes the writes which scatter.
    assert writes((i, j, k)) == {
        "sequential_writes": True,
        "scatter_writes": False,
        "workspace_writes": True,
    }
    # An outer product scatters from the outermost loop, so its workspace would
    # hold all n * n entries of c.
    assert writes((j, i, k)) == {
        "sequential_writes": False,
        "scatter_writes": True,
        "workspace_writes": False,
    }


def test_scatter_workspaces_store_the_one_field_scattered_into():
    i, j, k, m = Field("i"), Field("j"), Field("k"), Field("m")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    # Gustavson scatters into k alone, within each row of c.
    plan, _ = _matmul(c, a, b, (i, j, k))
    assert add_scatter_workspaces(plan, {}) == add_workspace(
        plan, c, 1, HardAlias("c_ws")
    )

    # The j loop scatters into both k and m, so a workspace would store two.
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k, m)))),
        (j,),
    )
    query = fuse_loop_order(Query(Table(c, (i, k, m)), rhs), (i, j, k, m))
    plan = Plan((query, Produces((c,))))
    assert add_scatter_workspaces(plan, {}) == plan
