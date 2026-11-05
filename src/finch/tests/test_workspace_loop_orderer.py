import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import StaticFillFType, TensorFType, ftype
from finch.autoschedule import CompilerFormLowerer, NotationGenerator
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import Prefix, WriteBasis
from finch.autoschedule.formatter.smart_formatter import WorkspaceCostFormatter
from finch.autoschedule.loop_orderer import (
    BFSLoopOrderer,
    CostedWorkspaceLoopOrderer,
    WorkspaceLoopOrderer,
)
from finch.autoschedule.loop_orderer.loop_order_workspace import (
    add_workspace,
    workspace_depth,
)
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DCStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.tensor_stats.uniform_stats import UniformStats
from finch.autoschedule.util import fuse_loop_order
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


def _sparse_matmul(n, per_row, seed=0):
    # Two n by n matrices with per_row entries in each row, in random columns.
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(2):
        mat = np.zeros((n, n))
        for row in range(n):
            mat[row, rng.choice(n, per_row, replace=False)] = rng.random(per_row)
        data.append(mat)
    return data


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


def _csr(mat):
    # A dense root over sorted lists of the non-zeros of each row.
    rows, cols = np.nonzero(mat)
    ptr = np.zeros(mat.shape[0] + 1, dtype=np.intp)
    np.add.at(ptr, rows + 1, 1)
    fmt = element(np.float64(0), ftype(np.float64), ftype(np.intp))
    lvl = SparseListLevel(
        ElementLevel(fmt, NumpyBuffer(mat[rows, cols])),
        np.intp(mat.shape[1]),
        NumpyBuffer(np.cumsum(ptr)),
        NumpyBuffer(cols.astype(np.intp)),
    )
    return FiberTensor(DenseLevel(lvl, np.intp(mat.shape[0])))


def _order(orderer_cls, tensor_a, tensor_b):
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(tensor_a, (i, j)),
        b: sf(tensor_b, (j, k)),
    }
    capture = LogicCapture()
    plan, _ = _matmul(c, a, b)
    orderer_cls(WorkspaceCostFormatter(CompilerFormLowerer(capture)))(
        plan, {a: tensor_a.ftype, b: tensor_b.ftype}, stats, sf
    )
    return capture


def test_workspace_loop_orderer_accumulates_rows_in_a_workspace():
    i, j, k = Field("i"), Field("j"), Field("k")
    c = HardAlias("c")
    # Sparse inputs only write their products into the workspace, so that the
    # output is sparse enough for its rows not to be dense.
    data_a, data_b = _sparse_matmul(256, 3)
    tensor_a, tensor_b = _csr(data_a), _csr(data_b)

    # The search scatters into the columns of c, for each of its rows.
    capture = _order(BFSLoopOrderer, tensor_a, tensor_b)
    match capture.last_prgm:
        case Plan((Fuse(i_, Fuse(j_, Fuse(k_, Query(Table(lhs, _), _)))), Produces())):
            assert (i_, j_, k_, lhs) == (i, j, k, c)
        case prgm:
            raise AssertionError(f"Expected the loop order (i, j, k): {prgm}")

    # Copies are propagated before the workspace is added, so its copy stays,
    # and the writes scatter into a workspace for each row instead.
    capture = _order(WorkspaceLoopOrderer, tensor_a, tensor_b)
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
    np.testing.assert_allclose(result.to_scipy().toarray(), data_a @ data_b)


@pytest.mark.parametrize(
    ("loop_order", "limit", "staged"),
    [
        # A slice of a workspace fused over i has the 4 entries of a row.
        (("i", "j", "k"), 4, True),
        (("i", "j", "k"), 3, False),
        # The loops scatter into i, so a workspace would store all of c.
        (("k", "i", "j"), 16, False),
    ],
)
def test_workspace_loop_orderer_adds_small_workspaces_with_fewer_fields(
    loop_order, limit, staged
):
    i, j = Field("i"), Field("j")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(np.ones((4, 4))), (i, j)),
        b: sf(fl.asarray(np.ones((4, 4))), (j, Field("k"))),
    }
    plan, _ = _matmul(c, a, b, tuple(map(Field, loop_order)))
    orderer = WorkspaceLoopOrderer(workspace_limit=limit)
    prgm = orderer.add_workspaces(plan, {}, stats, sf)
    fused = []
    for node in PostOrderDFS(prgm):
        match node:
            case FusedAlias():
                fused.append(node)
    assert bool(fused) == staged


@pytest.mark.parametrize(("per_row", "staged"), [(3, True), (16, False)])
def test_costed_workspace_loop_orderer_keeps_workspaces_which_cost_less(
    per_row, staged
):
    # A sparse product is accumulated row by row in a workspace, but one which
    # fills most of its output costs less written into a dense output.
    data_a, data_b = _sparse_matmul(256, per_row)
    capture = _order(CostedWorkspaceLoopOrderer, _csr(data_a), _csr(data_b))
    fused = []
    for node in PostOrderDFS(capture.last_prgm):
        match node:
            case FusedAlias():
                fused.append(node)
    assert bool(fused) == staged


def test_workspace_loop_orderer_writes_in_order_without_a_workspace():
    # The product of dense matrices is cheapest as an inner product, which
    # writes c in order, so it needs no workspace.
    rng = np.random.default_rng(0)
    data_a, data_b = rng.random((8, 8)), rng.random((8, 8))
    tensor_a, tensor_b = fl.asarray(data_a), fl.asarray(data_b)
    ordered = _order(BFSLoopOrderer, tensor_a, tensor_b).last_prgm
    assert workspace_depth(ordered, HardAlias("c")) is None
    assert _order(WorkspaceLoopOrderer, tensor_a, tensor_b).last_prgm == ordered


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
