import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import DynamicFillFType, StaticFillFType
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import (
    ByKind,
    DenseLevelOption,
    ElementLevelOption,
    LevelBasis,
    LevelChoice,
    LevelOption,
    LinearCostModel,
    Quadratic,
    SparseHashLevelOption,
)
from finch.autoschedule.cost_models.format_cost_models import alias_blocks, nnz_after
from finch.autoschedule.formatter.galley_formatter import estimate_nnz
from finch.autoschedule.formatter.smart_formatter import (
    CostFormatter,
    IterCostFormatter,
    StorageCostFormatter,
)
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DCStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.tensor_stats.uniform_stats import UniformStats
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
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


@pytest.mark.parametrize(
    "fill_type",
    [
        StaticFillFType(np.float64(0)),
        StaticFillFType(np.float64(1)),
        StaticFillFType(np.float64(-1)),
        DynamicFillFType(fl.float64),
    ],
)
@pytest.mark.parametrize(
    "level_type", [fl.SparseListLevelFType, fl.SparseHashLevelFType]
)
def test_sparse_costs_reduce_support(fill_type, level_type):
    i, j = Field("i"), Field("j")
    stats = UniformStats(BaseTensorStats((i, j), {i: 4, j: 3}, fill_type), nnz=6)
    factory = UniformStatsFactory()
    support = factory.support(stats)

    assert support is not stats
    assert support.fill_type == StaticFillFType(False)
    assert support.index_order == stats.index_order
    assert support.dim_sizes == stats.dim_sizes
    assert support.nnz == stats.nnz
    assert nnz_after(stats, factory, 0) == pytest.approx(3.5)
    assert estimate_nnz(stats, factory, (i,)) == pytest.approx(3.5)

    leaf = fl.element(fill_type, fl.float64)
    level = level_type(fl.dense(leaf))
    pos_size = np.dtype(np.intp).itemsize
    expected = (2 + 3.5) * pos_size + 3.5 * 3 * 8
    assert level.level_cost((i, j), stats, factory, 1, 0) == pytest.approx(expected)
    assert stats.fill_type == fill_type


def _format_matrix(formatter, matrix, fill_type):
    i, j = Field("i"), Field("j")
    A, M = HardAlias("A"), HardAlias("M")
    stats_factory = DCStatsFactory()
    query = fuse_loop_order(Query(Table(A, (i, j)), Table(M, (i, j))), (i, j))
    machine = StatsMachine(
        stats_factory, {M: stats_factory(fl.asarray(matrix), (i, j))}
    )
    machine(query)
    shape_type = tuple(fl.ftype(np.intp) for _ in matrix.shape)
    return formatter.get_tensor_ftype(A, query, machine, fill_type, shape_type)


def test_storage_cost_formatter_uses_dense_levels_for_dense_matrx():
    i, j = Field("i"), Field("j")
    A, B = HardAlias("A"), HardAlias("B")
    matrix = np.ones((2, 3))
    tensor = fl.asarray(matrix)
    stats_factory = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {A: stats_factory(tensor, (i, j))}
    capture = LogicCapture()
    formatter = StorageCostFormatter(capture)
    prgm = Plan(
        (
            fuse_loop_order(
                Query(
                    Table(B, (i, j)),
                    Aggregate(
                        Literal(ffuncs.overwrite),
                        Literal(0.0),
                        MapJoin(
                            Literal(ffuncs.add),
                            (Table(A, (i, j)), Table(A, (i, j))),
                        ),
                        (),
                    ),
                ),
                (i, j),
            ),
            Produces((B,)),
        )
    )
    formatter.lower(prgm, {A: tensor.ftype}, stats, stats_factory)
    ftype = capture.last_bindings[B]
    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.DenseLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.DenseLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)

    constructed = ftype.construct((2, 3))
    np.testing.assert_array_equal(constructed.to_numpy(), np.zeros((2, 3)))


def test_storage_cost_formatter_uses_sparse_hash_for_sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    ftype = _format_matrix(
        StorageCostFormatter(), matrix, StaticFillFType(np.float64(0.0))
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)


def test_iter_cost_formatter_uses_sparse_hash_for_sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    ftype = _format_matrix(
        IterCostFormatter(), matrix, StaticFillFType(np.float64(0.0))
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)


def _cost_output_pattern(formatter_cls, matrix):
    ftype = _format_matrix(formatter_cls(), matrix, StaticFillFType(np.float64(0.0)))

    lvl = ftype.lvl_t
    pattern = []
    while not isinstance(lvl, fl.ElementLevelFType):
        match lvl:
            case fl.DenseLevelFType():
                pattern.append("dense")
            case fl.SparseHashLevelFType():
                pattern.append("sparse_hash")
            case _:
                raise AssertionError(f"Unexpected level:{lvl}")
        lvl = lvl.lvl_t
    return tuple(pattern)


def test_check_expected_pattern_using_storage_cost_formatter():
    dense_matrix = np.ones((4, 4))
    sparse_matrix = np.zeros((50, 50))
    sparse_matrix[0, 0] = 1
    sparse_matrix[49, 49] = 1

    assert _cost_output_pattern(StorageCostFormatter, dense_matrix) == (
        "dense",
        "dense",
    )
    assert _cost_output_pattern(StorageCostFormatter, sparse_matrix) == (
        "sparse_hash",
        "sparse_hash",
    )


@pytest.mark.parametrize(
    "formatter_cls, expected",
    [
        (IterCostFormatter, [28, 9, 0]),
        (StorageCostFormatter, [0, 10 * np.dtype(np.intp).itemsize, 40]),
    ],
)
def test_format_cost_uses_depth_and_input_positions(formatter_cls, expected):
    i, j = Field("i"), Field("j")
    a = HardAlias("a")
    tensor = fl.asarray(np.eye(3, 4, dtype=np.float64) * [1, 1, 0, 0])
    query = fuse_loop_order(
        Query(Table(a, (i, j)), Table(Literal(tensor), (i, j))), (i, j)
    )
    machine = StatsMachine(DCStatsFactory())
    machine(query)
    model = formatter_cls().cost_model
    pos_size = np.dtype(np.intp).itemsize

    costs = [
        model.predict_component_cost(
            LevelChoice(query, a, depth, num_pos, option, pos_size), machine
        )
        for depth, num_pos, option in (
            (1, (7,), DenseLevelOption()),
            (1, (7,), SparseHashLevelOption()),
            (2, (5,), ElementLevelOption()),
        )
    ]
    np.testing.assert_allclose(costs, expected)


def test_format_access_features_separate_sequential_and_scatter():
    i, j, k = Field("i"), Field("j"), Field("k")
    X, Y, A, C = HardAlias("X"), HardAlias("Y"), HardAlias("A"), HardAlias("C")
    stats_factory = DCStatsFactory()
    ones = fl.asarray(np.ones((3, 3)))
    machine = StatsMachine(
        stats_factory,
        {X: stats_factory(ones, (i, k)), Y: stats_factory(ones, (k, j))},
    )
    prgm = Plan(
        (
            # A[i, j] = sum_k X[i, k] * Y[k, j], looping over (i, k, j).
            fuse_loop_order(
                Query(
                    Table(A, (i, j)),
                    Aggregate(
                        Literal(ffuncs.add),
                        Literal(0.0),
                        MapJoin(
                            Literal(ffuncs.mul), (Table(X, (i, k)), Table(Y, (k, j)))
                        ),
                        (k,),
                    ),
                ),
                (i, k, j),
            ),
            # C[i, j] = A[i, j], looping over (i, j).
            fuse_loop_order(
                Query(
                    Table(C, (i, j)),
                    Aggregate(Literal(ffuncs.add), Literal(0.0), Table(A, (i, j)), ()),
                ),
                (i, j),
            ),
            Produces((C,)),
        )
    )
    machine(prgm)

    def accessed(depth, option):
        features = LevelBasis(type(option)).get_component_features(
            LevelChoice(prgm, A, depth, (1.0,), option, np.dtype(np.intp).itemsize),
            machine,
        )
        return tuple(features.sum(axis=0)[-4:] > 0)

    # Reading A in loop order is sequential at every level. Writing A is
    # sequential at i, but the k loop makes it scatter at j.
    assert accessed(0, DenseLevelOption()) == (True, False, True, False)
    assert accessed(1, DenseLevelOption()) == (True, False, False, True)
    assert accessed(2, ElementLevelOption()) == (True, False, False, True)


def test_fused_alias_levels_describe_the_largest_slice():
    t, i = Field("t"), Field("i")
    a, scratch = HardAlias("a"), HardAlias("scratch")
    fused = FusedAlias(scratch, 1)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.eye(4)), (t, i))}
    copy = fuse_loop_order(Query(Table(fused, (t, i)), Table(a, (t, i))), (t, i))
    plan = Plan((copy, Produces((fused,))))
    basis = LevelBasis(SparseHashLevelOption)
    formats: dict[Alias, list[LevelOption]] = {
        scratch: [SparseHashLevelOption(), ElementLevelOption()]
    }
    machine = StatsMachine(sf, dict(stats))
    sparse, element = basis.components(plan, formats, machine)

    # Each slice of the diagonal stores the one entry of its row.
    assert (sparse.alias, sparse.depth, sparse.num_pos) == (scratch, 0, (1.0,))
    assert element.depth == 1
    assert element.num_pos == pytest.approx((1.0,))
    size = np.dtype(np.intp).itemsize
    features = basis.get_component_features(sparse, machine)
    np.testing.assert_allclose(features[0, :4], [1, 1, 2 * size, size])


def test_format_costs_sum_the_union_of_each_block():
    i = Field("i")
    a, b, c, out = (HardAlias(name) for name in ("a", "b", "c", "out"))
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(np.array([1.0, 0.0, 0.0, 0.0])), (i,)),
        b: sf(fl.asarray(np.array([0.0, 1.0, 0.0, 0.0])), (i,)),
        c: sf(fl.asarray(np.array([0.0, 0.0, 1.0, 1.0])), (i,)),
    }
    # The first block defines out and updates it, and the second redefines it.
    define = Query(Table(out, (i,)), Table(a, (i,)))
    update = QueryInto(Table(out, (i,)), Literal(ffuncs.add), Table(b, (i,)))
    redefine = Query(Table(out, (i,)), Table(c, (i,)))
    queries = (define, update, redefine)
    plan = Plan(
        (*(fuse_loop_order(query, (i,)) for query in queries), Produces((out,)))
    )
    machine = StatsMachine(sf, dict(stats))
    states = []
    for query in queries:
        machine(query)
        states.append(machine(out))
    union = sf.mapjoin(ffuncs.or_, sf.support(states[0]), sf.support(states[1]))
    nnz = [union.estimate_non_fill_values(), states[2].estimate_non_fill_values()]
    assert nnz[0] > states[1].estimate_non_fill_values()

    machine = StatsMachine(sf, dict(stats))
    blocks = alias_blocks(out, plan, machine)
    assert [block.estimate_non_fill_values() for block, _ in blocks] == nnz
    assert all(block.fill_type == states[0].fill_type for block, _ in blocks)

    basis = LevelBasis(SparseHashLevelOption)
    size = np.dtype(np.intp).itemsize
    choice = LevelChoice(plan, out, 0, (1.0, 1.0), SparseHashLevelOption(), size)
    features = basis.get_component_features(choice, machine)
    # An allocation row for each block, then a write row for each query.
    assert features.shape == (5, 8)
    np.testing.assert_array_equal(
        features[:2, :4], [[1, n, 2 * size, n * size] for n in nnz]
    )
    assert not features[:2, 4:].any()
    assert not features[2:, :4].any()
    assert (features[2:, 6] > 0).all()


def test_nonlinear_format_cost_retains_allocations_and_query_accesses():
    i = Field("i")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    tensor = fl.asarray(np.ones(3))
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i,))}
    plan = Plan(
        (
            fuse_loop_order(Query(Table(b, (i,)), Table(a, (i,))), (i,)),
            fuse_loop_order(Query(Table(c, (i,)), Table(b, (i,))), (i,)),
            Produces((c,)),
        )
    )
    bases = {
        "dense": LevelBasis(DenseLevelOption),
        "element": LevelBasis(ElementLevelOption),
    }
    basis = ByKind(bases)
    cost = LinearCostModel(
        ByKind({kind: Quadratic(sub) for kind, sub in bases.items()}), np.ones(72)
    )
    formats: dict[Alias, list[LevelOption]] = {
        b: [DenseLevelOption(), ElementLevelOption()]
    }
    features = basis.get_features(plan, formats, stats, sf)
    # Allocation, write, and read rows for both the dense and element levels.
    assert features.shape == (6, 12)
    np.testing.assert_array_equal(features[1:3, 2:6], [[0, 0, 3, 0], [3, 0, 0, 0]])
    np.testing.assert_array_equal(features[4:6, 8:12], [[0, 0, 3, 0], [3, 0, 0, 0]])
    expected = sum(row.sum() ** 2 for row in features)
    assert cost.predict_cost(plan, formats, stats, sf) == expected
    machine = StatsMachine(sf, dict(stats))
    machine(plan)
    assert expected == sum(
        cost.predict_component_cost(
            LevelChoice(plan, b, depth, num_pos, option, np.dtype(np.intp).itemsize),
            machine,
        )
        for depth, num_pos, option in (
            (0, (1,), DenseLevelOption()),
            (1, (3,), ElementLevelOption()),
        )
    )
    assert cost.get_features(Plan(()), {}, {}, sf).shape == (0, 72)
    assert cost.predict_cost(Plan(()), {}, {}, sf) == 0.0
    formatter = CostFormatter(cost_model=cost, options=(DenseLevelOption(),))
    fmt = formatter.get_tensor_ftype(
        b, plan, machine, tensor.ftype.fill_type, (fl.ftype(np.intp),)
    )
    assert isinstance(fmt.lvl_t, fl.DenseLevelFType)


def test_format_selection_uses_custom_model():
    i = Field("i")
    a = HardAlias("a")
    tensor = fl.asarray(np.ones(3))
    query = fuse_loop_order(Query(Table(a, (i,)), Table(Literal(tensor), (i,))), (i,))
    machine = StatsMachine(DCStatsFactory())
    machine(query)
    bases = {
        option_type.kind: LevelBasis(option_type)
        for option_type in (DenseLevelOption, SparseHashLevelOption, ElementLevelOption)
    }
    linear = LinearCostModel(ByKind(bases), np.zeros(ByKind(bases).n_features))
    # Charge the square of the dense slots, the second storage feature.
    slots = ByKind({**bases, "dense": Quadratic(bases["dense"])})
    coefficients = np.zeros(slots.n_features)
    coefficients[bases["dense"].n_features + 1] = 1.0
    nonlinear = LinearCostModel(slots, coefficients)

    def choose(cost):
        return CostFormatter(cost_model=cost).get_tensor_ftype(
            a, query, machine, tensor.ftype.fill_type, (fl.ftype(np.intp),)
        )

    assert isinstance(choose(linear).lvl_t, fl.DenseLevelFType)
    assert isinstance(choose(nonlinear).lvl_t, fl.SparseHashLevelFType)
