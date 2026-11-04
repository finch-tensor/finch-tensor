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
    SparseByteMapLevelOption,
    SparseHashLevelOption,
    SparseListLevelOption,
    allowed_options,
    write_protocols,
)
from finch.autoschedule.cost_models.format_cost_models import (
    alias_blocks,
    nnz_after,
    read_protocols,
)
from finch.autoschedule.formatter.galley_formatter import estimate_nnz
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    CostFormatter,
    IterCostFormatter,
    StorageCostFormatter,
)
from finch.autoschedule.loop_orderer.loop_order_workspace import add_workspace
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
    DefaultProtocol,
    Extrude,
    Field,
    Follow,
    FusedAlias,
    HardAlias,
    Laminate,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
    TensorStats,
    Walk,
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


def test_storage_cost_formatter_uses_sparse_list_for_sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    ftype = _format_matrix(
        StorageCostFormatter(), matrix, StaticFillFType(np.float64(0.0))
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseListLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseListLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)


def test_iter_cost_formatter_uses_sparse_list_for_sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    ftype = _format_matrix(
        IterCostFormatter(), matrix, StaticFillFType(np.float64(0.0))
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseListLevelFType)


def test_cost_formatter_uses_sparse_hash_for_transposed_matrix():
    # A transposed copy writes each column for every row, so it laminates the
    # levels of its output, which a list can't store.
    i, j = Field("i"), Field("j")
    A, M = HardAlias("A"), HardAlias("M")
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 98] = 1.0
    stats_factory = DCStatsFactory()
    query = fuse_loop_order(Query(Table(A, (j, i)), Table(M, (i, j))), (i, j))
    machine = StatsMachine(
        stats_factory, {M: stats_factory(fl.asarray(matrix), (i, j))}
    )
    machine(query)
    shape_type = (fl.ftype(np.intp), fl.ftype(np.intp))
    ftype = StorageCostFormatter().get_tensor_ftype(
        A, query, machine, StaticFillFType(np.float64(0.0)), shape_type
    )

    assert write_protocols(A, 0, query) == {Laminate()}
    assert write_protocols(A, 1, query) == {Laminate()}
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseHashLevelFType)


def test_write_protocols_extrude_fibers_whose_prefix_is_fixed():
    i, j = Field("i"), Field("j")
    a, x, y = HardAlias("a"), HardAlias("x"), HardAlias("y")

    def matvec(loop_order):
        product = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(x, (j,))))
        rhs = Aggregate(Literal(ffuncs.add), Literal(0.0), product, (j,))
        return fuse_loop_order(Query(Table(y, (i,)), rhs), loop_order)

    def update(op):
        return fuse_loop_order(
            QueryInto(Table(y, (i,)), Literal(op), Table(x, (i,))), (i,)
        )

    assert write_protocols(y, 0, matvec((i, j))) == {Extrude()}
    # Each row of y is written again for each column of a.
    assert write_protocols(y, 0, matvec((j, i))) == {Laminate()}
    # An update which reduces into y laminates what it stores, but one which
    # overwrites y starts it again.
    assert write_protocols(y, 0, update(ffuncs.add)) == {Laminate()}
    assert write_protocols(y, 0, update(ffuncs.overwrite)) == {Extrude()}
    assert [
        opt.kind for opt in allowed_options(y, 0, matvec((i, j)), LEVEL_OPTIONS)
    ] == [
        "dense",
        "sparse_list",
        "sparse_hash",
        "sparse_bytemap",
    ]
    assert [
        opt.kind for opt in allowed_options(y, 0, matvec((j, i)), LEVEL_OPTIONS)
    ] == [
        "dense",
        "sparse_hash",
        "sparse_bytemap",
    ]


def test_bytemap_allocates_its_children_densely():
    i, j = Field("i"), Field("j")
    fill_type = StaticFillFType(np.float64(0))
    stats = UniformStats(BaseTensorStats((i, j), {i: 4, j: 3}, fill_type), nnz=6)
    factory = UniformStatsFactory()
    option = SparseByteMapLevelOption()
    pos_size = np.dtype(np.intp).itemsize

    assert option.next_num_pos(stats, factory, 0, 1, ()) == 4
    np.testing.assert_allclose(
        option.get_storage_features(stats, factory, 0, 1, 8, pos_size, 8, ()),
        [1, 4, 3.5, (2 + 3.5) * pos_size, 3.5],
    )


def _cost_output_pattern(formatter_cls, matrix):
    ftype = _format_matrix(formatter_cls(), matrix, StaticFillFType(np.float64(0.0)))

    lvl = ftype.lvl_t
    pattern = []
    while not isinstance(lvl, fl.ElementLevelFType):
        match lvl:
            case fl.DenseLevelFType():
                pattern.append("dense")
            case fl.SparseListLevelFType():
                pattern.append("sparse_list")
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
        "sparse_list",
        "sparse_list",
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
        return tuple(features.sum(axis=0)[-6:] > 0)

    # Reading A in loop order visits its levels at their own loops, densely
    # or not, and steps through their entries, and looks up its values.
    # Writing A is sequential at i, but the k loop makes it scatter at j.
    read = (True, True, True, False)
    assert accessed(0, DenseLevelOption()) == (*read, True, False)
    assert accessed(1, DenseLevelOption()) == (*read, False, True)
    assert accessed(2, ElementLevelOption()) == (False, False, False, True, False, True)
    assert accessed(0, SparseHashLevelOption()) == (*read, True, False)


@pytest.mark.parametrize(
    ("op", "scatters"), [(ffuncs.overwrite, False), (ffuncs.add, True)]
)
def test_format_access_features_scatter_reductions_into_a_level(op, scatters):
    i, j = Field("i"), Field("j")
    X, A = HardAlias("X"), HardAlias("A")
    stats_factory = DCStatsFactory()
    ones = fl.asarray(np.ones((3, 3)))
    machine = StatsMachine(stats_factory, {X: stats_factory(ones, (i, j))})
    copy = Query(
        Table(A, (i, j)),
        Aggregate(Literal(ffuncs.add), Literal(0.0), Table(X, (i, j)), ()),
    )
    update = QueryInto(Table(A, (i, j)), Literal(op), Table(X, (i, j)))
    prgm = Plan(
        (
            fuse_loop_order(copy, (i, j)),
            fuse_loop_order(update, (i, j)),
            Produces((A,)),
        )
    )
    features = LevelBasis(DenseLevelOption).get_component_features(
        LevelChoice(prgm, A, 0, (1.0,), DenseLevelOption(), 8), machine
    )
    # Both queries write A in order, but an update which reduces into A
    # laminates what it stores, as write_protocols says.
    assert write_protocols(A, 0, prgm) == (
        {Extrude(), Laminate()} if scatters else {Extrude()}
    )
    assert tuple(features.sum(axis=0)[-2:] > 0) == (True, scatters)


def test_format_access_features_reach_a_level_through_its_parents():
    i, k = Field("i"), Field("k")
    X, C = HardAlias("X"), HardAlias("C")
    stats_factory = DCStatsFactory()
    machine = StatsMachine(
        stats_factory, {X: stats_factory(fl.asarray(np.ones((3, 3))), (i, k))}
    )
    # C[k, i] = X[i, k], looping over (k, i), against the order X stores.
    query = Query(
        Table(C, (k, i)),
        Aggregate(Literal(ffuncs.add), Literal(0.0), Table(X, (i, k)), ()),
    )
    prgm = Plan((fuse_loop_order(query, (k, i)), Produces((C,))))

    def reads(depth, option):
        features = LevelBasis(type(option)).get_component_features(
            LevelChoice(prgm, X, depth, (1.0,), option, np.dtype(np.intp).itemsize),
            machine,
        )
        return tuple(features.sum(axis=0)[-6:-2])

    # The i level is visited at its own loop, which steps through its 9
    # entries over the iterations of k, while the k level and the values of X
    # are reached through i, so each of the 9 iterations looks them up.
    assert reads(0, SparseHashLevelOption()) == (9, 9, 9, 0)
    assert reads(0, DenseLevelOption()) == (9, 9, 9, 0)
    assert reads(1, DenseLevelOption()) == (0, 0, 0, 9)
    assert reads(2, ElementLevelOption()) == (0, 0, 0, 9)


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


def test_inputs_are_formatted_from_their_bindings():
    i, j = Field("i"), Field("j")
    a, b = HardAlias("a"), HardAlias("b")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.eye(4)), (i, j))}
    copy = fuse_loop_order(Query(Table(b, (i, j)), Table(a, (i, j))), (i, j))
    plan = Plan((copy, Produces((b,))))
    basis = LevelBasis(DenseLevelOption)
    formats: dict[Alias, list[LevelOption]] = {
        a: [DenseLevelOption(), SparseHashLevelOption(), ElementLevelOption()]
    }
    dense, sparse, element = basis.components(
        plan, formats, StatsMachine(sf, dict(stats))
    )

    # The input stores its 4 rows, and then the one entry of each.
    assert [(c.depth, c.num_pos) for c in (dense, sparse, element)] == [
        (0, (1.0,)),
        (1, (4.0,)),
        (2, (4.0,)),
    ]
    features = basis.get_component_features(dense, StatsMachine(sf, dict(stats)))
    # The binding allocates the level once, and the copy visits it at its own
    # loop.
    assert features.shape[0] == 2
    assert tuple(features[1, -6:] > 0) == (True, True, True, False, False, False)


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
    assert features.shape == (5, 10)
    np.testing.assert_array_equal(
        features[:2, :4], [[1, n, 2 * size, n * size] for n in nnz]
    )
    assert not features[:2, 4:].any()
    assert not features[2:, :4].any()
    # The definitions write out in order, and the update reduces into what it
    # stores, which laminates it.
    np.testing.assert_array_equal(
        features[2:, 8:] > 0, [[True, False], [False, True], [True, False]]
    )


def test_format_access_features_ignore_protocols():
    i = Field("i")
    X, Y, C = HardAlias("X"), HardAlias("Y"), HardAlias("C")
    sf = DCStatsFactory()
    machine = StatsMachine(
        sf,
        {
            X: sf(fl.asarray(np.array([0.0, 1.0, 0.0, 0.0])), (i,)),
            Y: sf(fl.asarray(np.ones(4)), (i,)),
        },
    )

    def reads(protocol):
        product = MapJoin(
            Literal(ffuncs.mul), (Table(X, (i,)), Table(Y, (i,), (protocol,)))
        )
        query = Query(
            Table(C, (i,)), Aggregate(Literal(ffuncs.add), Literal(0.0), product, ())
        )
        prgm = Plan((fuse_loop_order(query, (i,)), Produces((C,))))
        features = LevelBasis(DenseLevelOption).get_component_features(
            LevelChoice(prgm, Y, 0, (1.0,), DenseLevelOption(), 8), machine
        )
        return tuple(features.sum(axis=0)[-6:-2])

    # Y is visited where X stores its 1, even if Y were dense, and stores the
    # 4 entries a walk would step through, whatever protocol it is read with.
    assert reads(Walk()) == (1, 1, 4, 0)
    assert reads(Follow()) == (1, 1, 4, 0)
    assert reads(DefaultProtocol()) == (1, 1, 4, 0)


def test_dense_levels_are_only_followed():
    i = Field("i")
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    tensor = fl.asarray(np.ones(3))
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i,))}

    def plan(protocols):
        return Plan(
            (
                fuse_loop_order(Query(Table(b, (i,)), Table(a, (i,))), (i,)),
                fuse_loop_order(Query(Table(c, (i,)), Table(b, (i,), protocols)), (i,)),
                Produces((c,)),
            )
        )

    # A default read takes whichever protocol the level supports, so it
    # constrains nothing, but a walk rules a dense level out.
    assert read_protocols(b, 0, plan(())) == set()
    assert read_protocols(b, 0, plan((Follow(),))) == {Follow()}
    assert read_protocols(b, 0, plan((Walk(),))) == {Walk()}
    formatter = CostFormatter(options=(DenseLevelOption(),))
    shape_type = (fl.ftype(np.intp),)
    with pytest.raises(ValueError, match="No level option supports"):
        formatter.get_tensor_ftype(
            b,
            plan((Walk(),)),
            StatsMachine(sf, dict(stats)),
            tensor.ftype.fill_type,
            shape_type,
        )
    for protocols in ((), (Follow(),)):
        fmt = formatter.get_tensor_ftype(
            b,
            plan(protocols),
            StatsMachine(sf, dict(stats)),
            tensor.ftype.fill_type,
            shape_type,
        )
        assert isinstance(fmt.lvl_t, fl.DenseLevelFType)


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
        ByKind({kind: Quadratic(sub) for kind, sub in bases.items()}), np.ones(145)
    )
    formats: dict[Alias, list[LevelOption]] = {
        b: [DenseLevelOption(), ElementLevelOption()]
    }
    features = basis.get_features(plan, formats, stats, sf)
    # Allocation, write, and read rows for both the dense and element levels:
    # the dense level is visited, and the values are looked up.
    assert features.shape == (6, 17)
    np.testing.assert_array_equal(
        features[1:3, 3:9], [[0, 0, 0, 0, 3, 0], [3, 3, 3, 0, 0, 0]]
    )
    np.testing.assert_array_equal(
        features[4:6, 11:17], [[0, 0, 0, 0, 3, 0], [0, 0, 0, 3, 0, 0]]
    )
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
    assert cost.get_features(Plan(()), {}, {}, sf).shape == (0, 145)
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
        options = (DenseLevelOption(), SparseHashLevelOption())
        return CostFormatter(cost_model=cost, options=options).get_tensor_ftype(
            a, query, machine, tensor.ftype.fill_type, (fl.ftype(np.intp),)
        )

    assert isinstance(choose(linear).lvl_t, fl.DenseLevelFType)
    assert isinstance(choose(nonlinear).lvl_t, fl.SparseHashLevelFType)


def _matmul(c, a, b, loop_order):
    # c[i, k] = sum_j a[i, j] * b[j, k], looping over loop_order.
    i, j, k = Field("i"), Field("j"), Field("k")
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k)))),
        (j,),
    )
    return Plan(
        (fuse_loop_order(Query(Table(c, (i, k)), rhs), loop_order), Produces((c,)))
    ), rhs


def _block_matmul(n, m):
    # Every row of a reads the same m rows of b, so the n * m * m products land
    # on only n * m entries of the output.
    data_a = np.zeros((n, n))
    data_a[:, :m] = 1
    data_b = np.zeros((n, n))
    data_b[:m, :m] = np.arange(1, m * m + 1).reshape(m, m)
    return data_a, data_b


def test_sparse_list_levels_reject_scatter_writes():
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c, w = (HardAlias(name) for name in "abcw")
    plan, _ = _matmul(c, a, b, (i, j, k))
    staged = add_workspace(plan, c, 1, w)
    options = [SparseListLevelOption()]

    # The query scatters into the columns of c, but the copy writes them in
    # order, and both write the rows of c in order.
    assert allowed_options(c, 1, plan, options) == []
    assert allowed_options(c, 1, staged, options) == options
    assert allowed_options(c, 0, plan, options) == options


@pytest.mark.parametrize(
    "option, expected",
    [
        (DenseLevelOption(), [1, 4, 16]),
        (SparseByteMapLevelOption(), [1, 4, 1, 3 * np.dtype(np.intp).itemsize, 4]),
    ],
)
def test_workspace_levels_reset_each_slice(option, expected):
    t, i = Field("t"), Field("i")
    a, scratch = HardAlias("a"), HardAlias("scratch")
    fused = FusedAlias(scratch, 1)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.eye(4)), (t, i))}
    copy = fuse_loop_order(Query(Table(fused, (t, i)), Table(a, (t, i))), (t, i))
    plan = Plan((copy, Produces((fused,))))
    basis = LevelBasis(type(option))
    machine = StatsMachine(sf, dict(stats))
    choice = LevelChoice(plan, scratch, 0, (1.0,), option, np.dtype(np.intp).itemsize)

    # Each of the 4 slices of the diagonal is reset before it is written.
    features = basis.get_component_features(choice, machine)
    np.testing.assert_allclose(features[0, : len(option.feature_names)], expected)
