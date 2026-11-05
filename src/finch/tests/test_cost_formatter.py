import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import DynamicFillFType, StaticFillFType
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    CostModel,
    DenseLevelOption,
    ElementLevelOption,
    LevelOption,
    ProtocolBasis,
    SparseByteMapLevelOption,
    SparseHashLevelOption,
    SparseListLevelOption,
    StorageChoice,
    alias_blocks,
    allowed_options,
    level_options,
    nnz_after,
    read_protocols,
    write_protocols,
)
from finch.autoschedule.formatter.galley_formatter import estimate_nnz
from finch.autoschedule.formatter.smart_formatter import (
    LEVEL_OPTIONS,
    CostFormatter,
    format_ftype,
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


def lists(ndim):
    """The format of a tensor of ndim list levels of float64."""
    fill_type = StaticFillFType(np.float64(0.0))
    return format_ftype(
        [SparseListLevelOption()] * ndim, fill_type, (fl.ftype(np.intp),) * ndim
    )


def formats_of(prgm, bindings, stats, sf, **kwargs):
    """The kinds of the levels of each tensor the formatter formats."""
    capture = LogicCapture()
    CostFormatter(capture, **kwargs).lower(prgm, dict(bindings), dict(stats), sf)
    return {
        alias: [option.kind for option in level_options(fmt) or ()]
        for alias, fmt in capture.last_bindings.items()
        if alias not in bindings
    }


def sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 98] = 1.0
    return matrix


def test_a_dense_matrix_is_copied_into_dense_levels():
    i, j = Field("i"), Field("j")
    A, B = HardAlias("A"), HardAlias("B")
    tensor = fl.asarray(np.ones((2, 3)))
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {A: sf(tensor, (i, j))}
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
    capture = LogicCapture()
    CostFormatter(capture).lower(prgm, {A: tensor.ftype}, stats, sf)
    fmt = capture.last_bindings[B]
    # Every slot of B is written, which a dense level has ready.
    assert [option.kind for option in level_options(fmt) or ()] == ["dense", "dense"]
    np.testing.assert_array_equal(fmt.construct((2, 3)).to_numpy(), np.zeros((2, 3)))


def test_an_output_read_on_its_own_is_stored_as_sparse_as_its_entries():
    # T = M, then y[i] = sum_j T[i, j], with 2 entries of M in 100 by 100.
    i, j = Field("i"), Field("j")
    M, T, y = HardAlias("M"), HardAlias("T"), HardAlias("y")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {M: sf(fl.asarray(sparse_matrix()), (i, j))}
    copy = fuse_loop_order(Query(Table(T, (i, j)), Table(M, (i, j))), (i, j))
    sums = fuse_loop_order(
        Query(
            Table(y, (i,)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Table(T, (i, j)), (j,)),
        ),
        (i, j),
    )
    prgm = Plan((copy, sums, Produces((y,))))
    # The copy writes T where M has entries, and the sums read T alone, so a
    # dense T would make them span the whole matrix, where lists step through
    # its 2 entries.
    assert formats_of(prgm, {M: lists(2)}, stats, sf)[T] == [
        "sparse_list",
        "sparse_list",
    ]


def test_a_transposed_copy_laminates_its_output():
    # A[j, i] = M[i, j] under (i, j), then y[j] = sum_i A[j, i].
    i, j = Field("i"), Field("j")
    A, M, y = HardAlias("A"), HardAlias("M"), HardAlias("y")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {M: sf(fl.asarray(sparse_matrix()), (i, j))}
    copy = fuse_loop_order(Query(Table(A, (j, i)), Table(M, (i, j))), (i, j))
    # Each column of A is written again for each row of M, so A is laminated,
    # which a list can't be.
    assert write_protocols(A, 0, copy) == {Laminate()}
    assert write_protocols(A, 1, copy) == {Laminate()}
    assert [opt.kind for opt in allowed_options(A, 0, copy, LEVEL_OPTIONS)] == [
        "dense",
        "sparse_hash",
        "sparse_bytemap",
    ]
    sums = fuse_loop_order(
        Query(
            Table(y, (j,)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Table(A, (j, i)), (i,)),
        ),
        (j, i),
    )
    prgm = Plan((copy, sums, Produces((y,))))
    # A dense A would make the sums span the whole matrix.
    kinds = formats_of(prgm, {M: lists(2)}, stats, sf)[A]
    assert all(kind in {"sparse_hash", "sparse_bytemap"} for kind in kinds)


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
        [1, 4, 3.5, (2 + 3.5) * pos_size],
    )


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

    def formats(protocols):
        return formats_of(
            plan(protocols), {a: tensor.ftype}, stats, sf, options=(DenseLevelOption(),)
        )

    with pytest.raises(ValueError, match="No level option supports"):
        formats((Walk(),))
    for protocols in ((), (Follow(),)):
        assert formats(protocols)[b] == ["dense"]


def test_format_selection_uses_custom_model():
    # T = M, then y[i] = sum_j T[i, j], as above.
    i, j = Field("i"), Field("j")
    M, T, y = HardAlias("M"), HardAlias("T"), HardAlias("y")
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {M: sf(fl.asarray(sparse_matrix()), (i, j))}
    copy = fuse_loop_order(Query(Table(T, (i, j)), Table(M, (i, j))), (i, j))
    sums = fuse_loop_order(
        Query(
            Table(y, (i,)),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Table(T, (i, j)), (j,)),
        ),
        (i, j),
    )
    prgm = Plan((copy, sums, Produces((y,))))

    def model(weights):
        return CostModel(
            PROTOCOL_BASIS,
            [weights.get(name, 0.0) for name in PROTOCOL_BASIS.feature_names],
        )

    def choose(weights):
        options = (DenseLevelOption(), SparseListLevelOption())
        formats = formats_of(
            prgm, {M: lists(2)}, stats, sf, cost_model=model(weights), options=options
        )
        return formats[T]

    # Charging only the seeks of lists makes a dense T free, and charging only
    # the seeks of dense levels makes lists free.
    assert choose({"sparse_list.seeks": 1.0}) == ["dense", "dense"]
    assert choose({"dense.seeks": 1.0}) == ["sparse_list", "sparse_list"]


def storage_components(basis, plan, formats, machine) -> list[StorageChoice]:
    """The levels among the components of basis, which price from the blocks
    they list, however far the machine has advanced."""
    levels = []
    for component in basis.components(plan, formats, machine):
        match component:
            case StorageChoice():
                levels.append(component)
    return levels


def test_fused_alias_levels_describe_the_largest_slice():
    t, i = Field("t"), Field("i")
    a, scratch = HardAlias("a"), HardAlias("scratch")
    fused = FusedAlias(scratch, 1)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.eye(4)), (t, i))}
    copy = fuse_loop_order(Query(Table(fused, (t, i)), Table(a, (t, i))), (t, i))
    plan = Plan((copy, Produces((fused,))))
    formats: dict[Alias, list[LevelOption]] = {
        scratch: [SparseHashLevelOption(), ElementLevelOption()]
    }
    basis = ProtocolBasis()
    machine = StatsMachine(sf, dict(stats))
    sparse, element = storage_components(basis, plan, formats, machine)

    # Each slice of the diagonal stores the one entry of its row.
    assert (sparse.alias, sparse.depth, sparse.num_pos) == (scratch, 0, (1.0,))
    assert element.depth == 1
    assert element.num_pos == pytest.approx((1.0,))
    size = np.dtype(np.intp).itemsize
    names = basis.feature_names
    columns = [
        names.index(f"sparse_hash.{name}")
        for name in ("positions", "entries", "position_bytes", "index_bytes")
    ]
    features = basis.get_component_features(sparse, machine)
    np.testing.assert_allclose(features[0, columns], [1, 1, 2 * size, size])


def test_storage_basis_prices_the_levels_of_each_block():
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

    formats: dict[Alias, list[LevelOption]] = {
        out: [SparseHashLevelOption(), ElementLevelOption()]
    }
    basis = ProtocolBasis()
    machine = StatsMachine(sf, dict(stats))
    rows = np.concatenate(
        [
            basis.get_component_features(level, machine)
            for level in storage_components(basis, plan, formats, machine)
        ]
    )
    # A row for each block of the hash level, then of the element level.
    assert rows.shape == (4, basis.n_features)
    names = basis.feature_names
    size = np.dtype(np.intp).itemsize
    columns = [
        names.index(f"sparse_hash.{name}")
        for name in ("positions", "entries", "position_bytes", "index_bytes")
    ]
    np.testing.assert_array_equal(
        rows[:2][:, columns], [[1, n, 2 * size, n * size] for n in nnz]
    )
    columns = [names.index(f"element.{name}") for name in ("values", "value_bytes")]
    np.testing.assert_array_equal(rows[2:][:, columns], [[n, 8 * n] for n in nnz])
