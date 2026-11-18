from itertools import product

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import DynamicFillFType, StaticFillFType, TensorFType
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    STORAGE_BASIS,
    CostModel,
    DenseLevelOption,
    ElementLevelOption,
    LevelOption,
    SparseByteMapLevelOption,
    SparseHashLevelOption,
    SparseListLevelOption,
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
from finch.autoschedule.protocolizer.protocolize import alias_costs, choose_protocols
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DCStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.tensor_stats.uniform_stats import UniformStats
from finch.autoschedule.util import fuse_loop_order
from finch.autoschedule.workspacer import add_workspace
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


def charging(weights):
    """A model which charges a feature of a kind of level, whatever the
    protocol, as weights say by (kind, feature), and nothing else."""
    return CostModel(
        PROTOCOL_BASIS,
        [
            weights.get((name.partition(".")[0], name.rpartition(".")[2]), 0.0)
            for name in PROTOCOL_BASIS.feature_names
        ],
    )


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
    # Every slot of B is written, which a dense level has ready, while the
    # sparse levels insert each.
    inserts = {
        (kind, "distinct"): 1.0
        for kind in ("sparse_list", "sparse_hash", "sparse_bytemap")
    }
    CostFormatter(capture, cost_model=charging(inserts)).lower(
        prgm, {A: tensor.ftype}, stats, sf
    )
    fmt = capture.last_bindings[B]
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
    kinds = formats_of(
        prgm,
        {M: lists(2)},
        stats,
        sf,
        cost_model=charging({("dense", "iterations"): 1.0}),
    )[A]
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

    def choose(weights):
        options = (DenseLevelOption(), SparseListLevelOption())
        formats = formats_of(
            prgm,
            {M: lists(2)},
            stats,
            sf,
            cost_model=charging(weights),
            options=options,
        )
        return formats[T]

    # Charging only the executions of accesses of lists makes a dense T free,
    # and charging only those of dense levels makes lists free.
    assert choose({("sparse_list", "executions"): 1.0}) == ["dense", "dense"]
    assert choose({("dense", "executions"): 1.0}) == ["sparse_list", "sparse_list"]


def test_storage_is_priced_at_the_writes_in_the_largest_fused_slice():
    # Each slice of the diagonal is written into scratch at the loop over i.
    t, i = Field("t"), Field("i")
    a, scratch = HardAlias("a"), HardAlias("scratch")
    fused = FusedAlias(scratch, 1)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.eye(4)), (t, i))}
    copy = fuse_loop_order(Query(Table(fused, (t, i)), Table(a, (t, i))), (t, i))
    plan = Plan((copy, Produces((fused,))))
    lst = SparseListLevelOption()
    formats: dict[Alias, list[LevelOption]] = {
        a: [lst, lst, ElementLevelOption()],
        scratch: [SparseHashLevelOption(), ElementLevelOption()],
    }
    basis = STORAGE_BASIS
    machine = StatsMachine(sf, dict(stats))
    rows = {}
    for access in basis.components(plan, formats, machine):
        row = basis.get_component_features(access, machine)[0]
        rows[access.alias.unfused, access.depth] = row
    assert set(rows) == {(a, 0), (a, 1), (scratch, 0), (scratch, 1)}
    # a is already stored, so reading it costs no storage.
    assert not rows[a, 0].any() and not rows[a, 1].any()
    # scratch stores one slice at a time, the largest of which holds the one
    # entry of its row, in one fiber of the hash level.
    size = np.dtype(np.intp).itemsize
    names = basis.feature_names
    columns = [
        names.index(f"sparse_hash.{name}")
        for name in ("positions", "entries", "position_bytes", "index_bytes")
    ]
    np.testing.assert_allclose(rows[scratch, 0][columns], [1, 1, 2 * size, size])
    columns = [names.index(f"element.{name}") for name in ("values", "value_bytes")]
    np.testing.assert_allclose(rows[scratch, 1][columns], [1, 8])


def test_storage_basis_prices_what_each_query_writes():
    i = Field("i")
    a, b, c, out = (HardAlias(name) for name in ("a", "b", "c", "out"))
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(np.array([1.0, 0.0, 0.0, 0.0])), (i,)),
        b: sf(fl.asarray(np.array([0.0, 1.0, 0.0, 0.0])), (i,)),
        c: sf(fl.asarray(np.array([0.0, 0.0, 1.0, 1.0])), (i,)),
    }
    # The first query defines out, the second updates it, and the third
    # redefines it, each storing what out holds after it.
    define = Query(Table(out, (i,)), Table(a, (i,)))
    update = QueryInto(Table(out, (i,)), Literal(ffuncs.add), Table(b, (i,)))
    redefine = Query(Table(out, (i,)), Table(c, (i,)))
    queries = (define, update, redefine)
    plan = Plan(
        (*(fuse_loop_order(query, (i,)) for query in queries), Produces((out,)))
    )
    machine = StatsMachine(sf, dict(stats))
    nnz = []
    for query in queries:
        machine(query)
        nnz.append(machine(out).estimate_non_fill_values())
    assert nnz[1] > nnz[0]

    formats: dict[Alias, list[LevelOption]] = {
        out: [SparseHashLevelOption(), ElementLevelOption()]
    }
    basis = STORAGE_BASIS
    machine = StatsMachine(sf, dict(stats))
    rows: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for access in basis.components(plan, formats, machine):
        rows[access.depth].append(basis.get_component_features(access, machine)[0])
    # Each query writes the hash level, which is one fiber, and the element
    # level under it.
    names = basis.feature_names
    size = np.dtype(np.intp).itemsize
    columns = [
        names.index(f"sparse_hash.{name}")
        for name in ("positions", "entries", "position_bytes", "index_bytes")
    ]
    np.testing.assert_allclose(
        np.array(rows[0])[:, columns], [[1, n, 2 * size, n * size] for n in nnz]
    )
    columns = [names.index(f"element.{name}") for name in ("values", "value_bytes")]
    np.testing.assert_allclose(np.array(rows[1])[:, columns], [[n, 8 * n] for n in nnz])


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


def test_alias_costs_match_choosing_protocols_for_each_format():
    # c = a b by rows, through a workspace w for each row, then d = c e, which
    # reads e out of the order it stores its fields, so that the default
    # protocol of a level of e is read at a loop before the level is reached.
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b, c, d, e, w = (HardAlias(name) for name in "abcdew")
    plan, _ = _matmul(c, a, b, (i, j, k))
    staged = add_workspace(plan, c, 1, w)
    assert isinstance(staged, Plan)
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(c, (i, k)), Table(e, (k, i)))),
        (),
    )
    second = fuse_loop_order(Query(Table(d, (i, k)), rhs), (i, k))
    prgm = Plan((*staged.bodies[:-1], second, Produces((d,))))

    rng = np.random.default_rng(0)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        tns: sf(fl.asarray((rng.random((6, 6)) < 0.4) * rng.random((6, 6))), idxs)
        for tns, idxs in ((a, (i, j)), (b, (j, k)), (e, (k, i)))
    }
    fill = StaticFillFType(np.float64(0.0))
    ranks = {c: 2, w: 1, d: 2}

    def formatted(alias, kinds):
        return format_ftype(kinds, fill, (fl.ftype(np.intp),) * ranks[alias])

    allowed = {
        alias: [
            allowed_options(alias, depth, prgm, LEVEL_OPTIONS) for depth in range(n)
        ]
        for alias, n in ranks.items()
    }
    bindings: dict[Alias, TensorFType] = {
        **{tns: lists(2) for tns in (a, b, e)},
        **{
            alias: formatted(alias, [lvl[0] for lvl in lvls])
            for alias, lvls in allowed.items()
        },
    }
    _, costs = choose_protocols(prgm, bindings, stats, sf)
    for alias, lvls in allowed.items():
        candidates = {kinds: formatted(alias, kinds) for kinds in product(*lvls)}
        priced = alias_costs(
            prgm, bindings, stats, sf, PROTOCOL_COST_MODEL, alias, candidates, costs
        )
        for kinds, fmt in candidates.items():
            _, expected = choose_protocols(prgm, {**bindings, alias: fmt}, stats, sf)
            assert priced[kinds] == expected, (alias, [opt.kind for opt in kinds])
