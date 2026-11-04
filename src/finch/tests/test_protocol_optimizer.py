import numpy as np

import finch as fl
from finch import ffuncs
from finch import finch_assembly as asm
from finch.algebra import ftype
from finch.autoschedule import (
    CompilerFormLowerer,
    CostProtocolizer,
    LogicCapture,
    NotationGenerator,
    optimize_protocols,
)
from finch.autoschedule.cost_models import (
    PROTOCOL_BASIS,
    PROTOCOL_COST_MODEL,
    Affine,
    DenseLevelOption,
    LevelOption,
    Prefix,
    ProtocolBasis,
    SparseHashLevelOption,
    SparseListLevelOption,
    level_options,
    level_reads,
    read_options,
)
from finch.autoschedule.formatter.smart_formatter import CostFormatter
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.compile import NotationCompiler
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Follow,
    Gallop,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
    TensorStats,
    Walk,
)
from finch.symbolic import PostOrderDFS
from finch.tensor import dense, element, fiber_tensor, sparse_hash, sparse_list

from .test_sparse_update import dense_values, run_copy, tensors
from .utils import with_workspaces

LIST = fiber_tensor(sparse_list(element(0.0), ftype(np.intp)))
HASH = fiber_tensor(sparse_hash(element(0.0), ftype(np.intp)))


def read_protocols(prgm, *aliases):
    """The protocols of the tables of each alias, in program order."""
    return [
        node.protocols
        for node in PostOrderDFS(prgm)
        if isinstance(node, Table) and node.tns in aliases
    ]


def product_plan(x, y, c, i, protocols=((), ())):
    tables = tuple(
        Table(tns, (i,), protos) for tns, protos in zip((x, y), protocols, strict=True)
    )
    expr = Aggregate(
        Literal(ffuncs.add), Literal(0.0), MapJoin(Literal(ffuncs.mul), tables), ()
    )
    return Plan((fuse_loop_order(Query(Table(c, (i,)), expr), (i,)), Produces((c,))))


def test_level_options_name_the_levels_of_a_format():
    csr = fiber_tensor(dense(sparse_list(element(0.0), ftype(np.intp))))
    assert [option.kind for option in level_options(csr) or ()] == [
        "dense",
        "sparse_list",
    ]
    assert [option.kind for option in level_options(HASH) or ()] == ["sparse_hash"]
    assert level_options(fl.asarray(np.ones(3)).ftype) is None


def test_read_options_walk_levels_reached_in_storage_order():
    i, j, t = Field("i"), Field("j"), Field("t")
    lst = SparseListLevelOption()
    everything = [Walk(), Gallop(), Follow()]
    assert read_options(lst, (i, j), 0, 0, (i, j)) == everything
    assert read_options(lst, (i, j), 0, 1, (i, j)) == everything
    # Under (j, i), the level of j is reached inside the loop over i.
    assert read_options(lst, (i, j), 0, 0, (j, i)) == everything
    assert read_options(lst, (i, j), 0, 1, (j, i)) == [Follow()]
    assert read_options(DenseLevelOption(), (i, j), 0, 0, (i, j)) == [Follow()]
    assert read_options(SparseHashLevelOption(), (i, j), 0, 1, (i, j)) == [
        Walk(),
        Follow(),
    ]
    # The fused fields of a table precede the levels its tensor stores.
    assert read_options(lst, (t, i), 1, 0, (t, i)) == everything


def test_level_reads_reach_each_level_at_its_loop():
    i, j = Field("i"), Field("j")
    A, B = HardAlias("A"), HardAlias("B")
    lst = SparseListLevelOption()
    formats: dict[Alias, list[LevelOption]] = {A: [lst, lst], B: [lst]}
    product = MapJoin(
        Literal(ffuncs.mul), (Table(A, (i, j)), Table(B, (j,), (Follow(),)))
    )
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), product, (j,))

    def reads(loops):
        return [
            (read.alias, read.depth, read.field, read.kind)
            for read in level_reads(Prefix(loops, expr, (i,)), formats)
        ]

    # A table without protocols reads each level with its default.
    assert reads(()) == []
    assert reads((i,)) == [(A, 0, i, "sparse_list.walk")]
    assert reads((i, j)) == [
        (A, 1, j, "sparse_list.walk"),
        (B, 0, j, "sparse_list.follow"),
    ]
    # Under (j, i), both levels of A are reached at the loop over i.
    assert reads((j,)) == [(B, 0, j, "sparse_list.follow")]
    assert reads((j, i)) == [
        (A, 0, i, "sparse_list.walk"),
        (A, 1, j, "sparse_list.walk"),
    ]


def test_protocol_basis_counts_the_reads_of_each_loop():
    sf = ExactStatsFactory()
    i = Field("i")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    stats: dict[Alias, TensorStats] = {
        x: sf(fl.asarray(np.array([0, 2, 0, 4, 5, 0, 0, 7, 0, 9.0])), (i,)),
        y: sf(fl.asarray(np.array([1, 0, 0, 3, 6, 0, 8, 7, 0, 0.0])), (i,)),
    }
    formats: dict[Alias, list[LevelOption]] = {
        x: [SparseListLevelOption()],
        y: [SparseListLevelOption()],
    }
    basis = ProtocolBasis()
    names = basis.feature_names
    block = [name.removeprefix("sparse_list.") for name in names[5:10]]
    assert block == [
        "seeks",
        "follow_visits",
        "walk_visits",
        "gallop_visits",
        "dense_visits",
    ]

    def lists(protocols):
        plan = product_plan(x, y, c, i, [(proto,) for proto in protocols])
        machine = StatsMachine(sf, dict(stats))
        rows = [
            basis.get_component_features(prefix, machine)[0]
            for prefix in basis.components(plan, formats, machine)
        ]
        # The empty prefix reads nothing, and the loop over i reads both
        # lists, which only fill the columns of their kind.
        assert len(rows) == 2 and not rows[0].any()
        assert not rows[1][:5].any() and not rows[1][10:].any()
        return rows[1][5:10].tolist()

    # Each list seeks one fiber. Walks step through the 5 entries of each,
    # and either would visit the 5 of the other if it were dense.
    assert lists((Walk(), Walk())) == [2, 0, 10, 0, 10]
    # Gallops jump to the entries they share, within the 5 of the smaller.
    assert lists((Gallop(), Gallop())) == [2, 0, 0, 10, 10]
    # A followed y is looked up at each of the 5 entries of x, and a dense x
    # would visit all 10.
    assert lists((Walk(), Follow())) == [2, 5, 5, 0, 15]
    # A gallop of y alone jumps to each of the 5 entries x walks, and a dense
    # x would visit the 5 entries y gallops to.
    assert lists((Walk(), Gallop())) == [2, 0, 5, 5, 10]
    plan = product_plan(x, y, c, i, ((Walk(),), (Follow(),)))
    rows = Affine(PROTOCOL_BASIS).get_features(plan, formats, stats, sf)
    assert rows.shape == (2, 21)
    assert PROTOCOL_COST_MODEL.predict(rows[:, :-1]) > 0


def test_optimize_protocols_chooses_the_cheaper_read():
    sf = DCStatsFactory()
    i = Field("i")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    plan = product_plan(x, y, c, i)
    sparse = fl.asarray(np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0.0]))
    full = fl.asarray(np.ones(10))
    stats: dict[Alias, TensorStats] = {x: sf(sparse, (i,)), y: sf(full, (i,))}
    # Walking y steps through every entry for the one entry of x, so y is
    # galloped, or followed by a hash, which can't gallop.
    out = optimize_protocols(plan, {x: LIST, y: LIST}, stats, sf)
    assert read_protocols(out, x, y) == [(Walk(),), (Gallop(),)]
    out = optimize_protocols(plan, {x: LIST, y: HASH}, stats, sf)
    assert read_protocols(out, x, y) == [(Walk(),), (Follow(),)]
    # Tensors of the same density are walked together.
    stats = {x: sf(full, (i,)), y: sf(full, (i,))}
    out = optimize_protocols(plan, {x: LIST, y: LIST}, stats, sf)
    assert read_protocols(out, x, y) == [(Walk(),), (Walk(),)]
    # A tensor without level options keeps the protocols it is given.
    out = optimize_protocols(plan, {x: LIST, y: full.ftype}, stats, sf)
    assert read_protocols(out, x, y) == [(Walk(),), (Follow(),)]


def test_optimize_protocols_keeps_the_protocols_a_program_gives():
    sf = DCStatsFactory()
    i = Field("i")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    sparse = fl.asarray(np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0.0]))
    full = fl.asarray(np.ones(10))
    stats: dict[Alias, TensorStats] = {x: sf(sparse, (i,)), y: sf(full, (i,))}
    # Galloping y would cost less, but the program follows x and walks y.
    plan = product_plan(x, y, c, i, ((Follow(),), (Walk(),)))
    out = optimize_protocols(plan, {x: LIST, y: LIST}, stats, sf)
    assert read_protocols(out, x, y) == [(Follow(),), (Walk(),)]
    # Only the read without a protocol is chosen.
    plan = product_plan(x, y, c, i, ((Walk(),), ()))
    out = optimize_protocols(plan, {x: LIST, y: LIST}, stats, sf)
    assert read_protocols(out, x, y) == [(Walk(),), (Gallop(),)]


def test_optimize_protocols_only_follows_levels_read_out_of_order():
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    A, C = HardAlias("A"), HardAlias("C")
    stats: dict[Alias, TensorStats] = {A: sf(fl.asarray(np.eye(4)), (i, j))}
    fmt = fiber_tensor(
        sparse_list(sparse_list(element(0.0), ftype(np.intp)), ftype(np.intp))
    )
    query = Query(
        Table(C, (j,)),
        Aggregate(Literal(ffuncs.add), Literal(0.0), Table(A, (i, j)), (i,)),
    )
    plan = Plan((fuse_loop_order(query, (j, i)), Produces((C,))))
    out = optimize_protocols(plan, {A: fmt}, stats, sf)
    assert read_protocols(out, A) == [(Walk(), Follow())]
    plan = Plan((fuse_loop_order(query, (i, j)), Produces((C,))))
    out = optimize_protocols(plan, {A: fmt}, stats, sf)
    assert read_protocols(out, A) == [(Walk(), Walk())]


def test_cost_protocolizer_states_the_protocols_it_chooses():
    loader = asm.AssemblyInterpreter()
    sf = DCStatsFactory()
    i = Field("i")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    data_x = np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bindings = {
        x: run_copy(loader, *tensors("S", data_x)),
        y: run_copy(loader, *tensors("S", data_y)),
    }
    stats = {x: sf(fl.asarray(data_x), (i,)), y: sf(fl.asarray(data_y), (i,))}
    plan = product_plan(x, y, c, i)
    capture = LogicCapture()
    formatter = CostFormatter(CostProtocolizer(CompilerFormLowerer(capture)))
    formatter(plan, {var: tns.ftype for var, tns in bindings.items()}, stats, sf)
    assert read_protocols(capture.last_prgm, x, y) == [(Walk(),), (Gallop(),)]
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    bindings[c] = capture.last_bindings[c].construct((10,))
    (result,) = NotationCompiler(loader)(program).main(
        *with_workspaces(program, *bindings.values())
    )
    np.testing.assert_array_equal(dense_values(result.lvl), data_x * data_y)
