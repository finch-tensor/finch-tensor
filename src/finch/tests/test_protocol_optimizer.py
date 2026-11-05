from dataclasses import replace

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
    LinearCostModel,
    Prefix,
    ProtocolBasis,
    ProtocolChoice,
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
    assert names[:2] == ("loop.executions", "loop.iterations")
    block = [name.removeprefix("sparse_list.") for name in names[12:22]]
    assert block == [
        "seeks",
        "parents",
        "follow_visits",
        "walk_visits",
        "gallop_visits",
        "dense_visits",
        "distinct_reads",
        "extrude_visits",
        "laminate_visits",
        "distinct_writes",
    ]

    def lists(protocols):
        plan = product_plan(x, y, c, i, [(proto,) for proto in protocols])
        machine = StatsMachine(sf, dict(stats))
        rows = [
            basis.get_component_features(component, machine)[0]
            for component in basis.components(plan, formats, machine)
        ]
        # The loop over i is entered once, and reads both lists, each an
        # access which only fills the columns of its kind. c has no format,
        # so it isn't written.
        assert len(rows) == 3
        assert rows[0][0] == 1 and not rows[0][2:].any()
        total = sum(rows[1:])
        assert not total[:12].any() and not total[22:].any()
        return total[12:19].tolist()

    # Each list seeks one fiber. Walks step through the 5 entries of each,
    # and either would visit the 5 of the other if it were dense. Each has 5
    # distinct entries, whatever reads them.
    assert lists((Walk(), Walk())) == [2, 2, 0, 10, 0, 10, 10]
    # Gallops jump to the entries they share, within the 5 of the smaller.
    assert lists((Gallop(), Gallop())) == [2, 2, 0, 0, 10, 10, 10]
    # A followed y is looked up at each of the 5 entries of x, and a dense x
    # would visit all 10.
    assert lists((Walk(), Follow())) == [2, 2, 5, 5, 0, 15, 10]
    # A gallop of y alone jumps to each of the 5 entries x walks, and a dense
    # x would visit the 5 entries y gallops to.
    assert lists((Walk(), Gallop())) == [2, 2, 0, 5, 5, 10, 10]
    plan = product_plan(x, y, c, i, ((Walk(),), (Follow(),)))
    rows = Affine(PROTOCOL_BASIS).get_features(plan, formats, stats, sf)
    assert rows.shape == (3, 43)
    assert PROTOCOL_COST_MODEL.predict(rows[:, :-1]) > 0


def test_protocol_basis_counts_writes_apart_from_their_locations():
    # c[i] = sum_j x[j, i], with 5 entries of x in the rows 0, 2, and 5 of c.
    sf = ExactStatsFactory()
    i, j = Field("i"), Field("j")
    x, c = HardAlias("x"), HardAlias("c")
    xd = np.zeros((8, 8))
    xd[[0, 1, 1, 3, 3], [0, 0, 2, 2, 5]] = 1.0
    stats: dict[Alias, TensorStats] = {x: sf(fl.asarray(xd), (j, i))}
    formats: dict[Alias, list[LevelOption]] = {
        x: [SparseListLevelOption(), SparseListLevelOption()],
        c: [SparseHashLevelOption()],
    }
    query = Query(
        Table(c, (i,)),
        Aggregate(Literal(ffuncs.add), Literal(0.0), Table(x, (j, i)), (j,)),
    )
    basis = ProtocolBasis()
    names = basis.feature_names

    def writes(loop_order, parents=None):
        plan = Plan((fuse_loop_order(query, loop_order), Produces((c,))))
        machine = StatsMachine(sf, dict(stats))
        accesses = []
        for component in basis.components(plan, formats, machine):
            match component:
                case ProtocolChoice(write=True):
                    accesses.append(component)
        # c has one level, written at one loop.
        (access,) = accesses
        row = basis.get_component_features(replace(access, parents=parents), machine)
        return [
            row[0, names.index(f"sparse_hash.{name}")]
            for name in (
                "seeks",
                "parents",
                "extrude_visits",
                "laminate_visits",
                "distinct_writes",
            )
        ]

    # Looping over j outside, c is reached at the loop over i, which runs in
    # the 3 rows of x, and laminated at each of the 5 entries of x, to 3
    # distinct entries. Looping over i outside, it is extruded once per row.
    # The level is the root of c, so it is one fiber, and the fibers it is
    # entered as are a feature of their own, which changes nothing else.
    assert writes((j, i)) == [3, 1, 0, 5, 3]
    assert writes((i, j)) == [1, 1, 3, 0, 3]
    assert writes((j, i), parents=2.0) == [3, 2, 0, 5, 3]


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


def product_of_matrices(x, y, c, i, j, protocols=((), ())):
    """c[i] = sum_j x[i, j] * y[i, j], under the loop order (i, j)."""
    tables = tuple(
        Table(tns, (i, j), protos)
        for tns, protos in zip((x, y), protocols, strict=True)
    )
    query = Query(
        Table(c, (i,)),
        Aggregate(
            Literal(ffuncs.add),
            Literal(0.0),
            MapJoin(Literal(ffuncs.mul), tables),
            (j,),
        ),
    )
    return Plan((fuse_loop_order(query, (i, j)), Produces((c,))))


def test_optimize_protocols_prices_a_follow_by_its_lookups():
    sf = ExactStatsFactory()
    i, j = Field("i"), Field("j")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    # x has a diagonal entry in rows 0-59, and y in rows 40-99.
    n = 100
    xd, yd = np.zeros((n, n)), np.zeros((n, n))
    xd[np.arange(60), np.arange(60)] = 1.0
    yd[np.arange(40, n), np.arange(40, n)] = 1.0
    stats: dict[Alias, TensorStats] = {
        x: sf(fl.asarray(xd), (i, j)),
        y: sf(fl.asarray(yd), (i, j)),
    }
    lists = fiber_tensor(
        sparse_list(sparse_list(element(0.0), ftype(np.intp)), ftype(np.intp))
    )
    lst = SparseListLevelOption()
    formats: dict[Alias, list[LevelOption]] = {x: [lst, lst], y: [lst, lst]}
    basis = ProtocolBasis()
    names = basis.feature_names

    def inner(protocols):
        """The seeks and steps of the lists at the loop over j."""
        plan = product_of_matrices(x, y, c, i, j, protocols)
        machine = StatsMachine(sf, dict(stats))
        total = sum(
            basis.get_component_features(component, machine)[0]
            for component in basis.components(plan, formats, machine)
            if len(component.prefix.prefix) == 2
        )
        return [
            total[names.index(f"sparse_list.{name}")]
            for name in ("seeks", "walk_visits")
        ]

    # Whether the lists are walked or followed at i, the loop over j runs only
    # in the 20 rows both have, since a lookup which finds no row runs nothing
    # within it, and there seeks a fiber of each and steps through its entry.
    assert inner(((Walk(), Walk()), (Walk(), Walk()))) == [40, 40]
    assert inner(((Follow(), Walk()), (Follow(), Walk()))) == [40, 40]

    def model(lookup):
        weights = {"follow_visits": lookup, "walk_visits": 1.0, "gallop_visits": 4.0}
        return LinearCostModel(
            PROTOCOL_BASIS,
            [
                weights.get(name.split(".")[1], 0.0)
                for name in PROTOCOL_BASIS.feature_names
            ],
        )

    # So the choice at i is priced by its own steps and lookups: walking both
    # steps through 120 rows, and following both looks each up at all 100.
    plan = product_of_matrices(x, y, c, i, j)
    out = optimize_protocols(plan, {x: lists, y: lists}, stats, sf, model(0.0))
    assert [protos[0] for protos in read_protocols(out, x, y)] == [Follow(), Follow()]
    out = optimize_protocols(plan, {x: lists, y: lists}, stats, sf, model(2.0))
    assert [protos[0] for protos in read_protocols(out, x, y)] == [Walk(), Walk()]


def test_a_dense_level_above_is_entered_at_every_slot():
    sf = ExactStatsFactory()
    i, j = Field("i"), Field("j")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    # x has 3 entries in each of rows 0-3, and y 5 in each of rows 2-9, of 10.
    n = 10
    xd, yd = np.zeros((n, n)), np.zeros((n, n))
    xd[:4, :3] = 1.0
    yd[2:, :5] = 1.0
    stats: dict[Alias, TensorStats] = {
        x: sf(fl.asarray(xd), (i, j)),
        y: sf(fl.asarray(yd), (i, j)),
    }
    plan = product_of_matrices(x, y, c, i, j)
    lst = SparseListLevelOption()
    basis = ProtocolBasis()
    names = basis.feature_names

    def inner(above):
        """The seeks, parents, and steps of each list at the loop over j,
        with the first level of x of the option above."""
        formats: dict[Alias, list[LevelOption]] = {x: [above, lst], y: [lst, lst]}
        machine = StatsMachine(sf, dict(stats))
        rows = {}
        for component in basis.components(plan, formats, machine):
            match component:
                case ProtocolChoice(prefix=Prefix(prefix=(_, _))) as access:
                    rows[access.alias] = basis.get_component_features(access, machine)[
                        0
                    ]
        return {
            tns: [
                row[names.index(f"sparse_list.{name}")]
                for name in ("seeks", "parents", "walk_visits")
            ]
            for tns, row in rows.items()
        }

    # A list above x has a row only where x has entries, and a lookup which
    # finds none runs nothing within it, so the loop over j is entered in the
    # 2 rows both have. A dense level above x has a fiber at every slot, so
    # the loop over j is entered in all 8 rows of y, and seeks an empty fiber
    # of x in 6 of them, where the merge steps through nothing of either.
    assert inner(lst) == {x: [2, 2, 6], y: [2, 2, 10]}
    assert inner(DenseLevelOption()) == {x: [8, 8, 6], y: [8, 8, 10]}


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
