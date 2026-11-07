import itertools
from collections import OrderedDict

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.cost_models import (
    WORKSPACE_LOOP_COST_MODEL,
    Concat,
    CostModel,
    LoopBasis,
    PigeonBasis,
    Prefix,
    Quadratic,
    TransposeBasis,
    WriteBasis,
)
from finch.autoschedule.cost_models.loop_cost_models import LoopSpaces
from finch.autoschedule.loop_orderer import loop_order_bnb, loop_order_greedy
from finch.autoschedule.loop_orderer.loop_order_bnb import (
    BFSLoopOrderer,
    BruteForceLoopOrderer,
    DFSLoopOrderer,
    loop_order_bfs,
    loop_order_brute_force,
    loop_order_dfs,
)
from finch.autoschedule.loop_orderer.loop_order_greedy import (
    GreedyLoopOrderer,
    greedy_loop_order,
)
from finch.autoschedule.stages import LoopOrderedForm
from finch.autoschedule.tensor_stats import DCStats, DCStatsFactory, UniformStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Field,
    Follow,
    Fuse,
    Gallop,
    HardAlias,
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


def test_bfs_and_dfs_are_no_worse_than_greedy():
    stats_factory = DCStatsFactory()
    i, j, k, l_, m = (Field(name) for name in "ijklm")
    a, b, c, d = (HardAlias(name) for name in "ABCD")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(a, (i, j)),
            Table(b, (j, k)),
            Table(c, (k, l_)),
            Table(d, (l_, m)),
        ),
    )
    stats: OrderedDict[Alias, DCStats] = OrderedDict(
        {
            a: stats_factory(fl.asarray(np.ones((2, 2))), (i, j)),
            b: stats_factory(fl.asarray(np.ones((2, 2))), (j, k)),
            c: stats_factory(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: stats_factory(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    greedy = greedy_loop_order(expr, stats_factory, stats)
    bfs = loop_order_bfs(expr, stats_factory, stats)
    dfs = loop_order_dfs(expr, stats_factory, stats)

    model = BFSLoopOrderer().cost_model

    def cost(order):
        query = Query(
            Table(HardAlias("out"), expr.fields()),
            Aggregate(Literal(ffuncs.add), Literal(0.0), expr, ()),
        )
        return model.predict_cost(
            fuse_loop_order(query, order), {}, stats, stats_factory
        )

    greedy_cost, bfs_cost, dfs_cost = map(cost, (greedy, bfs, dfs))

    assert set(bfs) == set(expr.fields())
    assert set(dfs) == set(expr.fields())
    assert bfs_cost <= greedy_cost
    assert dfs_cost <= greedy_cost
    assert bfs_cost == pytest.approx(dfs_cost)


def test_brute_force_is_no_worse_than_heuristics():
    stats_factory = DCStatsFactory()
    i, j, k, l_, m = (Field(name) for name in "ijklm")
    a, b, c, d = (HardAlias(name) for name in "ABCD")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(a, (i, j)),
            Table(b, (j, k)),
            Table(c, (k, l_)),
            Table(d, (l_, m)),
        ),
    )
    stats: OrderedDict[Alias, DCStats] = OrderedDict(
        {
            a: stats_factory(fl.asarray(np.ones((2, 2))), (i, j)),
            b: stats_factory(fl.asarray(np.ones((2, 2))), (j, k)),
            c: stats_factory(fl.asarray(np.ones((2, 2))), (k, l_)),
            d: stats_factory(fl.asarray(np.zeros((2, 2))), (l_, m)),
        }
    )

    greedy = greedy_loop_order(expr, stats_factory, stats)
    bfs = loop_order_bfs(expr, stats_factory, stats)
    dfs = loop_order_dfs(expr, stats_factory, stats)
    brute = loop_order_brute_force(expr, stats_factory, stats)

    model = BruteForceLoopOrderer().cost_model

    def cost(order):
        query = Query(
            Table(HardAlias("out"), expr.fields()),
            Aggregate(Literal(ffuncs.add), Literal(0.0), expr, ()),
        )
        return model.predict_cost(
            fuse_loop_order(query, order), {}, stats, stats_factory
        )

    greedy_cost, bfs_cost, dfs_cost, brute_cost = map(cost, (greedy, bfs, dfs, brute))

    assert set(brute) == set(expr.fields())
    assert brute_cost <= greedy_cost
    assert brute_cost <= bfs_cost
    assert brute_cost <= dfs_cost
    assert brute_cost == pytest.approx(
        min(cost(order) for order in itertools.permutations(tuple(expr.fields())))
    )


@pytest.mark.parametrize(
    "search", [loop_order_bfs, loop_order_dfs, loop_order_brute_force]
)
def test_loop_search_minimizes_nonlinear_component_cost(search):
    sf = DCStatsFactory()
    i, j, k = (Field(name) for name in "ijk")
    a, b, out = (HardAlias(name) for name in ("a", "b", "out"))
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(np.ones((2, 3))), (i, j)),
        b: sf(fl.asarray(np.ones((3, 4))), (j, k)),
    }
    arg = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (j,))
    basis = Concat(PigeonBasis(), TransposeBasis())
    model = CostModel(Quadratic(basis), np.ones(49))

    def cost(order):
        query = fuse_loop_order(Query(Table(out, (i, k)), expr), order)
        features = basis.get_features(query, {}, stats, sf)
        return sum(row.sum() ** 2 for row in features)

    chosen = search(expr, sf, stats, (i, k), cost_model=model)
    expected = min(
        cost(order)
        for order in itertools.permutations((i, j, k))
        if order.index(i) < order.index(k)
    )
    assert cost(chosen) == expected


class PreferFieldBasis(LoopBasis, LoopOrderedForm):
    @property
    def n_features(self) -> int:
        return 1

    def __init__(self, field):
        self.field = field

    def get_component_features(self, component, machine):
        prefix = component.prefix
        if not prefix:
            return np.array([[7.0]])
        return np.array(
            [[10.0 if len(prefix) == 1 and prefix[0] != self.field else 0.0]]
        )


@pytest.mark.parametrize(
    "search, orderer",
    [
        (greedy_loop_order, GreedyLoopOrderer),
        (loop_order_bfs, BFSLoopOrderer),
        (loop_order_dfs, DFSLoopOrderer),
        (loop_order_brute_force, BruteForceLoopOrderer),
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_loop_orderers_use_selected_model(search, orderer, inplace):
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = HardAlias("a"), HardAlias("out")
    expr = Table(a, (i, j))
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones((16, 16))), (i, j))}
    model = CostModel(PreferFieldBasis(j), [1.0])
    assert search(expr, sf, stats, cost_model=model) == (j, i)
    if inplace:
        stats[out] = sf(fl.asarray(np.ones(())), ())
        query = QueryInto(Table(out, ()), Literal(ffuncs.add), expr)
    else:
        query = Query(
            Table(out, ()), Aggregate(Literal(ffuncs.add), Literal(0.0), expr, (i, j))
        )
    plan = Plan((query, Produces((out,))))
    scheduler = orderer(cost_model=model)
    ordered = scheduler.set_loop_orders(plan, stats, sf)
    assert scheduler.cost_model is model
    assert model.predict_cost(ordered, {}, stats, sf) == 7
    match ordered.bodies[0]:
        case Fuse(outer, Fuse(inner, Query() | QueryInto())):
            assert (outer, inner) == (j, i)
        case _:
            pytest.fail("Expected an ordered query")
    assert orderer().cost_model is WORKSPACE_LOOP_COST_MODEL


@pytest.mark.parametrize(
    "search",
    [greedy_loop_order, loop_order_bfs, loop_order_dfs, loop_order_brute_force],
)
def test_loop_orderers_may_visit_outputs_out_of_order(search):
    sf = UniformStatsFactory()
    i, j, unit = Field("i"), Field("j"), Field("unit")
    a = HardAlias("a")
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), Table(a, (i, j)), (j,))
    # Without a cost for writes, the output's fields may come in any order.
    prefer_i = CostModel(PreferFieldBasis(i), [1.0])
    assert search(expr, sf, stats, (unit, i), cost_model=prefer_i) == (i, j, unit)


# A greedy search only sees the writes to a level once it reaches the level,
# after it has chosen the loops before it.
@pytest.mark.parametrize(
    "search", [loop_order_bfs, loop_order_dfs, loop_order_brute_force]
)
def test_loop_orderers_write_outputs_in_order_when_it_pays(search):
    sf = UniformStatsFactory()
    i, j, unit = Field("i"), Field("j"), Field("unit")
    a = HardAlias("a")
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), Table(a, (i, j)), (j,))
    # Scattering into the output costs more than looping over i first saves.
    basis = Concat(PreferFieldBasis(i), WriteBasis())
    writes = CostModel(basis, [1.0, 0.0, 100.0, 100.0])
    assert search(expr, sf, stats, (unit, i), cost_model=writes) == (unit, i, j)


@pytest.mark.parametrize(
    "op, init", [(ffuncs.add, 0.0), (ffuncs.mul, 1.0), (ffuncs.max, -float("inf"))]
)
def test_prefix_cost_uses_reduction_statistics(op, init):
    sf = UniformStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = HardAlias("a"), HardAlias("out")
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), (i, j))
    }
    table = Table(a, (i, j))
    expr = Aggregate(Literal(op), Literal(init), table, (j,))
    model = BFSLoopOrderer().cost_model
    machine = StatsMachine(sf, dict(stats))
    for order in ((i, j), (j, i)):
        query = fuse_loop_order(Query(Table(out, (i,)), expr), order)
        assert sum(
            model.predict_component_cost(Prefix(order[:n], expr, (i,)), machine)
            for n in range(len(order) + 1)
        ) == pytest.approx(model.predict_cost(query, {}, stats, sf))


@pytest.mark.parametrize(
    "orderer,module,search",
    [
        (GreedyLoopOrderer, loop_order_greedy, "greedy_loop_order"),
        (BFSLoopOrderer, loop_order_bnb, "loop_order_bfs"),
        (DFSLoopOrderer, loop_order_bnb, "loop_order_dfs"),
        (BruteForceLoopOrderer, loop_order_bnb, "loop_order_brute_force"),
    ],
)
def test_loop_orderers_refresh_stats_of_redefined_aliases(
    monkeypatch, orderer, module, search
):
    # The second copy is the same node as the first, but scratch has been
    # redefined in between, so its statistics must not come from a cache.
    i = Field("i")
    a, b, scratch, out, result = map(HardAlias, ("a", "b", "scratch", "out", "result"))
    copy = Query(Table(out, (i,)), Table(scratch, (i,)))
    total = Aggregate(Literal(ffuncs.add), Literal(0.0), Table(out, (i,)), (i,))
    plan = Plan(
        (
            Query(Table(scratch, (i,)), Table(a, (i,))),
            copy,
            Query(Table(scratch, (i,)), Table(b, (i,))),
            copy,
            Query(Table(result, ()), total),
            Produces((result,)),
        )
    )
    seen = []

    def choose(expr, stats_factory, stats, output_vars, **kwargs):
        seen.append(stats[out].estimate_non_fill_values())
        return expr.fields()

    monkeypatch.setattr(module, search, choose)
    factory = DCStatsFactory()
    sparse = fl.asarray(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    dense = fl.asarray(np.ones(5))
    orderer().set_loop_orders(
        plan, {a: factory(sparse, (i,)), b: factory(dense, (i,))}, factory
    )
    assert seen == [5]


@pytest.mark.parametrize(
    "search", [loop_order_bfs, loop_order_dfs, loop_order_brute_force, greedy_loop_order]
)
def test_loop_orderers_read_a_csr_out_of_order_through_its_copy(search):
    # X[i, k] = sum_j A[i, j] B[k, j] of CSR tensors. Looping over k within i
    # follows the dense rows of B at every k, for every i: all n^2 of them.
    # Looping over j and then k reads B through a transposed copy, which walks
    # its rows; when ordering loops it is priced as that copy will be read, not
    # with the protocols of the rows of B, which would price the walk of the
    # copy over all n slots of each of its rows, more than n^2 at 3 a row.
    rng = np.random.default_rng(0)
    n = 200
    data = np.zeros((n, n))
    data[np.repeat(np.arange(n), 3), rng.integers(0, n, 3 * n)] = 1.0
    sf = DCStatsFactory()
    i, j, k = (Field(name) for name in "ijk")
    a, b = HardAlias("a"), HardAlias("b")
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(data), (i, j)),
        b: sf(fl.asarray(data), (k, j)),
    }
    csr = (Follow(), Walk())
    expr = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(a, (i, j), csr), Table(b, (k, j), csr))),
        (j,),
    )
    assert search(expr, sf, stats, (i, k), ordering=True) == (i, j, k)


@pytest.mark.parametrize("order", ["ij", "ji"])
def test_ordering_gallops_the_shared_fields_of_reads_not_yet_formatted(order):
    # Y[j] = sum_i X[i, j] v[i] of an intermediate X, which has no protocols
    # until it is formatted: in either loop order it may gallop where it meets
    # v, at i, and walks j, which it reads alone.
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    x, v = HardAlias("x"), HardAlias("v")
    stats: dict[Alias, TensorStats] = {
        x: sf(fl.asarray(np.eye(4)), (i, j)),
        v: sf(fl.asarray(np.ones(4)), (i,)),
    }
    read = Table(x, (i, j))
    expr = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (read, Table(v, (i,), (Walk(),)))),
        (i,),
    )
    loops = tuple(Field(name) for name in order)
    machine = StatsMachine(sf, dict(stats))
    protocols = {
        loops[n - 1]: LoopSpaces(
            Prefix(loops[:n], expr, (j,), ordering=True), machine
        ).protocol(read)
        for n in (1, 2)
    }
    assert protocols == {i: Gallop(), j: DefaultProtocol()}
