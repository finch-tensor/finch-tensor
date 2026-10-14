import itertools

import pytest

import numpy as np
from scipy.linalg import block_diag

import finch as fl
from finch import ffuncs
from finch.autoschedule.cost_models import (
    ByKind,
    Concat,
    FlopsCostModel,
    LinearPredictor,
    PigeonCostModel,
    Prefix,
    QuadraticPredictor,
    Statement,
    Sum,
    TransposeCostModel,
)
from finch.autoschedule.tensor_stats import DCStatsFactory, UniformStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    Table,
)


@pytest.mark.parametrize(
    "predictor, expected",
    [
        (LinearPredictor(np.array([2.0, 3.0])), 26.0),
        (QuadraticPredictor(np.array([1.0, 2.0, 3.0, 4.0])), 160.0),
    ],
)
def test_predictors_sum_component_predictions(predictor, expected):
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert predictor(features) == expected
    assert predictor(features) == sum(predictor(row[None, :]) for row in features)
    assert predictor(np.empty((0, 2))) == 0.0


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_flops_cost_retains_query_components(factory):
    sf = factory()
    i, j = Field("i"), Field("j")
    a, b, out = Alias("a"), Alias("b"), Alias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats = {a: sf(tensor, (i, j))}
    plan = Plan(
        (
            Query(
                Table(b, (i,)),
                Aggregate(Literal(ffuncs.add), Literal(0.0), Table(a, (i, j)), (j,)),
            ),
            Query(Table(out, (i,)), Table(b, (i,))),
            Produces((out,)),
        )
    )
    cost = FlopsCostModel()
    np.testing.assert_array_equal(
        cost.get_features(plan, {}, stats, sf), [[6, 2], [2, 2]]
    )
    assert cost.predict_cost(plan, {}, stats, sf) == 48.0
    machine = StatsMachine(sf, dict(stats))
    query_costs = []
    for stmt in plan.bodies[:2]:
        query_costs.append(cost.predict_component_cost(Statement(stmt), machine))
        machine(stmt)
    assert query_costs == [26.0, 22.0]
    nonlinear = FlopsCostModel(QuadraticPredictor(np.array([1.0, 0.0, 0.0, 10.0])))
    assert nonlinear.predict_cost(plan, {}, stats, sf) == 120.0
    assert set(stats) == {a}


@pytest.mark.parametrize(
    "cost",
    [
        FlopsCostModel(),
        PigeonCostModel(),
        TransposeCostModel(),
        Sum(PigeonCostModel(), TransposeCostModel()),
        Concat(
            PigeonCostModel(),
            TransposeCostModel(),
            predictor=QuadraticPredictor(np.ones(25)),
        ),
        PigeonCostModel(QuadraticPredictor(np.ones(9))),
    ],
)
@pytest.mark.parametrize("plan", [Plan(()), Plan((Produces(()),))])
def test_empty_cost_components(cost, plan):
    sf = DCStatsFactory()
    assert cost.get_features(plan, {}, {}, sf).shape == (0, cost.n_features)
    assert cost.predict_cost(plan, {}, {}, sf) == 0.0


def test_concat_uses_joint_nonlinear_predictor():
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    stats = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    plan = Query(Table(out, (j, i)), Table(a, (i, j)))
    cost = Concat(
        PigeonCostModel(),
        TransposeCostModel(),
        predictor=QuadraticPredictor(np.ones(25)),
    )
    features = cost.get_features(plan, {}, stats, sf)
    assert features.shape == (3, 5)
    expected = sum(row.sum() ** 2 for row in features)
    assert cost.predict_cost(plan, {}, stats, sf) == expected
    assert expected != features.sum() ** 2


@pytest.mark.parametrize(
    "cost",
    [
        PigeonCostModel(QuadraticPredictor(np.ones(9))),
        TransposeCostModel(QuadraticPredictor(np.ones(4))),
        Sum(
            PigeonCostModel(QuadraticPredictor(np.ones(9))),
            TransposeCostModel(QuadraticPredictor(np.ones(4))),
        ),
        Concat(
            PigeonCostModel(),
            TransposeCostModel(),
            predictor=QuadraticPredictor(np.ones(25)),
        ),
    ],
)
def test_nonlinear_loop_costs_are_additive_over_prefixes(cost):
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    stats = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Table(a, (i, j))
    machine = StatsMachine(sf, dict(stats))
    for order in ((i, j), (j, i)):
        query = Query(Table(out, order), expr)
        expected = sum(
            cost.predict_component_cost(Prefix(order[:n], expr, order), machine)
            for n in range(len(order) + 1)
        )
        assert cost.predict_cost(query, {}, stats, sf) == expected


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("kind", ["copy", "aggregate", "update"])
def test_pigeon_cost_model_query_forms(factory, shape, kind):
    sf = factory()
    model = PigeonCostModel()
    idxs = tuple(Field(f"i{n}") for n in range(len(shape)))
    a, out = Alias("a"), Alias("out")
    tensor = fl.asarray(np.ones(shape))
    stats = {a: sf(tensor, idxs)}
    bindings = {a: tensor.ftype}
    table = Table(a, idxs)
    if kind == "copy":
        query = Query(Table(out, tuple(reversed(idxs))), table)
    else:
        out_idxs = idxs[:-1]
        arg = Reorder(MapJoin(Literal(ffuncs.mul), (table, table)), idxs)
        if kind == "aggregate":
            query = Query(
                Table(out, out_idxs),
                Aggregate(Literal(ffuncs.add), Literal(0.0), arg, idxs[-1:]),
            )
        else:
            output = fl.asarray(np.zeros(shape[:-1]))
            stats[out] = sf(output, out_idxs)
            bindings[out] = output.ftype
            query = QueryInto(Table(out, out_idxs), Literal(ffuncs.add), arg)
    plan = Plan((query, Produces((out,))))
    original_stats = dict(stats)
    original_fills = {alias: stat.fill_value for alias, stat in stats.items()}

    model.validate_inputs(plan, bindings, stats, sf)
    features = model.get_features(plan, bindings, stats, sf)

    assert features.shape == (len(idxs) + 1, 3)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)
    assert model.predict_cost(plan, bindings, stats, sf) > 0
    if idxs:
        assert np.all(features.sum(axis=0) > 0)
    else:
        np.testing.assert_array_equal(features[0, :2], [0, 0])
    assert stats == original_stats
    assert all(
        stat.fill_value is original_fills[alias] for alias, stat in stats.items()
    )


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_pigeon_cost_model_dependent_queries(factory):
    sf = factory()
    model = PigeonCostModel()
    i, j = Field("i"), Field("j")
    a, b, out = Alias("a"), Alias("b"), Alias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats = {a: sf(tensor, (i, j))}
    bindings = {a: tensor.ftype}
    queries = (
        Query(Table(b, (j, i)), Table(a, (i, j))),
        QueryInto(Table(b, (j, i)), Literal(ffuncs.overwrite), Literal(0.0)),
        QueryInto(
            Table(b, (j, i)),
            Literal(ffuncs.add),
            Reorder(Table(a, (i, j)), (j, i)),
        ),
        Query(
            Table(out, (j,)),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0.0),
                Reorder(Table(b, (j, i)), (j, i)),
                (i,),
            ),
        ),
    )
    plan = Plan((*queries, Produces((out,))))
    model.validate_inputs(plan, bindings, stats, sf)
    features = model.get_features(plan, bindings, stats, sf)

    intermediate_stats = dict(stats)
    machine = StatsMachine(sf, intermediate_stats)
    expected = []
    for query in queries:
        expected.append(model.get_features(query, bindings, intermediate_stats, sf))
        machine(query)

    np.testing.assert_allclose(features, np.concatenate(expected, axis=0))
    assert model.predict_cost(plan, bindings, stats, sf) == pytest.approx(
        sum(model.predictor(rows) for rows in expected)
    )
    assert set(stats) == {a}


@pytest.mark.parametrize(
    "factory, expected",
    [
        (
            DCStatsFactory,
            [
                [13, 8, 8],
                [13, 13, 8],
                [13, 8, 8],
                [15, 10, 10],
                [15, 15, 10],
                [15, 14, 10],
            ],
        ),
        (
            UniformStatsFactory,
            [
                [25.030714922766236, 8.397118199352539, 6.9057883730956355],
                [20.948107815782194, 10.936459669391933, 6.9057883730956355],
                [25.546706922536774, 8.172993988088525, 6.9057883730956355],
                [25.64527434095588, 9.008949572579267, 8.042275185340479],
                [22.592642244606303, 12.072946481636775, 8.042275185340479],
                [22.275728472420994, 11.506047408332968, 8.042275185340479],
            ],
        ),
    ],
)
def test_pigeon_prefix_costs_match_full_cost(factory, expected):
    sf = factory()
    model = PigeonCostModel()
    i, j, k = (Field(name) for name in "ijk")
    a, b, out = (Alias(name) for name in ("a", "b", "out"))
    stats = {
        a: sf(fl.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), (i, j)),
        b: sf(
            fl.asarray(
                [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            (j, k),
        ),
    }
    arg = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (j,))
    machine = StatsMachine(sf, dict(stats))
    spaces = list(stats.values())
    original_fills = [stat.fill_value for stat in spaces]
    original_idxs = [stat.index_order for stat in spaces]

    # These totals were captured before extracting the prefix calculation.
    for order, original_features in zip(
        itertools.permutations((i, j, k)), expected, strict=True
    ):
        out_idxs = tuple(idx for idx in order if idx != j)
        prefixes = [order[:n] for n in range(len(order) + 1)]
        features = np.concatenate(
            [
                model.get_component_features(Prefix(prefix, expr, out_idxs), machine)
                for prefix in prefixes
            ]
        )
        cost = sum(
            model.predict_component_cost(Prefix(prefix, expr, out_idxs), machine)
            for prefix in prefixes
        )
        query = Query(
            Table(out, out_idxs),
            Aggregate(Literal(ffuncs.add), Literal(0.0), Reorder(arg, order), (j,)),
        )

        np.testing.assert_allclose(features.sum(axis=0), original_features)
        np.testing.assert_allclose(model.get_features(query, {}, stats, sf), features)
        assert model.predict_cost(query, {}, stats, sf) == pytest.approx(cost)

    assert all(
        stat.fill_value is fill
        for stat, fill in zip(spaces, original_fills, strict=True)
    )
    assert [stat.index_order for stat in spaces] == original_idxs


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_pigeon_empty_prefix_and_loop_without_inputs(factory):
    sf = factory()
    model = PigeonCostModel()
    i = Field("i")
    a = Alias("a")
    stats = {a: sf(fl.asarray(np.ones(())), ())}
    scalar = Table(a, ())
    broadcast = Reorder(scalar, (i,))
    machine = StatsMachine(sf, dict(stats))

    np.testing.assert_array_equal(
        model.get_component_features(Prefix((), scalar), machine), [[0, 0, 1]]
    )
    assert model.predict_component_cost(Prefix((), scalar), machine) == 10
    np.testing.assert_array_equal(
        model.get_component_features(Prefix((i,), broadcast), machine), [[1, 0, 1]]
    )


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_transpose_cost_model_queries(factory, sparse, transpose, inplace):
    sf = factory()
    model = TransposeCostModel()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    tensor = fl.asarray(np.eye(16) if sparse else np.ones((16, 16)))
    stats = {a: sf(tensor, (i, j))}
    bindings = {a: tensor.ftype}
    order = (j, i) if transpose else (i, j)
    table = Table(a, (i, j))
    if inplace:
        output = fl.asarray(np.zeros((16, 16)))
        stats[out] = sf(output, order)
        bindings[out] = output.ftype
        query = QueryInto(Table(out, order), Literal(ffuncs.add), Reorder(table, order))
    else:
        query = Query(Table(out, order), table)
    plan = Plan((query, Produces((out,))))
    original_stats = dict(stats)
    expected = (16 * 60 if sparse else 256 * 0.5) if transpose else 0
    expected_features = ([0, 16] if sparse else [256, 0]) if transpose else [0, 0]

    model.validate_inputs(plan, bindings, stats, sf)
    np.testing.assert_array_equal(
        model.get_features(plan, bindings, stats, sf),
        [[0, 0], expected_features, [0, 0]],
    )
    assert model.predict_cost(plan, bindings, stats, sf) == expected
    assert stats == original_stats


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_transpose_prefix_costs_charge_each_input_once(factory):
    sf = factory()
    model = TransposeCostModel()
    i, j, k = (Field(name) for name in "ijk")
    a, b, out = (Alias(name) for name in ("a", "b", "out"))
    stats = {
        a: sf(fl.asarray(np.ones((2, 3))), (i, j)),
        b: sf(fl.asarray(np.ones((3, 4))), (j, k)),
    }
    a_table, b_table = Table(a, (i, j)), Table(b, (j, k))
    arg = MapJoin(Literal(ffuncs.mul), (a_table, a_table, b_table))
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (i, j, k))
    machine = StatsMachine(sf, dict(stats))
    original_fills = [stat.fill_value for stat in stats.values()]

    for order in itertools.permutations((i, j, k)):
        expected = (3 if order.index(j) < order.index(i) else 0) + (
            6 if order.index(k) < order.index(j) else 0
        )
        cost = sum(
            model.predict_component_cost(Prefix(order[:n], expr), machine)
            for n in range(len(order) + 1)
        )
        query = Query(
            Table(out, ()),
            Aggregate(
                Literal(ffuncs.add), Literal(0.0), Reorder(arg, order), (i, j, k)
            ),
        )

        assert cost == expected
        assert model.predict_cost(query, {}, stats, sf) == expected

    assert all(
        stat.fill_value is fill
        for stat, fill in zip(stats.values(), original_fills, strict=True)
    )


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_transpose_cost_model_dependent_queries(factory):
    sf = factory()
    model = TransposeCostModel()
    i, j = Field("i"), Field("j")
    a, b, c, reduced = (Alias(name) for name in ("a", "b", "c", "reduced"))
    tensor = fl.asarray(np.ones((2, 3)))
    stats = {a: sf(tensor, (i, j))}
    bindings = {a: tensor.ftype}
    plan = Plan(
        (
            Query(Table(b, (j, i)), Table(a, (i, j))),
            Query(
                Table(reduced, (j,)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0.0),
                    Reorder(Table(b, (j, i)), (j, i)),
                    (i,),
                ),
            ),
            Query(Table(c, (i, j)), Table(b, (j, i))),
            QueryInto(Table(c, (i, j)), Literal(ffuncs.overwrite), Literal(0.0)),
            Produces((c, reduced)),
        )
    )

    model.validate_inputs(plan, bindings, stats, sf)
    np.testing.assert_array_equal(
        model.get_features(plan, bindings, stats, sf).sum(axis=0), [12, 0]
    )
    assert model.predict_cost(plan, bindings, stats, sf) == 6
    assert set(stats) == {a}


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_sum_cost_model_sums_submodels(factory):
    sf = factory()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats = {a: sf(tensor, (i, j))}
    bindings = {a: tensor.ftype}
    plan = Plan((Query(Table(out, (j, i)), Table(a, (i, j))), Produces((out,))))
    models = (PigeonCostModel(), TransposeCostModel())
    model = Sum(*models)
    model.validate_inputs(plan, bindings, stats, sf)

    assert model.n_features == 5
    np.testing.assert_allclose(
        model.get_features(plan, bindings, stats, sf),
        np.concatenate(
            [sub.get_features(plan, bindings, stats, sf) for sub in models], axis=1
        ),
    )
    assert model.predict_cost(plan, bindings, stats, sf) == pytest.approx(
        sum(sub.predict_cost(plan, bindings, stats, sf) for sub in models)
    )


def test_by_kind_dispatches_components_by_key():
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = Alias("a"), Alias("out")
    stats = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Table(a, (i, j))
    query = Query(Table(out, (j, i)), expr)
    models = {0: PigeonCostModel(), 1: TransposeCostModel(), 2: PigeonCostModel()}
    cost = ByKind(models, key=lambda component: len(component.prefix))
    machine = StatsMachine(sf, dict(stats))
    prefixes = [Prefix((j, i)[:n], expr, (j, i)) for n in range(3)]

    np.testing.assert_array_equal(
        cost.get_features(query, {}, stats, sf),
        block_diag(
            *(
                models[n].get_component_features(prefix, machine)
                for n, prefix in enumerate(prefixes)
            )
        ),
    )
    assert cost.predict_cost(query, {}, stats, sf) == sum(
        models[n].predict_component_cost(prefix, machine)
        for n, prefix in enumerate(prefixes)
    )
