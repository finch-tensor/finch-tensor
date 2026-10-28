import itertools

import pytest

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import nnls

import finch as fl
from finch import ffuncs
from finch.algebra import TensorFType
from finch.autoschedule.cost_models import (
    FLOPS_COST_MODEL,
    LOOP_COST_MODEL,
    ByKind,
    Concat,
    FlopsBasis,
    KNearest,
    LinearCostModel,
    PigeonBasis,
    Prefix,
    Quadratic,
    RowMap,
    Statement,
    Stratify,
    TransposeBasis,
)
from finch.autoschedule.tensor_stats import DCStatsFactory, UniformStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Reorder,
    Table,
    TensorStats,
)

PIGEON = LinearCostModel(PigeonBasis(), [1.0, 1.0, 10.0])
TRANSPOSE = LinearCostModel(TransposeBasis(), [0.5, 60.0])


def test_linear_model_sums_row_predictions():
    model = LinearCostModel(TransposeBasis(), [2.0, 3.0])
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert model.predict(rows) == 26.0
    assert model.predict(rows) == sum(model.predict(row[None, :]) for row in rows)
    assert model.predict(np.empty((0, 2))) == 0.0


def random_programs(rng, n_features, n_programs=40):
    return [rng.random((rng.integers(0, 4), n_features)) for _ in range(n_programs)]


@pytest.mark.parametrize(
    "basis",
    [
        PigeonBasis(),
        Quadratic(PigeonBasis()),
        Concat(PigeonBasis(), TransposeBasis()),
        Concat(PigeonBasis(), Quadratic(PigeonBasis())),
        KNearest(PigeonBasis(), np.random.default_rng(0).random((6, 3)), 2),
        Concat(
            PigeonBasis(),
            KNearest(PigeonBasis(), np.random.default_rng(1).random((4, 3))),
        ),
        Stratify(PigeonBasis(), 0, [0.3, 0.7]),
        Stratify(Quadratic(TransposeBasis()), 0, [0.5]),
    ],
)
def test_linear_model_fits_program_runtimes(rng, basis):
    # Only the runtimes of whole programs are known, not those of their rows.
    true = LinearCostModel(basis, rng.random(basis.n_features))
    features = random_programs(rng, basis.n_features)
    runtimes = [true.predict(rows) for rows in features]
    untrained = LinearCostModel(basis, np.zeros(basis.n_features))
    fitted = untrained.fit(features, runtimes)
    assert fitted.basis is basis
    np.testing.assert_allclose(untrained.coefficients, 0)
    for rows in random_programs(rng, basis.n_features, 5):
        assert fitted.predict(rows) == pytest.approx(true.predict(rows))


def test_linear_model_checks_coefficients():
    with pytest.raises(ValueError, match="Expected 3 coefficients"):
        LinearCostModel(PigeonBasis(), [1.0, 2.0])


def test_stratify_places_rows_in_regime_blocks():
    basis = Stratify(TransposeBasis(), 1, [1.0, 3.0])
    assert basis.n_regimes == 3
    assert basis.n_features == 6
    np.testing.assert_array_equal(basis.ridge, np.ones(6))
    rows = np.array([[5.0, 0.5], [6.0, 1.0], [7.0, 2.0], [8.0, 3.5]])
    np.testing.assert_array_equal(
        basis.transform(rows),
        [
            [5.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 3.5],
        ],
    )
    assert basis.transform(np.empty((0, 2))).shape == (0, 6)
    with pytest.raises(ValueError, match="Expected 0 <= feature < 2"):
        Stratify(TransposeBasis(), 2, [1.0])
    with pytest.raises(ValueError, match="increasing thresholds"):
        Stratify(TransposeBasis(), 0, [3.0, 1.0])


def test_stratify_fits_separate_coefficients_per_regime(rng):
    # The per-unit costs differ once the first feature reaches the threshold,
    # and only program runtimes are observed.
    basis = Stratify(TransposeBasis(), 0, [0.5])
    coefficients = np.array([1.0, 2.0, 10.0, 0.5])
    programs = random_programs(rng, 2)
    features = [basis.transform(rows) for rows in programs]
    runtimes = [rows.sum(axis=0) @ coefficients for rows in features]
    fitted = LinearCostModel(basis, np.zeros(4)).fit(features, runtimes)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-9)


def test_knearest_basis_averages_nearest_landmark_costs():
    # A linear model of KNearest is k nearest neighbors over the landmarks.
    basis = KNearest(TransposeBasis(), [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], 2)
    rows = np.array([[0.2, 0.0], [5.0, 4.0]])
    np.testing.assert_allclose(
        basis.transform(rows), [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
    )
    model = LinearCostModel(basis, [1.0, 3.0, 100.0])
    assert model.predict(basis.transform(rows)) == 53.5
    assert basis.transform(np.empty((0, 2))).shape == (0, 3)
    with pytest.raises(ValueError, match="Expected 1 <= k <= 3"):
        KNearest(TransposeBasis(), np.zeros((3, 2)), 4)


def test_knearest_fit_apportions_program_runtimes(rng):
    # Each program's runtime is the sum of the costs of the landmarks nearest
    # its rows, and least squares recovers those costs from the sums alone.
    landmarks = rng.random((5, 2)) * 10
    costs = rng.random(5)
    basis = KNearest(TransposeBasis(), landmarks)
    features, runtimes = [], []
    for _ in range(40):
        nearest = rng.integers(0, 5, size=rng.integers(0, 4))
        rows = landmarks[nearest] + rng.normal(scale=0.1, size=(len(nearest), 2))
        features.append(basis.transform(rows))
        runtimes.append(costs[nearest].sum())
    fitted = LinearCostModel(basis, np.zeros(5)).fit(features, runtimes)
    np.testing.assert_allclose(fitted.coefficients, costs)


class Unpenalized(RowMap):
    @property
    def n_features(self):
        return self.basis.n_features

    @property
    def ridge(self):
        return np.zeros(self.n_features)

    def transform(self, rows):
        return rows


def test_fit_is_nonnegative_least_squares_with_basis_ridge(rng):
    # Six landmark costs from three runtimes is underdetermined, and the
    # penalty is alpha times the ridge weights of each basis.
    raw = Unpenalized(TransposeBasis())
    nearest = KNearest(TransposeBasis(), rng.random((6, 2)))
    basis = Concat(raw, nearest)
    np.testing.assert_array_equal(basis.ridge, [0, 0, 1, 1, 1, 1, 1, 1])
    features = []
    for _ in range(3):
        rows = rng.random((3, 2))
        features.append(
            np.concatenate([raw.transform(rows), nearest.transform(rows)], axis=1)
        )
    runtimes = np.array([1.0, 2.0, 3.0])
    design = np.stack([rows.sum(axis=0) for rows in features])
    for alpha in (0.0, 0.5, 2.0):
        fitted = LinearCostModel(basis, np.zeros(8)).fit(features, runtimes, alpha)
        augmented = np.concatenate([design, np.sqrt(alpha) * np.diag(basis.ridge)])
        expected = nnls(augmented, np.concatenate([runtimes, np.zeros(8)]))[0]
        np.testing.assert_allclose(fitted.coefficients, expected, atol=1e-9)
        assert fitted.coefficients.min() >= 0


def test_fit_clamps_negative_coefficients(rng):
    basis = Concat(PigeonBasis(), TransposeBasis())
    features = random_programs(rng, 5)
    design = np.stack([rows.sum(axis=0) for rows in features])
    runtimes = design @ [1.0, -2.0, 3.0, -1.0, 0.5] + rng.normal(scale=0.1, size=40)
    fitted = LinearCostModel(basis, np.zeros(5)).fit(features, runtimes)
    np.testing.assert_allclose(fitted.coefficients, nnls(design, runtimes)[0])
    assert fitted.coefficients[1] == 0 and fitted.coefficients[3] == 0
    assert np.any(np.linalg.lstsq(design, runtimes)[0] < 0)


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_flops_cost_retains_query_components(factory):
    sf = factory()
    i, j = Field("i"), Field("j")
    a, b, out = HardAlias("a"), HardAlias("b"), HardAlias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i, j))}
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
    cost = FLOPS_COST_MODEL
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
    nonlinear = LinearCostModel(Quadratic(FlopsBasis()), [1.0, 0.0, 0.0, 10.0])
    assert nonlinear.predict_cost(plan, {}, stats, sf) == 120.0
    assert set(stats) == {a}


@pytest.mark.parametrize(
    "cost",
    [
        FLOPS_COST_MODEL,
        PIGEON,
        TRANSPOSE,
        LOOP_COST_MODEL,
        LinearCostModel(
            Quadratic(Concat(PigeonBasis(), TransposeBasis())), np.ones(25)
        ),
        LinearCostModel(Quadratic(PigeonBasis()), np.ones(9)),
        LinearCostModel(KNearest(PigeonBasis(), np.ones((2, 3))), np.ones(2)),
        LinearCostModel(Stratify(PigeonBasis(), 0, [4.0]), np.ones(6)),
    ],
)
@pytest.mark.parametrize("plan", [Plan(()), Plan((Produces(()),))])
def test_empty_cost_components(cost, plan):
    sf = DCStatsFactory()
    assert cost.get_features(plan, {}, {}, sf).shape == (0, cost.basis.n_features)
    assert cost.predict_cost(plan, {}, {}, sf) == 0.0


def test_quadratic_of_concat_crosses_bases():
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = HardAlias("a"), HardAlias("out")
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    plan = fuse_loop_order(Query(Table(out, (j, i)), Table(a, (i, j))), (j, i))
    basis = Concat(PigeonBasis(), TransposeBasis())
    cost = LinearCostModel(Quadratic(basis), np.ones(25))
    features = basis.get_features(plan, {}, stats, sf)
    assert features.shape == (3, 5)
    assert cost.get_features(plan, {}, stats, sf).shape == (3, 25)
    expected = sum(row.sum() ** 2 for row in features)
    assert cost.predict_cost(plan, {}, stats, sf) == expected
    assert expected != features.sum() ** 2


@pytest.mark.parametrize(
    "cost",
    [
        LinearCostModel(Quadratic(PigeonBasis()), np.ones(9)),
        LinearCostModel(Quadratic(TransposeBasis()), np.ones(4)),
        LinearCostModel(
            Concat(Quadratic(PigeonBasis()), Quadratic(TransposeBasis())), np.ones(13)
        ),
        LinearCostModel(
            Quadratic(Concat(PigeonBasis(), TransposeBasis())), np.ones(25)
        ),
        LinearCostModel(Concat(PigeonBasis(), Quadratic(PigeonBasis())), np.ones(12)),
        LinearCostModel(
            Concat(
                KNearest(PigeonBasis(), [[0, 0, 0], [6, 2, 2], [6, 6, 0]], 2),
                Quadratic(TransposeBasis()),
            ),
            np.arange(7),
        ),
        LinearCostModel(Stratify(PigeonBasis(), 0, [2.0, 4.0]), np.arange(9)),
    ],
)
def test_nonlinear_loop_costs_are_additive_over_prefixes(cost):
    sf = DCStatsFactory()
    i, j = Field("i"), Field("j")
    a, out = HardAlias("a"), HardAlias("out")
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Table(a, (i, j))
    machine = StatsMachine(sf, dict(stats))
    for order in ((i, j), (j, i)):
        query = fuse_loop_order(Query(Table(out, order), expr), order)
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
    model = PIGEON
    idxs = tuple(Field(f"i{n}") for n in range(len(shape)))
    a, out = HardAlias("a"), HardAlias("out")
    tensor = fl.asarray(np.ones(shape))
    stats: dict[Alias, TensorStats] = {a: sf(tensor, idxs)}
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype}
    table = Table(a, idxs)
    if kind == "copy":
        order = tuple(reversed(idxs))
        query = Query(Table(out, order), table)
    else:
        order = idxs
        out_idxs = idxs[:-1]
        arg = MapJoin(Literal(ffuncs.mul), (table, table))
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
    plan = Plan((fuse_loop_order(query, order), Produces((out,))))
    original_stats = dict(stats)
    original_fills = {alias: stat.fill_type for alias, stat in stats.items()}

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
    assert all(stat.fill_type is original_fills[alias] for alias, stat in stats.items())


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_pigeon_cost_model_dependent_queries(factory):
    sf = factory()
    model = PIGEON
    i, j = Field("i"), Field("j")
    a, b, out = HardAlias("a"), HardAlias("b"), HardAlias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i, j))}
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype}
    queries = (
        fuse_loop_order(Query(Table(b, (j, i)), Table(a, (i, j))), (j, i)),
        QueryInto(Table(b, (j, i)), Literal(ffuncs.overwrite), Literal(0.0)),
        fuse_loop_order(
            QueryInto(Table(b, (j, i)), Literal(ffuncs.add), Table(a, (i, j))),
            (j, i),
        ),
        fuse_loop_order(
            Query(
                Table(out, (j,)),
                Aggregate(Literal(ffuncs.add), Literal(0.0), Table(b, (j, i)), (i,)),
            ),
            (j, i),
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
        sum(model.predict(rows) for rows in expected)
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
    model = PIGEON
    i, j, k = (Field(name) for name in "ijk")
    a, b, out = (HardAlias(name) for name in ("a", "b", "out"))
    stats: dict[Alias, TensorStats] = {
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
    original_fills = [stat.fill_type for stat in spaces]
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
        query = fuse_loop_order(Query(Table(out, out_idxs), expr), order)

        np.testing.assert_allclose(features.sum(axis=0), original_features)
        np.testing.assert_allclose(model.get_features(query, {}, stats, sf), features)
        assert model.predict_cost(query, {}, stats, sf) == pytest.approx(cost)

    assert all(
        stat.fill_type is fill
        for stat, fill in zip(spaces, original_fills, strict=True)
    )
    assert [stat.index_order for stat in spaces] == original_idxs


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_pigeon_empty_prefix_and_loop_without_inputs(factory):
    sf = factory()
    model = PIGEON
    i = Field("i")
    a = HardAlias("a")
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones(())), ())}
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
    model = TRANSPOSE
    i, j = Field("i"), Field("j")
    a, out = HardAlias("a"), HardAlias("out")
    tensor = fl.asarray(np.eye(16) if sparse else np.ones((16, 16)))
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i, j))}
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype}
    order = (j, i) if transpose else (i, j)
    table = Table(a, (i, j))
    if inplace:
        output = fl.asarray(np.zeros((16, 16)))
        stats[out] = sf(output, order)
        bindings[out] = output.ftype
        query = QueryInto(Table(out, order), Literal(ffuncs.add), table)
    else:
        query = Query(Table(out, order), table)
    plan = Plan((fuse_loop_order(query, order), Produces((out,))))
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
    model = TRANSPOSE
    i, j, k = (Field(name) for name in "ijk")
    a, b, out = (HardAlias(name) for name in ("a", "b", "out"))
    stats: dict[Alias, TensorStats] = {
        a: sf(fl.asarray(np.ones((2, 3))), (i, j)),
        b: sf(fl.asarray(np.ones((3, 4))), (j, k)),
    }
    a_table, b_table = Table(a, (i, j)), Table(b, (j, k))
    arg = MapJoin(Literal(ffuncs.mul), (a_table, a_table, b_table))
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (i, j, k))
    machine = StatsMachine(sf, dict(stats))
    original_fills = [stat.fill_type for stat in stats.values()]

    for order in itertools.permutations((i, j, k)):
        expected = (3 if order.index(j) < order.index(i) else 0) + (
            6 if order.index(k) < order.index(j) else 0
        )
        cost = sum(
            model.predict_component_cost(Prefix(order[:n], expr), machine)
            for n in range(len(order) + 1)
        )
        query = fuse_loop_order(Query(Table(out, ()), expr), order)

        assert cost == expected
        assert model.predict_cost(query, {}, stats, sf) == expected

    assert all(
        stat.fill_type is fill
        for stat, fill in zip(stats.values(), original_fills, strict=True)
    )


@pytest.mark.parametrize("factory", [DCStatsFactory, UniformStatsFactory])
def test_transpose_cost_model_dependent_queries(factory):
    sf = factory()
    model = TRANSPOSE
    i, j = Field("i"), Field("j")
    a, b, c, reduced = (HardAlias(name) for name in ("a", "b", "c", "reduced"))
    tensor = fl.asarray(np.ones((2, 3)))
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i, j))}
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype}
    plan = Plan(
        (
            fuse_loop_order(Query(Table(b, (j, i)), Table(a, (i, j))), (j, i)),
            fuse_loop_order(
                Query(
                    Table(reduced, (j,)),
                    Aggregate(
                        Literal(ffuncs.add), Literal(0.0), Table(b, (j, i)), (i,)
                    ),
                ),
                (j, i),
            ),
            fuse_loop_order(Query(Table(c, (i, j)), Table(b, (j, i))), (i, j)),
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
def test_linear_model_of_concat_sums_models(factory):
    sf = factory()
    i, j = Field("i"), Field("j")
    a, out = HardAlias("a"), HardAlias("out")
    tensor = fl.asarray(np.ones((2, 3)))
    stats: dict[Alias, TensorStats] = {a: sf(tensor, (i, j))}
    bindings: dict[Alias, TensorFType] = {a: tensor.ftype}
    copy = fuse_loop_order(Query(Table(out, (j, i)), Table(a, (i, j))), (j, i))
    plan = Plan((copy, Produces((out,))))
    models = (PIGEON, TRANSPOSE)
    model = LinearCostModel(
        Concat(*(sub.basis for sub in models)),
        np.concatenate([sub.coefficients for sub in models]),
    )
    model.validate_inputs(plan, bindings, stats, sf)

    assert model.basis.n_features == 5
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
    a, out = HardAlias("a"), HardAlias("out")
    stats: dict[Alias, TensorStats] = {a: sf(fl.asarray(np.ones((2, 3))), (i, j))}
    expr = Table(a, (i, j))
    query = fuse_loop_order(Query(Table(out, (j, i)), expr), (j, i))
    models = {0: PIGEON, 1: TRANSPOSE, 2: PIGEON}
    cost = LinearCostModel(
        ByKind(
            {n: model.basis for n, model in models.items()},
            key=lambda component: len(component.prefix),
        ),
        np.concatenate([model.coefficients for model in models.values()]),
    )
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
