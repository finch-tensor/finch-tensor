import itertools
import json

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
    PROTOCOL_COST_MODEL,
    STORAGE_BASIS,
    Affine,
    ByKind,
    Concat,
    CostModel,
    FlopsBasis,
    PigeonBasis,
    Prefix,
    Quadratic,
    RowMap,
    Statement,
    Stratify,
)
from finch.autoschedule.tensor_stats import DCStatsFactory, UniformStatsFactory
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.finch_logic import (
    Aggregate,
    Alias,
    DefaultProtocol,
    Field,
    Follow,
    Gallop,
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
    Walk,
)

from .utils import TransposeBasis

PIGEON = CostModel(PigeonBasis(), [1.0, 1.0, 10.0, 1.0, 0.0])
TRANSPOSE = CostModel(TransposeBasis(), [0.5, 60.0])


def test_linear_model_sums_row_predictions():
    model = CostModel(TransposeBasis(), [2.0, 3.0])
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert model.predict(rows) == 26.0
    assert model.predict(rows) == sum(model.predict(row[None, :]) for row in rows)
    assert model.predict(np.empty((0, 2))) == 0.0


def random_programs(rng, n_features, n_programs=40):
    return [rng.random((rng.integers(1, 4), n_features)) for _ in range(n_programs)]


def first_class(rows):
    return np.zeros(len(rows), dtype=int)


def above(cutoffs, feature):
    """The class of each row: the number of cutoffs its feature exceeds."""

    def classify(rows):
        return np.searchsorted(cutoffs, rows[:, feature], side="left")

    return classify


@pytest.mark.parametrize(
    "basis",
    [
        PigeonBasis(),
        Quadratic(PigeonBasis()),
        Concat(PigeonBasis(), TransposeBasis()),
        Concat(PigeonBasis(), Quadratic(PigeonBasis())),
        Stratify(PigeonBasis(), 3, first_class),
        Stratify(Quadratic(TransposeBasis()), 2, first_class),
        Affine(PigeonBasis()),
        Affine(Stratify(PigeonBasis(), 2, first_class)),
    ],
)
def test_linear_model_fits_program_runtimes(rng, basis):
    # Only the runtimes of whole programs are known, not those of their rows.
    true = CostModel(basis, rng.random(basis.n_features))
    features = random_programs(rng, basis.n_features)
    runtimes = [true.predict(rows) for rows in features]
    untrained = CostModel(basis, np.zeros(basis.n_features))
    fitted = untrained.fit(features, runtimes)
    assert fitted.basis is basis
    np.testing.assert_allclose(untrained.coefficients, 0)
    for rows in random_programs(rng, basis.n_features, 5):
        assert fitted.predict(rows) == pytest.approx(true.predict(rows))


def test_linear_model_checks_coefficients():
    with pytest.raises(ValueError, match="Expected 5 coefficients"):
        CostModel(PigeonBasis(), [1.0, 2.0])


def test_stratify_places_rows_in_the_block_of_their_class():
    basis = Stratify(TransposeBasis(), 3, above([1.0, 3.0], 1))
    assert basis.n_features == 6
    rows = np.array([[5.0, 0.5], [6.0, 1.0], [7.0, 2.0], [8.0, 3.5]])
    np.testing.assert_array_equal(
        basis.transform(rows),
        [
            [5.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 3.5],
        ],
    )
    assert basis.transform(np.empty((0, 2))).shape == (0, 6)
    with pytest.raises(ValueError, match="at least one class"):
        Stratify(TransposeBasis(), 0, first_class)
    with pytest.raises(ValueError, match="classes below 2"):
        Stratify(TransposeBasis(), 2, lambda rows: [2] * len(rows)).transform(rows)


def test_stratify_fits_separate_coefficients_per_class(rng):
    # The per-unit costs differ once the first feature exceeds the cutoff,
    # and only program runtimes are observed.
    basis = Stratify(TransposeBasis(), 2, above([0.5], 0))
    coefficients = np.array([1.0, 2.0, 10.0, 0.5])
    programs = random_programs(rng, 2)
    features = [basis.transform(rows) for rows in programs]
    runtimes = [rows.sum(axis=0) @ coefficients for rows in features]
    fitted = CostModel(basis, np.zeros(4)).fit(features, runtimes)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-9)


@pytest.mark.parametrize(
    "model",
    [
        FLOPS_COST_MODEL,
        LOOP_COST_MODEL,
        CostModel(Affine(PigeonBasis()), np.arange(6)),
        PROTOCOL_COST_MODEL,
        CostModel(Affine(STORAGE_BASIS), np.arange(17)),
        CostModel(
            Concat(
                Stratify(Quadratic(PigeonBasis()), 3, first_class), TransposeBasis()
            ),
            np.arange(77),
        ),
    ],
)
def test_models_round_trip_through_json(rng, model):
    data = json.loads(json.dumps(model.to_json()))
    assert data["feature_names"] == list(model.basis.feature_names)
    assert len(data["feature_names"]) == model.basis.n_features
    restored = model.from_json(data)
    assert restored.basis is model.basis
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    for rows in random_programs(rng, model.basis.n_features, 3):
        assert restored.predict(rows) == model.predict(rows)


def test_from_json_rejects_other_bases():
    with pytest.raises(ValueError, match="Expected the coefficients of"):
        LOOP_COST_MODEL.from_json(PIGEON.to_json())


def test_feature_names_compose():
    assert Affine(PigeonBasis()).feature_names == (
        "iterations",
        "operations",
        "outputs",
        "lookups",
        "dense_iterations",
        "constant",
    )
    assert Quadratic(TransposeBasis()).feature_names == (
        "dense_transposes*dense_transposes",
        "dense_transposes*sparse_transposes",
        "sparse_transposes*dense_transposes",
        "sparse_transposes*sparse_transposes",
    )
    assert Stratify(TransposeBasis(), 2, first_class).feature_names == (
        "class0.dense_transposes",
        "class0.sparse_transposes",
        "class1.dense_transposes",
        "class1.sparse_transposes",
    )
    assert ByKind({"a": TransposeBasis(), "b": PigeonBasis()}).feature_names[:3] == (
        "a.dense_transposes",
        "a.sparse_transposes",
        "b.iterations",
    )
    assert PROTOCOL_COST_MODEL.basis.feature_names[:16] == STORAGE_BASIS.feature_names
    assert STORAGE_BASIS.feature_names[:6] == (
        "dense.positions",
        "dense.slots",
        "sparse_list.positions",
        "sparse_list.entries",
        "sparse_list.position_bytes",
        "sparse_list.index_bytes",
    )


def test_fit_to_no_programs_is_zero():
    basis = Affine(PigeonBasis())
    fitted = CostModel(basis, np.ones(6)).fit([], [])
    np.testing.assert_array_equal(fitted.coefficients, np.zeros(6))
    assert fitted.predict(np.ones((2, 6))) == 0.0


class Unpenalized(RowMap):
    @property
    def n_features(self):
        return self.basis.n_features

    @property
    def ridge(self):
        return np.zeros(self.n_features)

    def transform(self, rows):
        return rows


def test_fit_is_nonnegative_relative_least_squares_with_basis_ridge(rng):
    # Six coefficients from three runtimes is underdetermined, and the penalty
    # is alpha times the ridge weights of each basis.
    raw = Unpenalized(TransposeBasis())
    squares = Quadratic(TransposeBasis())
    basis = Concat(raw, squares)
    np.testing.assert_array_equal(basis.ridge, [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(Affine(raw).ridge, [0, 0, 1])
    np.testing.assert_array_equal(Stratify(raw, 2, first_class).ridge, np.zeros(4))
    features = []
    for _ in range(3):
        rows = rng.random((3, 2))
        features.append(
            np.concatenate([raw.transform(rows), squares.transform(rows)], axis=1)
        )
    runtimes = np.array([1.0, 2.0, 3.0])
    design = np.stack([rows.sum(axis=0) for rows in features])
    for alpha in (0.0, 0.5, 2.0):
        fitted = CostModel(basis, np.zeros(6)).fit(features, runtimes, alpha)
        augmented = np.concatenate(
            [design / runtimes[:, None], np.sqrt(alpha) * np.diag(basis.ridge)]
        )
        expected = nnls(augmented, np.concatenate([np.ones(3), np.zeros(6)]))[0]
        np.testing.assert_allclose(fitted.coefficients, expected, atol=1e-9)
        assert fitted.coefficients.min() >= 0


def test_fit_clamps_negative_coefficients(rng):
    basis = Concat(PigeonBasis(), TransposeBasis())
    features = random_programs(rng, 7)
    design = np.stack([rows.sum(axis=0) for rows in features])
    noise = rng.lognormal(sigma=0.1, size=40)
    runtimes = design @ [1.0, -0.5, 3.0, 0.7, 0.2, -0.3, 0.5] * noise
    relative = design / runtimes[:, None]
    ones = np.ones(len(runtimes))
    fitted = CostModel(basis, np.zeros(7)).fit(features, runtimes)
    np.testing.assert_allclose(fitted.coefficients, nnls(relative, ones)[0])
    assert fitted.coefficients[1] == 0 and fitted.coefficients[5] == 0
    assert np.any(np.linalg.lstsq(relative, ones)[0] < 0)


def test_fit_requires_positive_runtimes(rng):
    features = random_programs(rng, 2, 3)
    with pytest.raises(ValueError, match="Expected positive runtimes"):
        CostModel(TransposeBasis(), np.zeros(2)).fit(features, [1.0, 0.0, 2.0])


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
    nonlinear = CostModel(Quadratic(FlopsBasis()), [1.0, 0.0, 0.0, 10.0])
    assert nonlinear.predict_cost(plan, {}, stats, sf) == 120.0
    assert set(stats) == {a}
    # Each query costs a constant.
    affine = CostModel(Affine(FlopsBasis()), [2.0, 3.0, 100.0])
    np.testing.assert_array_equal(
        affine.get_features(plan, {}, stats, sf), [[6, 2, 1], [2, 2, 1]]
    )
    assert affine.predict_cost(plan, {}, stats, sf) == 228.0
    machine = StatsMachine(sf, dict(stats))
    assert affine.predict_component_cost(Statement(plan.bodies[0]), machine) == 118.0


@pytest.mark.parametrize(
    "cost",
    [
        FLOPS_COST_MODEL,
        PIGEON,
        TRANSPOSE,
        LOOP_COST_MODEL,
        CostModel(Affine(PigeonBasis()), np.ones(6)),
        CostModel(Quadratic(Concat(PigeonBasis(), TransposeBasis())), np.ones(49)),
        CostModel(Stratify(PigeonBasis(), 2, first_class), np.ones(10)),
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
    cost = CostModel(Quadratic(basis), np.ones(49))
    features = basis.get_features(plan, {}, stats, sf)
    assert features.shape == (3, 7)
    assert cost.get_features(plan, {}, stats, sf).shape == (3, 49)
    expected = sum(row.sum() ** 2 for row in features)
    assert cost.predict_cost(plan, {}, stats, sf) == expected
    assert expected != features.sum() ** 2


@pytest.mark.parametrize(
    "cost",
    [
        PIGEON,
        TRANSPOSE,
        LOOP_COST_MODEL,
        CostModel(Affine(PigeonBasis()), np.arange(6)),
        CostModel(Quadratic(PigeonBasis()), np.ones(25)),
        CostModel(
            Concat(Quadratic(PigeonBasis()), Quadratic(TransposeBasis())), np.ones(29)
        ),
        CostModel(Quadratic(Concat(PigeonBasis(), TransposeBasis())), np.ones(49)),
        CostModel(Concat(PigeonBasis(), Quadratic(PigeonBasis())), np.ones(30)),
        CostModel(Stratify(PigeonBasis(), 3, above([2.0, 4.0], 0)), np.arange(15)),
    ],
)
def test_loop_costs_are_additive_over_prefixes(cost):
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
        assert cost.predict_cost(query, {}, stats, sf) == pytest.approx(expected)


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

    assert features.shape == (len(idxs) + 1, 5)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)
    assert model.predict_cost(plan, bindings, stats, sf) > 0
    # Every table walks, so nothing is looked up.
    np.testing.assert_array_equal(features[:, 3], 0)
    if idxs:
        assert np.all(features[:, :3].sum(axis=0) > 0)
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
            # The iterations of a merge are 13, 27, 11, 13, 29, and 21, and
            # the dense iterations are those of the loops' whole extents.
            [
                [13, 8, 8, 0, 20],
                [32, 19, 8, 0, 34],
                [11, 8, 8, 0, 21],
                [13, 10, 10, 0, 25],
                [34, 21, 10, 0, 36],
                [21, 14, 10, 0, 26],
            ],
        ),
        (
            UniformStatsFactory,
            [
                [11.079412928, 7.130046592, 6.905788373, 0, 17.818076074],
                [19.244186036, 13.951194992, 6.905788373, 0, 25.786424202],
                [9.273718471, 6.607801656, 6.905788373, 0, 17.538022651],
                [9.928979852, 7.705168554, 8.042275185, 0, 18.406439897],
                [20.596274450, 15.359836158, 8.042275185, 0, 27.074833635],
                [14.988081724, 11.044695822, 8.042275185, 0, 20.899160892],
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
                model.basis.get_component_features(
                    Prefix(prefix, expr, out_idxs), machine
                )
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


@pytest.mark.parametrize(
    "b, expected",
    [
        # A dense loop visits every row, and walks every column of each.
        (np.ones((4, 4)), [4, 16]),
        # A merge doesn't look ahead, so it visits the row which A and B
        # share, and walks the union of their columns there, though their
        # product is a single entry.
        (np.diag([1.0, 0.0, 0.0, 0.0]), [4, 4]),
    ],
)
def test_pigeon_counts_iterations_of_a_merge(b, expected):
    sf = DCStatsFactory()
    j, k = Field("j"), Field("k")
    a_, b_ = HardAlias("a"), HardAlias("b")
    stats = {a_: sf(fl.asarray(np.ones((4, 4))), (j, k)), b_: sf(fl.asarray(b), (j, k))}
    arg = MapJoin(Literal(ffuncs.mul), (Table(a_, (j, k)), Table(b_, (j, k))))
    expr = Aggregate(Literal(ffuncs.add), Literal(0.0), arg, (j, k))
    machine = StatsMachine(sf, stats)
    iterations = [
        PIGEON.basis.get_component_features(Prefix(prefix, expr, ()), machine)[0, 0]
        for prefix in [(j,), (j, k)]
    ]
    assert iterations == expected


@pytest.mark.parametrize(
    "op, protocols, iterations, lookups",
    [
        # Walks step through the union of x and y, which stores 7 of 10.
        (ffuncs.mul, (DefaultProtocol(), DefaultProtocol()), 7, 0),
        (ffuncs.mul, (Walk(), Walk()), 7, 0),
        (ffuncs.add, (Walk(), Walk()), 7, 0),
        # y is looked up at each of the 5 entries of x.
        (ffuncs.mul, (Walk(), Follow()), 5, 5),
        # Gallops visit the 3 entries x and y share, and the walk of y steps
        # through its 5.
        (ffuncs.mul, (Gallop(), Gallop()), 3, 6),
        (ffuncs.mul, (Gallop(), Walk()), 5, 3),
        # A sum runs wherever y could be, which is everywhere.
        (ffuncs.add, (Walk(), Follow()), 10, 10),
        # Without a walk or gallop, the loop is dense.
        (ffuncs.mul, (Follow(), Follow()), 10, 20),
    ],
)
def test_pigeon_counts_protocols(op, protocols, iterations, lookups):
    sf = ExactStatsFactory()
    i = Field("i")
    x, y = HardAlias("x"), HardAlias("y")
    stats: dict[Alias, TensorStats] = {
        x: sf(fl.asarray(np.array([0, 2, 0, 4, 5, 0, 0, 7, 0, 9.0])), (i,)),
        y: sf(fl.asarray(np.array([1, 0, 0, 3, 6, 0, 8, 7, 0, 0.0])), (i,)),
    }
    tables = tuple(
        Table(alias, (i,), (proto,))
        for alias, proto in zip((x, y), protocols, strict=True)
    )
    expr = MapJoin(Literal(op), tables)
    machine = StatsMachine(sf, dict(stats))
    features = PigeonBasis().get_component_features(Prefix((i,), expr, (i,)), machine)
    assert (features[0, 0], features[0, 3]) == (iterations, lookups)


def test_pigeon_counts_the_dense_iterations_of_a_loop():
    sf = ExactStatsFactory()
    i, j = Field("i"), Field("j")
    x, y = HardAlias("x"), HardAlias("y")
    stats: dict[Alias, TensorStats] = {
        x: sf(fl.asarray(np.ones((4, 4))), (i, j)),
        y: sf(fl.asarray(np.array([1.0, 0.0, 0.0, 0.0])), (j,)),
    }
    expr = MapJoin(Literal(ffuncs.mul), (Table(x, (i, j)), Table(y, (j,), (Walk(),))))
    machine = StatsMachine(sf, dict(stats))
    basis = PigeonBasis()
    # Each loop runs over its whole extent wherever the loops outside it go,
    # whatever its tables store or how they are read: the product restricts
    # j to one value, so loops inside j run once, while loops inside i run
    # for each of its 4 values.
    for prefix, space in (((), 0), ((i,), 4), ((i, j), 16), ((j,), 4), ((j, i), 4)):
        rows = basis.get_component_features(Prefix(prefix, expr, ()), machine)
        assert rows[0, 4] == space
    assert basis.feature_names[4] == "dense_iterations"


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
        model.basis.get_component_features(Prefix((), scalar), machine),
        [[0, 0, 1, 0, 0]],
    )
    assert model.predict_component_cost(Prefix((), scalar), machine) == 10
    np.testing.assert_array_equal(
        model.basis.get_component_features(Prefix((i,), broadcast), machine),
        [[1, 0, 1, 0, 1]],
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
    model = CostModel(
        Concat(*(sub.basis for sub in models)),
        np.concatenate([sub.coefficients for sub in models]),
    )
    model.validate_inputs(plan, bindings, stats, sf)

    assert model.basis.n_features == 7
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
    cost = CostModel(
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
                models[n].basis.get_component_features(prefix, machine)
                for n, prefix in enumerate(prefixes)
            )
        ),
    )
    assert cost.predict_cost(query, {}, stats, sf) == sum(
        models[n].predict_component_cost(prefix, machine)
        for n, prefix in enumerate(prefixes)
    )
