import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.cost_models import (
    ByKind,
    DenseLevelOption,
    ElementLevelOption,
    LevelChoice,
    LevelCostModel,
    LinearPredictor,
    QuadraticPredictor,
    SparseHashLevelOption,
)
from finch.autoschedule.formatter.smart_formatter import (
    CostFormatter,
    IterCostFormatter,
    StorageCostFormatter,
)
from finch.autoschedule.tensor_stats import DCStatsFactory
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
    Reorder,
    Table,
)


def _format_matrix(formatter, matrix, fill_value):
    i, j = Field("i"), Field("j")
    A, M = Alias("A"), Alias("M")
    stats_factory = DCStatsFactory()
    query = Query(Table(A, (i, j)), Table(M, (i, j)))
    machine = StatsMachine(
        stats_factory, {M: stats_factory(fl.asarray(matrix), (i, j))}
    )
    machine(query)
    shape_type = tuple(fl.ftype(np.intp) for _ in matrix.shape)
    return formatter.get_tensor_ftype(A, query, machine, fill_value, shape_type)


def test_storage_cost_formatter_uses_dense_levels_for_dense_matrx():
    i, j = Field("i"), Field("j")
    A, B = Alias("A"), Alias("B")
    matrix = np.ones((2, 3))
    tensor = fl.asarray(matrix)
    stats_factory = DCStatsFactory()
    stats = {A: stats_factory(tensor, (i, j))}
    capture = LogicCapture()
    formatter = StorageCostFormatter(capture)
    prgm = Plan(
        (
            Query(
                Table(B, (i, j)),
                Aggregate(
                    Literal(ffuncs.overwrite),
                    Literal(0.0),
                    Reorder(
                        MapJoin(
                            Literal(ffuncs.add),
                            (Table(A, (i, j)), Table(A, (i, j))),
                        ),
                        (i, j),
                    ),
                    (),
                ),
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
    ftype = _format_matrix(StorageCostFormatter(), matrix, np.float64(0.0))

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)


def test_iter_cost_formatter_uses_sparse_hash_for_sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    ftype = _format_matrix(IterCostFormatter(), matrix, np.float64(0.0))

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)


def _cost_output_pattern(formatter_cls, matrix):
    ftype = _format_matrix(formatter_cls(), matrix, np.float64(0.0))

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
    a = Alias("a")
    tensor = fl.asarray(np.eye(3, 4, dtype=np.float64) * [1, 1, 0, 0])
    query = Query(Table(a, (i, j)), Table(Literal(tensor), (i, j)))
    machine = StatsMachine(DCStatsFactory())
    machine(query)
    model = formatter_cls().cost_model

    costs = [
        model.predict_component_cost(
            LevelChoice(query, a, depth, num_pos, option), machine
        )
        for depth, num_pos, option in (
            (1, 7, DenseLevelOption()),
            (1, 7, SparseHashLevelOption()),
            (2, 5, ElementLevelOption()),
        )
    ]
    np.testing.assert_allclose(costs, expected)


def test_format_access_features_separate_sequential_and_scatter():
    i, j, k = Field("i"), Field("j"), Field("k")
    X, Y, A, C = Alias("X"), Alias("Y"), Alias("A"), Alias("C")
    stats_factory = DCStatsFactory()
    ones = fl.asarray(np.ones((3, 3)))
    machine = StatsMachine(
        stats_factory,
        {X: stats_factory(ones, (i, k)), Y: stats_factory(ones, (k, j))},
    )
    prgm = Plan(
        (
            # A[i, j] = sum_k X[i, k] * Y[k, j], looping over (i, k, j).
            Query(
                Table(A, (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0.0),
                    Reorder(
                        MapJoin(
                            Literal(ffuncs.mul), (Table(X, (i, k)), Table(Y, (k, j)))
                        ),
                        (i, k, j),
                    ),
                    (k,),
                ),
            ),
            # C[i, j] = A[i, j], looping over (i, j).
            Query(
                Table(C, (i, j)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0.0),
                    Reorder(Table(A, (i, j)), (i, j)),
                    (),
                ),
            ),
            Produces((C,)),
        )
    )
    machine(prgm)

    def accessed(depth, option):
        model = LevelCostModel(option, LinearPredictor(np.zeros(option.n_features + 4)))
        features = model.get_component_features(
            LevelChoice(prgm, A, depth, 1.0, option), machine
        )
        return tuple(features.sum(axis=0)[-4:] > 0)

    # Reading A in loop order is sequential at every level. Writing A is
    # sequential at i, but the k loop makes it scatter at j.
    assert accessed(0, DenseLevelOption()) == (True, False, True, False)
    assert accessed(1, DenseLevelOption()) == (True, False, False, True)
    assert accessed(2, ElementLevelOption()) == (True, False, False, True)


def test_nonlinear_format_cost_retains_allocations_and_query_accesses():
    i = Field("i")
    a, b, c = Alias("a"), Alias("b"), Alias("c")
    tensor = fl.asarray(np.ones(3))
    sf = DCStatsFactory()
    stats = {a: sf(tensor, (i,))}
    plan = Plan(
        (
            Query(Table(b, (i,)), Table(a, (i,))),
            Query(Table(c, (i,)), Table(b, (i,))),
            Produces((c,)),
        )
    )
    cost = ByKind(
        {
            "dense": LevelCostModel(
                DenseLevelOption(), QuadraticPredictor(np.ones(36))
            ),
            "element": LevelCostModel(
                ElementLevelOption(), QuadraticPredictor(np.ones(36))
            ),
        }
    )
    formats = {b: [DenseLevelOption()]}
    features = cost.get_features(plan, formats, stats, sf)
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
            LevelChoice(plan, b, depth, num_pos, option), machine
        )
        for depth, num_pos, option in (
            (0, 1, DenseLevelOption()),
            (1, 3, ElementLevelOption()),
        )
    )
    assert cost.get_features(Plan(()), {}, {}, sf).shape == (0, 12)
    assert cost.predict_cost(Plan(()), {}, {}, sf) == 0.0
    fmt = CostFormatter(cost_model=cost, options=formats[b]).get_tensor_ftype(
        b, plan, machine, tensor.fill_value, (fl.ftype(np.intp),)
    )
    assert isinstance(fmt.lvl_t, fl.DenseLevelFType)


def test_format_selection_uses_custom_predictor():
    i = Field("i")
    a = Alias("a")
    tensor = fl.asarray(np.ones(3))
    query = Query(Table(a, (i,)), Table(Literal(tensor), (i,)))
    machine = StatsMachine(DCStatsFactory())
    machine(query)
    zeros = {
        option.kind: LevelCostModel(
            option, LinearPredictor(np.zeros(option.n_features + 4))
        )
        for option in (
            DenseLevelOption(),
            SparseHashLevelOption(),
            ElementLevelOption(),
        )
    }
    slots = LevelCostModel(
        DenseLevelOption(), lambda rows: float(np.square(rows[:, 1]).sum())
    )
    linear = ByKind(zeros)
    nonlinear = ByKind({**zeros, "dense": slots})

    def choose(cost):
        return CostFormatter(cost_model=cost).get_tensor_ftype(
            a, query, machine, tensor.fill_value, (fl.ftype(np.intp),)
        )

    assert isinstance(choose(linear).lvl_t, fl.DenseLevelFType)
    assert isinstance(choose(nonlinear).lvl_t, fl.SparseHashLevelFType)
