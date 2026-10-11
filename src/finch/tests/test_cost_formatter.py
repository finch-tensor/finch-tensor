import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.formatter.smart_formatter import (
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
        model.predict_format_cost(a, level, num_pos, key, query, machine)
        for level, num_pos, key in (
            (1, 7, "dense"),
            (1, 7, "sparse_hash"),
            (2, 5, None),
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
    model = StorageCostFormatter().cost_model

    def accessed(level, key, block):
        features = model.get_format_features(A, level, 1.0, key, prgm, machine)
        return tuple(features[10 * block + 6 : 10 * block + 10] > 0)

    # Reading A in loop order is sequential at every level. Writing A is
    # sequential at i, but the k loop makes it scatter at j.
    assert accessed(0, "dense", 0) == (True, False, True, False)
    assert accessed(1, "dense", 0) == (True, False, False, True)
    assert accessed(2, None, 2) == (True, False, False, True)
