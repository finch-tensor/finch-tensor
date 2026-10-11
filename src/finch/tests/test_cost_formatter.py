import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import DynamicFillFType, StaticFillFType
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.formatter.galley_formatter import estimate_nnz
from finch.autoschedule.formatter.smart_formatter import (
    IterCostFormatter,
    StorageCostFormatter,
    nnz_after,
)
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DCStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.uniform_stats import UniformStats
from finch.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
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
    assert nnz_after((i, j), stats, factory, 0) == pytest.approx(3.5)
    assert estimate_nnz(stats, factory, (i,)) == pytest.approx(3.5)

    leaf = fl.element(fill_type, fl.float64)
    level = level_type(fl.dense(leaf))
    pos_size = np.dtype(np.intp).itemsize
    expected = (2 + 3.5) * pos_size + 3.5 * 3 * 8
    assert level.level_cost((i, j), stats, factory, 1, 0) == pytest.approx(expected)
    assert stats.fill_type == fill_type


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
                MapJoin(
                    Literal(ffuncs.add),
                    (Table(A, (i, j)), Table(A, (i, j))),
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
    i, j = Field("i"), Field("j")
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    stats_factory = DCStatsFactory()
    stats = stats_factory(fl.asarray(matrix), (i, j))

    formatter = StorageCostFormatter()
    formatter._stats_factory = stats_factory
    ftype = formatter.get_tensor_ftype(
        StaticFillFType(np.float64(0.0)), (fl.ftype(np.intp), fl.ftype(np.intp)), stats
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t, fl.SparseHashLevelFType)
    assert isinstance(ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)


def test_iter_cost_formatter_uses_sparse_hash_for_sparse_matrix():
    i, j = Field("i"), Field("j")
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    stats_factory = DCStatsFactory()
    stats = stats_factory(fl.asarray(matrix), (i, j))

    formatter = IterCostFormatter()
    formatter._stats_factory = stats_factory
    ftype = formatter.get_tensor_ftype(
        StaticFillFType(np.float64(0.0)), (fl.ftype(np.intp), fl.ftype(np.intp)), stats
    )

    assert isinstance(ftype, fl.FiberTensorFType)
    assert isinstance(ftype.lvl_t, fl.SparseHashLevelFType)


def _cost_output_pattern(formatter_cls, stats, stats_factory):
    formatter = formatter_cls()
    formatter._stats_factory = stats_factory
    shape_type = tuple(fl.ftype(np.intp) for _ in stats.index_order)
    ftype = formatter.get_tensor_ftype(stats.fill_type, shape_type, stats)

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
    i, j = Field("i"), Field("j")
    stats_factory = DCStatsFactory()
    dense_matrix = np.ones((4, 4))
    sparse_matrix = np.zeros((50, 50))
    sparse_matrix[0, 0] = 1
    sparse_matrix[49, 49] = 1

    dense_stats = stats_factory(fl.asarray(dense_matrix), (i, j))
    sparse_stats = stats_factory(fl.asarray(sparse_matrix), (i, j))

    assert _cost_output_pattern(StorageCostFormatter, dense_stats, stats_factory) == (
        "dense",
        "dense",
    )
    assert _cost_output_pattern(StorageCostFormatter, sparse_stats, stats_factory) == (
        "sparse_hash",
        "sparse_hash",
    )
