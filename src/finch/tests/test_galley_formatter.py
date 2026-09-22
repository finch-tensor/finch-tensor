import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.formatter import GalleyFormatter
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.finch_logic import (
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
from finch.tensor.level import ElementLevelFType

i, j = Field("i"), Field("j")
A, B = Alias("A"), Alias("B")


def level_ftypes(tensor_ftype):
    """The level ftypes of a fiber tensor, outermost first."""
    levels = []
    lvl = tensor_ftype.lvl_t
    while not isinstance(lvl, ElementLevelFType):
        levels.append(type(lvl))
        lvl = lvl.lvl_t
    return levels


def format_query(rhs, array):
    """Run `GalleyFormatter` over a single query and return the output ftype."""
    tensor = fl.asarray(array)
    stats_factory = DCStatsFactory()
    capture = LogicCapture()
    GalleyFormatter(capture).lower(
        Plan((Query(B, rhs), Produces((B,)))),
        {A: tensor.ftype},
        {A: stats_factory(tensor, (i, j))},
        stats_factory,
    )
    return capture.last_bindings[B]


def sparse_matrix():
    matrix = np.zeros((100, 100))
    matrix[0, 0] = 1.0
    matrix[99, 99] = 1.0
    return matrix


def test_galley_formatter_uses_dense_levels_for_a_dense_matrix():
    rhs = MapJoin(Literal(ffuncs.add), (Table(A, (i, j)), Table(A, (i, j))))
    ftype = format_query(rhs, np.ones((10, 10)))

    assert isinstance(ftype, fl.FiberTensorFType)
    assert level_ftypes(ftype) == [fl.DenseLevelFType, fl.DenseLevelFType]


def test_galley_formatter_uses_sparse_lists_for_a_sparse_matrix():
    rhs = MapJoin(Literal(ffuncs.add), (Table(A, (i, j)), Table(A, (i, j))))
    ftype = format_query(rhs, sparse_matrix())

    assert level_ftypes(ftype) == [fl.SparseListLevelFType, fl.SparseListLevelFType]


def test_galley_formatter_uses_hash_levels_when_writes_are_random():
    # A transpose is looped in the order its input is stored, so every level of
    # the output is written out of order.
    ftype = format_query(Reorder(Table(A, (i, j)), (j, i)), sparse_matrix())

    assert level_ftypes(ftype) == [fl.SparseHashLevelFType, fl.SparseHashLevelFType]
