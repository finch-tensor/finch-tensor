import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch.algebra import StaticFillFType, TensorFType, ffuncs
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.formatter.galley_formatter import GalleyFormatter
from finch.autoschedule.formatter.smart_formatter import (
    CostFormatter,
    FDFormatter,
)
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DenseStatsFactory,
    FDStats,
    FDStatsFactory,
    StatsInterpreter,
)
from finch.autoschedule.tensor_stats.exact_stats import (
    ExactStats,
    ExactStatsFactory,
)
from finch.autoschedule.util import fuse_loop_order


@pytest.mark.parametrize("nfused", range(4))
@pytest.mark.parametrize("shape", [(2, 3, 4), (0, 3, 4), (2, 0, 4)])
@pytest.mark.parametrize(
    "formatter_cls,factory_cls",
    [
        (FDFormatter, FDStatsFactory),
        (CostFormatter, DenseStatsFactory),
        (GalleyFormatter, DenseStatsFactory),
    ],
)
def test_formatters_keep_logical_stats_and_format_stored_dimensions(
    monkeypatch, formatter_cls, factory_cls, nfused, shape
):
    fields = tuple(map(lgc.Field, "tij"))
    a, scratch, out = map(lgc.HardAlias, ("a", "scratch", "out"))
    fused = lgc.FusedAlias(scratch, nfused)
    tensor = ft.asarray(np.ones(shape))
    factory = factory_cls()
    capture = LogicCapture()
    # Inspect the formatter's output before lowering temporal storage to loops.
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)), fields
            ),
            fuse_loop_order(
                lgc.Query(lgc.Table(out, fields), lgc.Table(fused, fields)), fields
            ),
            lgc.Produces((out,)),
        )
    )
    formatter_cls(capture).lower(
        plan, {a: tensor.ftype}, {a: factory(tensor, fields)}, factory
    )
    assert fused in capture.last_stats
    assert scratch not in capture.last_stats
    assert capture.last_stats[fused].index_order == fields
    assert capture.last_stats[out].index_order == fields
    scratch_type = capture.last_bindings[scratch]
    assert scratch_type.shape_type == tensor.shape_type[nfused:]
    assert scratch_type.ndim == 3 - nfused
    assert capture.last_bindings[out].ndim == 3
    scratch_tensor = scratch_type.construct(tensor.shape[nfused:])
    assert scratch_tensor.shape == tensor.shape[nfused:]


@pytest.mark.parametrize("formatter_cls", [GalleyFormatter, CostFormatter])
def test_statistics_for_alias_views_do_not_collide(monkeypatch, formatter_cls):
    i, j = lgc.Field("i"), lgc.Field("j")
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    # Both views store every dimension of scratch, so one format holds both.
    fused = lgc.FusedAlias(scratch, 0)
    tensor = ft.asarray(np.ones((2, 3)))
    factory = DenseStatsFactory()
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(a, (i, j))), (i, j)
            ),
            fuse_loop_order(
                lgc.Query(lgc.Table(fused, (j, i)), lgc.Table(scratch, (i, j))), (j, i)
            ),
            lgc.Produces((fused,)),
        )
    )
    formatter_cls(capture).lower(
        plan, {a: tensor.ftype}, {a: factory(tensor, (i, j))}, factory
    )
    assert capture.last_stats[scratch].index_order == (i, j)
    assert capture.last_stats[fused].index_order == (j, i)
    assert capture.last_stats[scratch] is not capture.last_stats[fused]
    with pytest.raises(ValueError, match="undefined tensor alias"):
        StatsInterpreter(factory)(fused, {scratch: factory(tensor, (i, j))})


@pytest.mark.parametrize("nfused", range(5))
@pytest.mark.parametrize("shape", [(2, 2, 3, 4), (0, 2, 3, 4)])
def test_galley_formats_only_the_fields_a_workspace_stores(monkeypatch, nfused, shape):
    data = np.zeros(shape, dtype=np.float64)
    if data.size:
        data[0, 0] = 1
        data[1, 0, 0, 0] = 1
        data[1, 1, 1, :] = 1
    fields = tuple(map(lgc.Field, "tuij"))
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, nfused)
    tensor = ft.asarray(data)
    factory = ExactStatsFactory()
    stats = factory(tensor, fields)
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)),
            lgc.Produces((fused,)),
        )
    )
    GalleyFormatter(capture).lower(plan, {a: tensor.ftype}, {a: stats}, factory)

    # The workspace stores one level for each field it isn't fused over.
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    assert fmt.shape_type == tensor.shape_type[nfused:]
    assert capture.last_stats[fused].index_order == fields
    assert stats.index_order == fields


def test_galley_formats_a_workspace_by_its_slices(monkeypatch):
    # Each slice of the diagonal holds one value, though every slice together
    # covers the whole of each row, so the slice it stores is sparse.
    t, i = fields = tuple(map(lgc.Field, "ti"))
    tensor = ft.asarray(np.eye(100))
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    factory = ExactStatsFactory()
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)), fields
            ),
            lgc.Produces((fused,)),
        )
    )
    GalleyFormatter(capture).lower(
        plan, {a: tensor.ftype}, {a: factory(tensor, fields)}, factory
    )
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    assert isinstance(fmt.lvl_t, ft.SparseListLevelFType)
    assert isinstance(fmt.lvl_t.lvl_t, ft.ElementLevelFType)


def test_galley_makes_a_small_scattered_workspace_a_byte_map(monkeypatch):
    # C[t, i] = sum_j A[t, j] B[j, i], row by row: the loop over j comes
    # between t and i, so each row of C is scattered into, through a workspace
    # fused over t. A row holds a few of its 50 slots, too sparse for a dense
    # or byte map level by density, but a slice that small is a byte map, which
    # resets and copies out cheaply, rather than a hash.
    t, j, i = fields = tuple(map(lgc.Field, "tji"))
    rng = np.random.default_rng(0)
    n = 50
    a_data, b_data = np.zeros((n, n)), np.zeros((n, n))
    a_data[np.arange(n), rng.integers(0, n, n)] = 1.0
    b_data[np.arange(n), rng.integers(0, n, n)] = 1.0
    a, b, scratch = lgc.HardAlias("a"), lgc.HardAlias("b"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    factory = ExactStatsFactory()
    stats: dict[lgc.Alias, ExactStats] = {
        a: factory(ft.asarray(a_data), (t, j)),
        b: factory(ft.asarray(b_data), (j, i)),
    }
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    rhs = lgc.Aggregate(
        lgc.Literal(ffuncs.add),
        lgc.Literal(0.0),
        lgc.MapJoin(
            lgc.Literal(ffuncs.mul), (lgc.Table(a, (t, j)), lgc.Table(b, (j, i)))
        ),
        (j,),
    )
    plan = lgc.Plan(
        (
            fuse_loop_order(lgc.Query(lgc.Table(fused, (t, i)), rhs), fields),
            lgc.Produces((fused,)),
        )
    )
    bindings: dict[lgc.Alias, TensorFType] = {
        a: ft.asarray(a_data).ftype,
        b: ft.asarray(b_data).ftype,
    }
    GalleyFormatter(capture).lower(plan, bindings, stats, factory)
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    assert isinstance(fmt.lvl_t, ft.SparseByteMapLevelFType)


@pytest.mark.parametrize("formatter_cls", [CostFormatter])
def test_cost_formatters_format_the_largest_fused_slice(monkeypatch, formatter_cls):
    # Each slice of the diagonal holds one value, though every slice together
    # covers the whole of each row.
    t, i = fields = tuple(map(lgc.Field, "ti"))
    tensor = ft.asarray(np.eye(8))
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    factory = ExactStatsFactory()
    stats = factory(tensor, fields)
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)), fields
            ),
            lgc.Produces((fused,)),
        )
    )
    formatter_cls(capture).lower(plan, {a: tensor.ftype}, {a: stats}, factory)
    # The slice is stored, as one level over i, of whichever kind the model
    # prices lowest.
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    assert isinstance(fmt.lvl_t.lvl_t, ft.ElementLevelFType)


@pytest.mark.parametrize("transpose", [False, True])
def test_galley_removes_fused_fields_from_loop_order(monkeypatch, transpose):
    data = np.zeros((100, 2, 100))
    data[0, 0, 0] = 1
    data[99, 1, 99] = 1
    i, t, j = fields = tuple(map(lgc.Field, "itj"))
    output_fields = (t, j, i) if transpose else (t, i, j)
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    tensor = ft.asarray(data)
    factory = ExactStatsFactory()
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(fused, output_fields), lgc.Table(a, fields)),
                (t, i, j),
            ),
            lgc.Produces((fused,)),
        )
    )
    GalleyFormatter(capture).lower(
        plan, {a: tensor.ftype}, {a: factory(tensor, fields)}, factory
    )
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    expected = ft.SparseHashLevelFType if transpose else ft.SparseListLevelFType
    assert isinstance(fmt.lvl_t, expected)
    assert isinstance(fmt.lvl_t.lvl_t, expected)
    assert capture.last_stats[fused].index_order == output_fields


def test_fd_formatter_projects_temporal_fields_before_choosing_levels(monkeypatch):
    t, i, j = fields = tuple(map(lgc.Field, "tij"))
    stats = FDStats(
        BaseTensorStats(fields, {t: 10, i: 3, j: 4}, StaticFillFType(0.0)),
        {frozenset({i}), frozenset({i, j})},
    )
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    tensor = ft.asarray(np.ones((10, 3, 4)))
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)),
            lgc.Produces((fused,)),
        )
    )
    FDFormatter(capture).lower(plan, {a: tensor.ftype}, {a: stats}, FDStatsFactory())
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    assert isinstance(fmt.lvl_t, ft.DenseLevelFType)
    assert isinstance(fmt.lvl_t.lvl_t, ft.DenseLevelFType)
    assert isinstance(fmt.lvl_t.lvl_t.lvl_t, ft.ElementLevelFType)
    assert stats.index_order == fields
    assert stats.dense_props == {frozenset({i}), frozenset({i, j})}
