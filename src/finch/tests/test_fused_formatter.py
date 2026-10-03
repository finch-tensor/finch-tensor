import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.formatter.galley_formatter import GalleyFormatter
from finch.autoschedule.formatter.smart_formatter import (
    FDFormatter,
    IterCostFormatter,
    StorageCostFormatter,
)
from finch.autoschedule.tensor_stats import (
    BaseTensorStats,
    DenseStatsFactory,
    FDStats,
    FDStatsFactory,
    StatsInterpreter,
)
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory


@pytest.mark.parametrize("nfused", range(4))
@pytest.mark.parametrize("shape", [(2, 3, 4), (0, 3, 4), (2, 0, 4)])
@pytest.mark.parametrize(
    "formatter_cls,factory_cls",
    [
        (FDFormatter, FDStatsFactory),
        (StorageCostFormatter, DenseStatsFactory),
        (IterCostFormatter, DenseStatsFactory),
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
            lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)),
            lgc.Query(lgc.Table(out, fields), lgc.Table(fused, fields)),
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


@pytest.mark.parametrize("formatter_cls", [GalleyFormatter, StorageCostFormatter])
def test_statistics_for_alias_views_do_not_collide(monkeypatch, formatter_cls):
    i, j = lgc.Field("i"), lgc.Field("j")
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    tensor = ft.asarray(np.ones((2, 3)))
    factory = DenseStatsFactory()
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(a, (i, j))),
            lgc.Query(lgc.Table(fused, (j, i)), lgc.Table(scratch, (i, j))),
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
def test_galley_uses_support_projected_over_fused_dimensions(
    monkeypatch, nfused, shape
):
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

    projected = np.any(data, axis=tuple(range(nfused))).astype(data.dtype)
    projected_stats = factory(ft.asarray(projected), fields[nfused:])
    expected = GalleyFormatter().get_tensor_ftype(
        stats.fill_value,
        tensor.shape_type[nfused:],
        projected_stats,
        factory,
        fields[nfused:],
    )
    assert capture.last_bindings[scratch] == expected
    assert capture.last_stats[fused].index_order == fields
    assert stats.index_order == fields


@pytest.mark.parametrize("formatter_cls", [StorageCostFormatter, IterCostFormatter])
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
            lgc.Query(lgc.Table(fused, fields), lgc.Table(a, fields)),
            lgc.Produces((fused,)),
        )
    )
    formatter_cls(capture).lower(plan, {a: tensor.ftype}, {a: stats}, factory)
    fmt = capture.last_bindings[scratch]
    assert isinstance(fmt, ft.FiberTensorFType)
    assert isinstance(fmt.lvl_t, ft.SparseHashLevelFType)

    # The support projected over the fused dimension would be dense.
    formatter = formatter_cls()
    formatter._stats_factory = factory
    projected = factory(ft.asarray(np.ones(8)), (i,))
    projected_fmt = formatter.get_tensor_ftype(
        stats.fill_value, tensor.shape_type[1:], projected
    )
    assert isinstance(projected_fmt, ft.FiberTensorFType)
    assert isinstance(projected_fmt.lvl_t, ft.DenseLevelFType)


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
            lgc.Query(lgc.Table(fused, output_fields), lgc.Table(a, fields)),
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
        BaseTensorStats(fields, {t: 10, i: 3, j: 4}, 0.0),
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
