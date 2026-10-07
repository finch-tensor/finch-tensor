import pytest

import numpy as np

import finch as ft
from finch import finch_logic as lgc
from finch.algebra import ffuncs
from finch.autoschedule import (
    INTERPRET_NOTATION,
    CompilerFormLowerer,
    DefaultLogicFormatter,
    LogicCapture,
    LogicCompiler,
    LogicExecutor,
)
from finch.autoschedule.formatter.galley_formatter import GalleyFormatter
from finch.autoschedule.formatter.smart_formatter import (
    FDFormatter,
    StorageCostFormatter,
)
from finch.autoschedule.loop_orderer import loop_order_bnb, loop_order_greedy
from finch.autoschedule.loop_orderer.loop_order_bnb import (
    BFSLoopOrderer,
    BruteForceLoopOrderer,
    DFSLoopOrderer,
)
from finch.autoschedule.loop_orderer.loop_order_greedy import GreedyLoopOrderer
from finch.autoschedule.tensor_stats import DenseStatsFactory, FDStatsFactory


@pytest.mark.parametrize("shape", [(5,), (0,), (2, 3), ()])
def test_query_redefines_shape(shape):
    i, j = map(lgc.Field, "ij")
    a, b, scratch, before, after = map(
        lgc.HardAlias, ("a", "b", "scratch", "before", "after")
    )
    fields = (i, j)[: len(shape)]
    data_a = np.arange(3)
    data_b = np.arange(np.prod(shape)).reshape(shape)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(scratch, (i,)), lgc.Table(a, (i,))),
            lgc.Query(lgc.Table(before, (i,)), lgc.Table(scratch, (i,))),
            lgc.Query(lgc.Table(scratch, fields), lgc.Table(b, fields)),
            lgc.Query(lgc.Table(after, fields), lgc.Table(scratch, fields)),
            lgc.Produces((before, after)),
        )
    )
    dims = plan.infer_shape({a: data_a.shape, b: shape})
    assert dims[before] == (3,)
    assert dims[scratch] == dims[after] == shape
    result = lgc.LogicInterpreter()(
        plan, {a: ft.asarray(data_a), b: ft.asarray(data_b)}
    )
    np.testing.assert_array_equal(result[0].to_numpy(), data_a)
    np.testing.assert_array_equal(result[1].to_numpy(), data_b)


def test_query_reads_previous_shape_before_rebinding():
    i, j = map(lgc.Field, "ij")
    a = lgc.HardAlias("a")
    data = np.arange(6).reshape(2, 3)
    original = ft.asarray(data)
    bindings = {a: original}
    plan = lgc.Plan(
        (
            lgc.Query(
                lgc.Table(a, (i,)),
                lgc.Aggregate(
                    lgc.Literal(ffuncs.add), lgc.Literal(0), lgc.Table(a, (i, j)), (j,)
                ),
            ),
            lgc.Produces((a,)),
        )
    )
    assert plan.infer_shape({a: data.shape}) == {a: (2,)}
    result = lgc.LogicInterpreter()(plan, bindings)[0]
    assert bindings[a] is result
    assert result is not original
    np.testing.assert_array_equal(result.to_numpy(), data.sum(axis=1))
    np.testing.assert_array_equal(original.to_numpy(), data)


@pytest.mark.parametrize("op", [ffuncs.add, ffuncs.overwrite])
@pytest.mark.parametrize("old_size,new_size", [(3, 5), (0, 1), (1, 0)])
def test_queryinto_rejects_shape_change(op, old_size, new_size):
    i = lgc.Field("i")
    a, b = map(lgc.HardAlias, "ab")
    original = ft.asarray(np.arange(old_size))
    bindings = {a: original, b: ft.asarray(np.arange(new_size))}
    query = lgc.QueryInto(lgc.Table(a, (i,)), lgc.Literal(op), lgc.Table(b, (i,)))
    dims: dict[lgc.Alias, tuple[int, ...]] = {a: (old_size,), b: (new_size,)}
    with pytest.raises(ValueError, match="Dimension mismatch"):
        query.infer_shape(dims)
    assert dims[a] == (old_size,)
    with pytest.raises(ValueError):
        lgc.LogicInterpreter()(query, bindings)
    assert bindings[a] is original
    np.testing.assert_array_equal(original.to_numpy(), np.arange(old_size))


@pytest.mark.parametrize("op", [ffuncs.add, ffuncs.overwrite])
def test_queryinto_uses_redefined_shape(op):
    i = lgc.Field("i")
    a, b = map(lgc.HardAlias, "ab")
    original = ft.asarray(np.arange(3))
    bindings = {a: original, b: ft.asarray(np.arange(5))}
    dims: dict[lgc.Alias, tuple[int, ...]] = {a: (3,), b: (5,)}
    define = lgc.Query(lgc.Table(a, (i,)), lgc.Table(b, (i,)))
    update = lgc.QueryInto(lgc.Table(a, (i,)), lgc.Literal(op), lgc.Table(b, (i,)))
    define.infer_shape(dims)
    update.infer_shape(dims)
    assert dims[a] == (5,)
    interpreter = lgc.LogicInterpreter()
    interpreter(define, bindings)
    redefined = bindings[a]
    interpreter(update, bindings)
    assert bindings[a] is redefined
    np.testing.assert_array_equal(redefined.to_numpy(), op(np.arange(5), np.arange(5)))
    np.testing.assert_array_equal(original.to_numpy(), np.arange(3))


@pytest.mark.parametrize(
    "formatter,factory_cls",
    [
        (DefaultLogicFormatter, DenseStatsFactory),
        (FDFormatter, FDStatsFactory),
        (StorageCostFormatter, DenseStatsFactory),
        (GalleyFormatter, DenseStatsFactory),
    ],
)
def test_formatters_infer_each_definition(monkeypatch, formatter, factory_cls):
    i, j = map(lgc.Field, "ij")
    a, b, scratch, before, after = map(
        lgc.HardAlias, ("a", "b", "scratch", "before", "after")
    )
    a_tensor, b_tensor = ft.asarray(np.ones(3)), ft.asarray(np.ones((2, 4)))
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(scratch, (i,)), lgc.Table(a, (i,))),
            lgc.Query(lgc.Table(before, (i,)), lgc.Table(scratch, (i,))),
            lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(b, (i, j))),
            lgc.Query(lgc.Table(after, (i, j)), lgc.Table(scratch, (i, j))),
            lgc.Produces((before, after)),
        )
    )
    factory = factory_cls()
    capture = LogicCapture()
    monkeypatch.setattr(capture, "ctx", lambda *args: None)
    formatter(capture)(
        plan,
        {a: a_tensor.ftype, b: b_tensor.ftype},
        {a: factory(a_tensor, (i,)), b: factory(b_tensor, (i, j))},
        factory,
    )
    assert capture.last_bindings[scratch].ndim == 1
    assert capture.last_bindings[before].ndim == 1
    assert capture.last_bindings[after].ndim == 2


@pytest.mark.parametrize(
    "orderer,module,search",
    [
        (GreedyLoopOrderer, loop_order_greedy, "greedy_loop_order"),
        (BFSLoopOrderer, loop_order_bnb, "loop_order_bfs"),
        (DFSLoopOrderer, loop_order_bnb, "loop_order_dfs"),
        (BruteForceLoopOrderer, loop_order_bnb, "loop_order_brute_force"),
    ],
)
def test_loop_orderers_refresh_stats(monkeypatch, orderer, module, search):
    i = lgc.Field("i")
    a, b, scratch, out, result = map(
        lgc.HardAlias, ("a", "b", "scratch", "out", "result")
    )
    copy = lgc.Query(lgc.Table(out, (i,)), lgc.Table(scratch, (i,)))
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(scratch, (i,)), lgc.Table(a, (i,))),
            copy,
            lgc.Query(lgc.Table(scratch, (i,)), lgc.Table(b, (i,))),
            copy,
            lgc.Query(
                lgc.Table(result, ()),
                lgc.Aggregate(
                    lgc.Literal(ffuncs.add),
                    lgc.Literal(0.0),
                    lgc.Table(out, (i,)),
                    (i,),
                ),
            ),
            lgc.Produces((result,)),
        )
    )
    seen_sizes = []

    def choose(expr, factory, stats, output_vars, **kwargs):
        seen_sizes.append(stats[out].dim_sizes[i])
        return expr.fields()

    monkeypatch.setattr(module, search, choose)
    factory = DenseStatsFactory()
    orderer().set_loop_orders(
        plan,
        {
            a: factory(ft.asarray(np.ones(2)), (i,)),
            b: factory(ft.asarray(np.ones(5)), (i,)),
        },
        factory,
    )
    assert seen_sizes == [5]


def test_compiled_intermediate_redefinition():
    i, j = map(lgc.Field, "ij")
    a, b, scratch, before, after = map(
        lgc.HardAlias, ("a", "b", "scratch", "before", "after")
    )
    data_a, data_b = np.arange(3), np.arange(8).reshape(2, 4)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(scratch, (i,)), lgc.Table(a, (i,))),
            lgc.Query(
                lgc.Table(before, (i,)),
                lgc.MapJoin(
                    lgc.Literal(ffuncs.add), (lgc.Table(scratch, (i,)), lgc.Literal(1))
                ),
            ),
            lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(b, (i, j))),
            lgc.Query(
                lgc.Table(after, (i, j)),
                lgc.MapJoin(
                    lgc.Literal(ffuncs.add),
                    (lgc.Table(scratch, (i, j)), lgc.Literal(2)),
                ),
            ),
            lgc.Produces((before, after)),
        )
    )
    result = INTERPRET_NOTATION(plan, {a: ft.asarray(data_a), b: ft.asarray(data_b)})
    np.testing.assert_array_equal(result[0].to_numpy(), data_a + 1)
    np.testing.assert_array_equal(result[1].to_numpy(), data_b + 2)


@pytest.mark.parametrize("query_cls", [lgc.Query, lgc.QueryInto])
def test_allocated_buffer_shape_mismatch(query_cls):
    i = lgc.Field("i")
    a, b, out = map(lgc.HardAlias, ("a", "b", "out"))
    if query_cls is lgc.Query:
        query = lgc.Query(lgc.Table(out, (i,)), lgc.Table(b, (i,)))
    else:
        query = lgc.QueryInto(
            lgc.Table(out, (i,)),
            lgc.Literal(ffuncs.add),
            lgc.Reorder(lgc.Table(b, (i,)), (i,)),
        )
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(out, (i,)), lgc.Table(a, (i,))),
            query,
            lgc.Produces((out,)),
        )
    )
    executor = LogicExecutor(
        DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler()))
    )
    original = ft.asarray(np.full(3, -1))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        executor(
            plan,
            {a: ft.asarray(np.arange(3)), b: ft.asarray(np.arange(5)), out: original},
        )
    np.testing.assert_array_equal(original.to_numpy(), np.full(3, -1))


def test_allocated_buffer_rank_change():
    i, j = map(lgc.Field, "ij")
    a, b, out = map(lgc.HardAlias, ("a", "b", "out"))
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(out, (i,)), lgc.Table(a, (i,))),
            lgc.Query(lgc.Table(out, (i, j)), lgc.Table(b, (i, j))),
            lgc.Produces((out,)),
        )
    )
    executor = LogicExecutor(
        DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler()))
    )
    with pytest.raises(ValueError, match="Cannot change the rank of allocated tensor"):
        executor(plan, {a: ft.asarray(np.ones(3)), b: ft.asarray(np.ones((2, 4)))})


def test_allocation_inference_preserves_query_unit_dimensions():
    i = lgc.Field("i")
    a, b = map(lgc.HardAlias, "ab")
    original = ft.asarray(np.arange(3))
    plan = lgc.Plan(
        (lgc.Query(lgc.Table(a, (i,)), lgc.Table(b, ())), lgc.Produces((a,)))
    )
    executor = LogicExecutor(
        DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler()))
    )
    with pytest.raises(ValueError, match="dimension must have size 1"):
        executor(plan, {a: original, b: ft.asarray(np.array(2))})
    np.testing.assert_array_equal(original.to_numpy(), np.arange(3))
