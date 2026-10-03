import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import TensorFType, ffuncs, ftype
from finch.autoschedule import CompilerFormLowerer, LogicCapture, NotationGenerator
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.normalize import normalize_names
from finch.autoschedule.stages import AliasedForm, CompilerForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsInterpreter
from finch.compile import NotationCompiler
from finch.finch_logic.interpreter import MockFusedTensor


def test_fused_alias_node():
    alias = lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(alias, 1)
    assert isinstance(alias, lgc.Alias)
    assert alias.unfused is alias
    assert isinstance(fused, lgc.Alias)
    assert not isinstance(fused, lgc.HardAlias)
    match alias:
        case lgc.Alias(name):
            assert name == "scratch"
        case _:
            pytest.fail("HardAlias did not match its Alias parent")
    assert fused.alias == alias
    assert fused.name == alias.name
    assert fused.unfused == alias
    assert fused == lgc.FusedAlias(alias, 1)
    assert fused != lgc.FusedAlias(alias, 2)
    assert fused != alias
    assert len({alias, fused, lgc.FusedAlias(alias, 2)}) == 3
    assert eval(repr(fused), vars(lgc)) == fused
    assert str(fused) == "FusedAlias(scratch, 1)"
    with pytest.raises(NotImplementedError):
        fused.fields()
    match fused:
        case lgc.FusedAlias(inner, n):
            assert (inner, n) == (alias, 1)
        case _:
            pytest.fail("FusedAlias did not match")


def test_alias_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        lgc.Alias("scratch")


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_invalid_fused_dimension_count(n):
    with pytest.raises((ValueError, TypeError)):
        lgc.FusedAlias(lgc.HardAlias("scratch"), n)


def test_invalid_fused_alias():
    with pytest.raises(TypeError, match="wrap a HardAlias"):
        lgc.FusedAlias("scratch", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="unfused"):
        nested = lgc.FusedAlias(lgc.HardAlias("scratch"), 1)
        lgc.FusedAlias(nested, 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="table rank"):
        lgc.Table(lgc.FusedAlias(lgc.HardAlias("scratch"), 2), (lgc.Field("i"),))


@pytest.mark.parametrize(
    "shape,n",
    [((2, 3), 0), ((2, 3), 1), ((2, 3), 2), ((), 0), ((0, 3), 1), ((2, 0), 1)],
)
def test_fused_alias_interpreter(shape, n):
    data = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    a, scratch, out = map(lgc.HardAlias, ("a", "scratch", "out"))
    idxs = tuple(lgc.Field(f"i{d}") for d in range(len(shape)))
    fused = lgc.FusedAlias(scratch, n)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(fused, idxs), lgc.Table(a, idxs)),
            lgc.QueryInto(
                lgc.Table(fused, idxs), lgc.Literal(ffuncs.add), lgc.Table(a, idxs)
            ),
            lgc.Query(
                lgc.Table(out, idxs),
                lgc.MapJoin(
                    lgc.Literal(ffuncs.mul), (lgc.Table(fused, idxs), lgc.Literal(3))
                ),
            ),
            lgc.Produces((out, fused)),
        )
    )
    bindings = {a: ft.asarray(data)}
    result, mock = lgc.LogicInterpreter()(plan, bindings)
    np.testing.assert_array_equal(result.to_numpy(), data * 6)
    assert mock is bindings[scratch]
    assert isinstance(mock, MockFusedTensor)
    assert mock.shape == shape
    assert mock.inner_shape == shape[n:]
    expected_slots = np.prod(shape[:n], dtype=int) if data.size else 0
    assert len(mock.store_tns) == expected_slots
    for outer, tns in mock.store_tns.items():
        np.testing.assert_array_equal(tns.to_numpy(), data[outer] * 2)


def _redefine(old_view, new_view, read_view):
    i, j = lgc.Field("i"), lgc.Field("j")
    a = lgc.HardAlias("a")
    return lgc.Plan(
        (
            lgc.Query(lgc.Table(old_view, (i, j)), lgc.Table(a, (i, j))),
            lgc.Query(lgc.Table(new_view, (i, j)), lgc.Table(old_view, (i, j))),
            lgc.Produces((read_view,)),
        )
    )


_SCRATCH = lgc.HardAlias("scratch")
_VIEWS = [_SCRATCH, *(lgc.FusedAlias(_SCRATCH, n) for n in range(3))]


@pytest.mark.parametrize("old_view", _VIEWS)
@pytest.mark.parametrize("new_view", _VIEWS)
@pytest.mark.parametrize("read_view", _VIEWS)
def test_defining_a_view_invalidates_the_others(old_view, new_view, read_view):
    data = np.arange(6).reshape(2, 3)
    a = lgc.HardAlias("a")
    plan = _redefine(old_view, new_view, read_view)
    bindings = {a: ft.asarray(data)}
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ftype(bindings[a]),
        _SCRATCH: ftype(bindings[a]),
    }
    if read_view == new_view:
        AliasedForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
        (result,) = lgc.LogicInterpreter()(plan, bindings)
        match new_view:
            case lgc.FusedAlias(_, n):
                assert isinstance(result, MockFusedTensor)
                assert result.n == n
            case _:
                assert not isinstance(result, MockFusedTensor)
        np.testing.assert_array_equal(
            [[result[x, y] for y in range(3)] for x in range(2)], data
        )
    else:
        with pytest.raises(ValueError, match="invalidated"):
            AliasedForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
        with pytest.raises(ValueError, match="invalidated"):
            lgc.LogicInterpreter()(plan, bindings)


def test_updating_an_invalidated_view_is_rejected():
    i = lgc.Field("i")
    a = lgc.HardAlias("a")
    fused = lgc.FusedAlias(_SCRATCH, 1)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(fused, (i,)), lgc.Table(a, (i,))),
            lgc.QueryInto(
                lgc.Table(_SCRATCH, (i,)), lgc.Literal(ffuncs.add), lgc.Table(a, (i,))
            ),
            lgc.Produces((_SCRATCH,)),
        )
    )
    ftypes: dict[lgc.Alias, TensorFType] = {a: ftype(ft.asarray(np.ones(2)))}
    with pytest.raises(ValueError, match="invalidated"):
        AliasedForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_compiler_form_initialization_defines_a_view():
    i = lgc.Field("i")
    a = lgc.HardAlias("a")
    fused = lgc.FusedAlias(_SCRATCH, 1)
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ftype(ft.asarray(np.ones(2))),
        _SCRATCH: ftype(ft.asarray(np.ones(2))),
    }
    fold = lgc.QueryInto(
        lgc.Table(fused, (i,)),
        lgc.Literal(ffuncs.add),
        lgc.Reorder(lgc.Table(a, (i,)), (i,)),
    )
    init = lgc.QueryInto(
        lgc.Table(fused, (i,)), lgc.Literal(ffuncs.overwrite), lgc.Literal(0.0)
    )
    CompilerForm.validate_inputs(
        lgc.Plan((init, fold, lgc.Produces((fused,)))), ftypes, {}, DenseStatsFactory()
    )
    with pytest.raises(ValueError, match="invalidated"):
        CompilerForm.validate_inputs(
            lgc.Plan((fold, lgc.Produces((fused,)))), ftypes, {}, DenseStatsFactory()
        )
    with pytest.raises(ValueError, match="invalidated"):
        CompilerForm.validate_inputs(
            lgc.Plan((init, fold, lgc.Produces((_SCRATCH,)))),
            ftypes,
            {},
            DenseStatsFactory(),
        )


def test_fused_alias_inference_and_normalization():
    i, j = lgc.Field("i"), lgc.Field("j")
    a, scratch = lgc.HardAlias("a"), lgc.HardAlias("scratch")
    fused = lgc.FusedAlias(scratch, 1)
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(fused, (j, i)), lgc.Table(a, (i, j))),
            lgc.Produces((fused,)),
        )
    )
    assert plan.infer_shape({a: (2, 3)}) == {a: (2, 3), scratch: (3, 2)}
    assert fused.dimmap(max, {scratch: (3, 2)}) == (3, 2)
    assert plan.infer_element_type({a: ftype(np.int64)})[scratch] == ftype(np.int64)
    renamed, bindings = normalize_names(plan, {a: "input"})
    assert isinstance(renamed, lgc.Plan)
    query, produces = renamed.bodies
    assert isinstance(query, lgc.Query)
    assert isinstance(query.rhs, lgc.Table)
    assert isinstance(produces, lgc.Produces)
    lhs = query.lhs.tns
    assert isinstance(lhs, lgc.FusedAlias)
    assert lhs.n == 1
    assert produces.args == (lhs,)
    assert query.rhs.tns in bindings
    factory = DenseStatsFactory()
    stats = factory(ft.asarray(np.zeros((2, 3))), (i, j))
    results = StatsInterpreter(factory)(plan, {a: stats})
    assert isinstance(results, tuple)
    (result,) = results
    assert result.index_order == (j, i)


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_fused_alias_compilation(compiler, n):
    i, j, k = map(lgc.Field, ("i", "j", "k"))
    a, b, scratch, out = map(lgc.HardAlias, ("a", "b", "scratch", "out"))
    fused = lgc.FusedAlias(scratch, n)
    data_a = np.array([[1, 2, 0], [0, 3, 4]])
    data_b = np.array([[1, 0], [0, 1], [2, 2]])
    plan = lgc.Plan(
        (
            lgc.Query(
                lgc.Table(fused, (i, j)),
                lgc.Aggregate(
                    lgc.Literal(ffuncs.add),
                    lgc.Literal(0),
                    lgc.Reorder(
                        lgc.MapJoin(
                            lgc.Literal(ffuncs.mul),
                            (lgc.Table(a, (i, k)), lgc.Table(b, (k, j))),
                        ),
                        (i, k, j),
                    ),
                    (k,),
                ),
            ),
            lgc.Query(lgc.Table(out, (j, i)), lgc.Table(fused, (i, j))),
            lgc.Produces((out, fused)),
        )
    )
    bindings = {a: ft.asarray(data_a), b: ft.asarray(data_b)}
    capture = LogicCapture()
    DefaultLogicFormatter(CompilerFormLowerer(capture))(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )
    assert scratch in capture.last_bindings
    assert fused not in capture.last_bindings
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    bindings[scratch] = capture.last_bindings[scratch].construct((2, 2))
    bindings[out] = capture.last_bindings[out].construct((2, 2))
    result, intermediate = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), (data_a @ data_b).T)
    np.testing.assert_array_equal(intermediate.to_numpy(), data_a @ data_b)


@pytest.mark.parametrize(
    "op, error",
    [
        # Overwriting defines the fused view, after reading the hard one.
        (ffuncs.overwrite, "both read and write"),
        # Updating requires the fused view to be current already.
        (ffuncs.add, "invalidated"),
    ],
)
def test_fused_alias_cannot_hide_inplace_read(op, error):
    i = lgc.Field("i")
    a = lgc.HardAlias("a")
    fused = lgc.FusedAlias(a, 1)
    plan = lgc.Plan(
        (
            lgc.QueryInto(
                lgc.Table(fused, (i,)),
                lgc.Literal(op),
                lgc.Reorder(lgc.Table(a, (i,)), (i,)),
            ),
            lgc.Produces((fused,)),
        )
    )
    with pytest.raises(ValueError, match=error):
        CompilerForm.validate_inputs(
            plan, {a: ftype(ft.asarray(np.ones(2)))}, {}, DenseStatsFactory()
        )


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_compiling_a_redefined_view(compiler, n):
    i, j = lgc.Field("i"), lgc.Field("j")
    a, scratch, out, out2 = map(lgc.HardAlias, ("a", "scratch", "out", "out2"))
    fused = lgc.FusedAlias(scratch, n)
    data = np.array([[1, 2, 0], [0, 3, 4]])
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(a, (i, j))),
            lgc.Query(lgc.Table(out, (j, i)), lgc.Table(scratch, (i, j))),
            lgc.Query(
                lgc.Table(fused, (i, j)),
                lgc.Aggregate(
                    lgc.Literal(ffuncs.add),
                    lgc.Literal(0),
                    lgc.Reorder(
                        lgc.MapJoin(
                            lgc.Literal(ffuncs.mul),
                            (lgc.Table(a, (i, j)), lgc.Literal(2)),
                        ),
                        (i, j),
                    ),
                    (),
                ),
            ),
            lgc.Query(lgc.Table(out2, (j, i)), lgc.Table(fused, (i, j))),
            lgc.Produces((out, out2)),
        )
    )
    bindings = {a: ft.asarray(data)}
    capture = LogicCapture()
    DefaultLogicFormatter(CompilerFormLowerer(capture))(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )
    assert isinstance(capture.last_prgm, lgc.Plan)
    CompilerForm.validate_inputs(
        capture.last_prgm, capture.last_bindings, {}, DenseStatsFactory()
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    for var, shape in ((scratch, (2, 3)), (out, (3, 2)), (out2, (3, 2))):
        bindings[var] = capture.last_bindings[var].construct(shape)
    result, result2 = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), data.T)
    np.testing.assert_array_equal(result2.to_numpy(), 2 * data.T)
