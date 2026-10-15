import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import DynamicFill, TensorFType, ffuncs, ftype
from finch.autoschedule import CompilerFormLowerer, LogicCapture, NotationGenerator
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.normalize import normalize_names
from finch.autoschedule.stages import AliasedForm, CompilerForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsInterpreter
from finch.autoschedule.util import fuse_loop_order
from finch.codegen import MLIRCompiler
from finch.compile import CompilerMode, NotationCompiler
from finch.finch_logic.interpreter import MockFusedTensor
from finch.finch_notation.nodes import NotationTree
from finch.tensor import BufferizedNDArray


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
        lgc.Alias("scratch")  # ty: ignore[call-non-callable]


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
    for idx in reversed(idxs[:n]):
        plan = lgc.Fuse(idx, plan)
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


def test_compiler_form_definition_defines_a_view():
    i = lgc.Field("i")
    a = lgc.HardAlias("a")
    # A view without fused fields is read after its Fuse, like any tensor.
    fused = lgc.FusedAlias(_SCRATCH, 0)
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ftype(ft.asarray(np.ones(2))),
        _SCRATCH: ftype(ft.asarray(np.ones(2))),
    }
    fold = fuse_loop_order(
        lgc.QueryInto(
            lgc.Table(fused, (i,)),
            lgc.Literal(ffuncs.add),
            lgc.Table(a, (i,)),
        ),
        (i,),
    )
    init = fuse_loop_order(lgc.Query(lgc.Table(fused, (i,)), lgc.Table(a, (i,))), (i,))
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


def within_fused_fields(stmts, fields):
    # A fused alias stores a slice for each value of its fused fields, so its
    # queries share the Fuse nodes of those fields.
    if not fields:
        return stmts
    return (fuse_loop_order(lgc.Plan(stmts), fields),)


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_fused_alias_compilation(compiler, n):
    i, j, k = map(lgc.Field, ("i", "j", "k"))
    a, b, scratch, out = map(lgc.HardAlias, ("a", "b", "scratch", "out"))
    fused = lgc.FusedAlias(scratch, n)
    materialized = lgc.HardAlias("materialized")
    data_a = np.array([[1, 2, 0], [0, 3, 4]])
    data_b = np.array([[1, 0], [0, 1], [2, 2]])
    define = fuse_loop_order(
        lgc.Query(
            lgc.Table(fused, (i, j)),
            lgc.Aggregate(
                lgc.Literal(ffuncs.add),
                lgc.Literal(0),
                lgc.MapJoin(
                    lgc.Literal(ffuncs.mul),
                    (lgc.Table(a, (i, k)), lgc.Table(b, (j, k))),
                ),
                (k,),
            ),
        ),
        (i, j, k)[n:],
    )
    materialize = fuse_loop_order(
        lgc.Query(lgc.Table(materialized, (i, j)), lgc.Table(fused, (i, j))),
        (i, j)[n:],
    )
    plan = lgc.Plan(
        (
            *within_fused_fields((define, materialize), (i, j)[:n]),
            fuse_loop_order(
                lgc.Query(lgc.Table(out, (j, i)), lgc.Table(materialized, (i, j))),
                (j, i),
            ),
            lgc.Produces((out, materialized)),
        )
    )
    bindings = {a: ft.asarray(data_a), b: ft.asarray(data_b.T)}
    capture = LogicCapture()
    DefaultLogicFormatter(CompilerFormLowerer(capture))(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )
    assert scratch in capture.last_bindings
    assert fused not in capture.last_bindings
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    bindings[scratch] = capture.last_bindings[scratch].construct((2, 2))
    bindings[materialized] = capture.last_bindings[materialized].construct((2, 2))
    bindings[out] = capture.last_bindings[out].construct((2, 2))
    result, intermediate = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), (data_a @ data_b).T)
    np.testing.assert_array_equal(intermediate.to_numpy(), data_a @ data_b)


@pytest.mark.parametrize(
    "op, error",
    [
        # A definition defines the fused view, after reading the hard one.
        (None, "cannot occur on both lhs and rhs"),
        # Updating requires the fused view to be current already, even when
        # every element is overwritten.
        (ffuncs.overwrite, "invalidated"),
        (ffuncs.add, "invalidated"),
    ],
)
def test_fused_alias_cannot_hide_inplace_read(op, error):
    i = lgc.Field("i")
    a = lgc.HardAlias("a")
    fused = lgc.FusedAlias(a, 1)
    stmt = (
        fuse_loop_order(lgc.Query(lgc.Table(fused, (i,)), lgc.Table(a, (i,))), (i,))
        if op is None
        else fuse_loop_order(
            lgc.QueryInto(
                lgc.Table(fused, (i,)),
                lgc.Literal(op),
                lgc.Table(a, (i,)),
            ),
            (i,),
        )
    )
    plan = lgc.Plan((stmt, lgc.Produces((fused,))))
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
    materialized = lgc.HardAlias("materialized")
    data = np.array([[1, 2, 0], [0, 3, 4]])
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(a, (i, j))), (i, j)
            ),
            fuse_loop_order(
                lgc.Query(lgc.Table(out, (j, i)), lgc.Table(scratch, (i, j))), (j, i)
            ),
            *within_fused_fields(
                (
                    fuse_loop_order(
                        lgc.Query(
                            lgc.Table(fused, (i, j)),
                            lgc.Aggregate(
                                lgc.Literal(ffuncs.add),
                                lgc.Literal(0),
                                lgc.MapJoin(
                                    lgc.Literal(ffuncs.mul),
                                    (lgc.Table(a, (i, j)), lgc.Literal(2)),
                                ),
                                (),
                            ),
                        ),
                        (i, j)[n:],
                    ),
                    fuse_loop_order(
                        lgc.Query(
                            lgc.Table(materialized, (i, j)), lgc.Table(fused, (i, j))
                        ),
                        (i, j)[n:],
                    ),
                ),
                (i, j)[:n],
            ),
            fuse_loop_order(
                lgc.Query(lgc.Table(out2, (j, i)), lgc.Table(materialized, (i, j))),
                (j, i),
            ),
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
    for var, shape in (
        (scratch, (2, 3)),
        (out, (3, 2)),
        (materialized, (2, 3)),
        (out2, (3, 2)),
    ):
        bindings[var] = capture.last_bindings[var].construct(shape)
    result, result2 = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), data.T)
    np.testing.assert_array_equal(result2.to_numpy(), 2 * data.T)


def gustavson_plan(a, b, w, c):
    # C[i, k] = sum_j A[i, j] * B[j, k], one row of C at a time.
    i, j, k = map(lgc.Field, "ijk")
    fused = lgc.FusedAlias(w, 1)
    return lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Plan(
                    (
                        fuse_loop_order(
                            lgc.Query(
                                lgc.Table(fused, (i, k)),
                                lgc.Aggregate(
                                    lgc.Literal(ffuncs.add),
                                    lgc.Literal(0),
                                    lgc.MapJoin(
                                        lgc.Literal(ffuncs.mul),
                                        (lgc.Table(a, (i, j)), lgc.Table(b, (j, k))),
                                    ),
                                    (j,),
                                ),
                            ),
                            (j, k),
                        ),
                        fuse_loop_order(
                            lgc.Query(lgc.Table(c, (i, k)), lgc.Table(fused, (i, k))),
                            (k,),
                        ),
                    )
                ),
            ),
            lgc.Produces((c,)),
        )
    )


def declare_depths(program):
    """The number of loops around each declaration in `program`, by tensor."""
    depths: dict[str, list[int]] = {}

    def visit(node, depth):
        match node:
            case ntn.Declare(ntn.Slot(name, _), _, _, _):
                depths.setdefault(name.strip("#_").split("#")[0], []).append(depth)
            case ntn.Loop(_, _, body):
                visit(body, depth + 1)
            case NotationTree():
                for child in node.children:
                    visit(child, depth)

    visit(program, 0)
    return depths


@pytest.mark.parametrize(
    "compiler",
    [
        ntn.NotationInterpreter,
        NotationCompiler,
        pytest.param(
            lambda: NotationCompiler(MLIRCompiler()), marks=pytest.mark.mlir_backend
        ),
    ],
)
def test_fuse_compilation_uses_a_workspace(compiler):
    # The fused alias stores only its trailing field, as the formatters which
    # format stored dimensions choose, so each row reuses one workspace.
    a, b, w, c = map(lgc.HardAlias, "abwc")
    data_a = np.array([[1, 2, 0], [0, 3, 4]])
    data_b = np.array([[1, 0], [0, 1], [2, 2]])
    bindings = {
        a: ft.asarray(data_a),
        b: ft.asarray(data_b),
        w: ft.asarray(np.zeros(0, dtype=np.int64)),
        c: ft.asarray(np.zeros((0, 0), dtype=np.int64)),
    }
    plan = gustavson_plan(a, b, w, c)
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: val.ftype for var, val in bindings.items()
    }
    CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    program = NotationGenerator()(plan, ftypes, {}, None)
    # The workspace is reset for each row, and the output only once.
    assert declare_depths(program) == {"w": [1], "c": [0]}
    (result,) = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), data_a @ data_b)
    assert bindings[w].shape == (2,)


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
def test_fuse_compilation_with_full_storage(compiler):
    # The default formatter stores every field of the fused alias, so it spans
    # the fused iterations like any other output.
    a, b, w, c = map(lgc.HardAlias, "abwc")
    data_a = np.array([[1, 2, 0], [0, 3, 4]])
    data_b = np.array([[1, 0], [0, 1], [2, 2]])
    bindings = {a: ft.asarray(data_a), b: ft.asarray(data_b)}
    capture = LogicCapture()
    DefaultLogicFormatter(CompilerFormLowerer(capture))(
        gustavson_plan(a, b, w, c),
        {var: ftype(val) for var, val in bindings.items()},
        {},
        None,
    )
    assert capture.last_bindings[w].ndim == 2
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    for var in (w, c):
        bindings[var] = capture.last_bindings[var].construct((0, 0))
    (result,) = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), data_a @ data_b)


@pytest.mark.parametrize(
    "compiler",
    [
        ntn.NotationInterpreter,
        NotationCompiler,
        pytest.param(
            lambda: NotationCompiler(MLIRCompiler()), marks=pytest.mark.mlir_backend
        ),
    ],
)
def test_nested_fuse_with_scalar_workspaces(compiler):
    i, j = lgc.Field("i"), lgc.Field("j")
    a, w, z, c, d = map(lgc.HardAlias, "awzcd")
    fw, fz = lgc.FusedAlias(w, 2), lgc.FusedAlias(z, 2)

    def scaled(factor):
        return lgc.Aggregate(
            lgc.Literal(ffuncs.overwrite),
            lgc.Literal(0.0),
            lgc.MapJoin(
                lgc.Literal(ffuncs.mul), (lgc.Table(a, (i, j)), lgc.Literal(factor))
            ),
            (),
        )

    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Fuse(
                    j,
                    lgc.Plan(
                        (
                            lgc.Query(lgc.Table(fw, (i, j)), scaled(2.0)),
                            lgc.Query(lgc.Table(fz, (i, j)), scaled(3.0)),
                            lgc.Query(lgc.Table(c, (i, j)), lgc.Table(fw, (i, j))),
                            lgc.Query(lgc.Table(d, (i, j)), lgc.Table(fz, (i, j))),
                        )
                    ),
                ),
            ),
            lgc.Produces((c, d)),
        )
    )
    data = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
    bindings = {
        a: ft.asarray(data),
        w: ft.asarray(np.array(0.0)),
        z: ft.asarray(np.array(0.0)),
        c: ft.asarray(np.zeros((0, 0))),
        d: ft.asarray(np.zeros((0, 0))),
    }
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: val.ftype for var, val in bindings.items()
    }
    CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    program = NotationGenerator()(plan, ftypes, {}, None)
    result_c, result_d = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result_c.to_numpy(), 2 * data)
    np.testing.assert_array_equal(result_d.to_numpy(), 3 * data)


def test_workspace_needs_its_fuse():
    # Without the Fuse, a workspace can't hold every row of the fused alias.
    a, b, w, c = map(lgc.HardAlias, "abwc")
    plan = gustavson_plan(a, b, w, c)
    (fuse, produces) = plan.bodies
    unfused = lgc.Plan((*fuse.body.bodies, produces))
    bindings = {
        a: ft.asarray(np.ones((2, 3))),
        b: ft.asarray(np.ones((3, 2))),
        w: ft.asarray(np.zeros(0)),
        c: ft.asarray(np.zeros((0, 0))),
    }
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: val.ftype for var, val in bindings.items()
    }
    with pytest.raises(ValueError, match="must match the enclosing Fuse fields"):
        NotationGenerator()(unfused, ftypes, {}, None)


def test_workspace_cant_be_read_after_its_fuse():
    i, j = lgc.Field("i"), lgc.Field("j")
    a, w, c = map(lgc.HardAlias, "awc")
    fused = lgc.FusedAlias(w, 1)
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i, j))), (i, j)
            ),
            fuse_loop_order(
                lgc.Query(lgc.Table(c, (i, j)), lgc.Table(fused, (i, j))), (i, j)
            ),
            lgc.Produces((c,)),
        )
    )
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ftype(ft.asarray(np.ones((2, 2)))),
        # The workspace stores only the field after its fused one.
        w: ftype(ft.asarray(np.ones(2))),
        c: ftype(ft.asarray(np.ones((2, 2)))),
    }
    with pytest.raises(ValueError, match="can't be read outside the Fuse nodes"):
        AliasedForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    # The fused alias says which fields it stores, whatever its format.
    ftypes[w] = ftype(ft.asarray(np.ones((2, 2))))
    with pytest.raises(ValueError, match="can't be read outside the Fuse nodes"):
        AliasedForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_hoisted_reset_cant_read_a_fill_its_fuse_defines():
    # The reset of c runs before the loop over i, but the dynamic fill of the
    # workspace it copies is only set within that loop.
    i, k = lgc.Field("i"), lgc.Field("k")
    a, w, c = map(lgc.HardAlias, "awc")
    fused = lgc.FusedAlias(w, 1)
    increment = lgc.Aggregate(
        lgc.Literal(ffuncs.overwrite),
        lgc.FillTable(a),
        lgc.MapJoin(lgc.Literal(ffuncs.add), (lgc.Table(a, (i, k)), lgc.Literal(1.0))),
        (),
    )
    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Plan(
                    (
                        lgc.Fuse(k, lgc.Query(lgc.Table(fused, (i, k)), increment)),
                        lgc.Fuse(
                            k, lgc.Query(lgc.Table(c, (i, k)), lgc.Table(fused, (i, k)))
                        ),
                    )
                ),
            ),
            lgc.Produces((c,)),
        )
    )

    def dynamic(arr):
        fill = DynamicFill(np.float64(0.0))
        return BufferizedNDArray.from_numpy(arr, fill_value=fill).ftype

    ftypes: dict[lgc.Alias, TensorFType] = {
        a: dynamic(np.ones((2, 2))),
        w: dynamic(np.zeros(0)),
        c: dynamic(np.zeros((0, 0))),
    }
    with pytest.raises(ValueError, match="can't read the fill of w"):
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


def test_transposed_read_needs_its_own_fuse():
    # A transposed read revisits a field of its Fuse nodes, which would run the
    # other statements of their bodies again.
    i, j = lgc.Field("i"), lgc.Field("j")
    a, b, x, y = map(lgc.HardAlias, "abxy")
    body = lgc.Plan(
        (
            lgc.Query(lgc.Table(x, (i, j)), lgc.Table(a, (j, i))),
            lgc.Query(lgc.Table(y, (i, j)), lgc.Table(b, (i, j))),
        )
    )
    plan = lgc.Plan((fuse_loop_order(body, (i, j)), lgc.Produces((x, y))))
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: ftype(ft.asarray(np.ones((2, 2)))) for var in (a, b, x, y)
    }
    with pytest.raises(ValueError, match="out of loop order"):
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


@pytest.mark.parametrize("size", [2, 3])
def test_fused_statements_check_their_extents(size):
    i = lgc.Field("i")
    a, b, c, d = map(lgc.HardAlias, "abcd")
    body = lgc.Plan(
        (
            lgc.Query(lgc.Table(c, (i,)), lgc.Table(a, (i,))),
            lgc.Query(lgc.Table(d, (i,)), lgc.Table(b, (i,))),
        )
    )
    plan = lgc.Plan((lgc.Fuse(i, body), lgc.Produces((c, d))))
    bindings = {
        a: ft.asarray(np.arange(2.0)),
        b: ft.asarray(np.arange(size, dtype=np.float64)),
        c: ft.asarray(np.zeros(0)),
        d: ft.asarray(np.zeros(0)),
    }
    ftypes = {var: val.ftype for var, val in bindings.items()}
    program = NotationGenerator(mode=CompilerMode(safe=True))(plan, ftypes, {}, None)
    kernel = ntn.NotationInterpreter()(program).main
    if size != 2:
        with pytest.raises(AssertionError, match="Finch assertion failed"):
            kernel(*bindings.values())
        return
    result_c, result_d = kernel(*bindings.values())
    np.testing.assert_array_equal(result_c.to_numpy(), np.arange(2.0))
    np.testing.assert_array_equal(result_d.to_numpy(), np.arange(2.0))


def test_compiler_form_checks_fuse_bodies():
    # A Fuse can't both read and write a tensor along its field.
    i = lgc.Field("i")
    a, b = lgc.HardAlias("a"), lgc.HardAlias("b")
    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Plan(
                    (
                        lgc.Query(lgc.Table(b, (i,)), lgc.Table(a, (i,))),
                        lgc.Query(lgc.Table(a, (i,)), lgc.Table(b, (i,))),
                    )
                ),
            ),
            lgc.Produces((a,)),
        )
    )
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: ft.asarray(np.ones(2)).ftype for var in (a, b)
    }
    with pytest.raises(ValueError, match="cannot occur on both lhs and rhs"):
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
def test_fuse_nodes_give_the_loop_order(compiler):
    # Without Reorders, the Fuse nodes around each query are its loop order.
    i, j, k = map(lgc.Field, "ijk")
    a, b, w, c = map(lgc.HardAlias, "abwc")
    fused = lgc.FusedAlias(w, 1)
    product = lgc.MapJoin(
        lgc.Literal(ffuncs.mul), (lgc.Table(a, (i, j)), lgc.Table(b, (j, k)))
    )
    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Plan(
                    (
                        lgc.Fuse(
                            j,
                            lgc.Fuse(
                                k,
                                lgc.Query(
                                    lgc.Table(fused, (i, k)),
                                    lgc.Aggregate(
                                        lgc.Literal(ffuncs.add),
                                        lgc.Literal(0),
                                        product,
                                        (j,),
                                    ),
                                ),
                            ),
                        ),
                        lgc.Fuse(
                            k, lgc.Query(lgc.Table(c, (i, k)), lgc.Table(fused, (i, k)))
                        ),
                    )
                ),
            ),
            lgc.Produces((c,)),
        )
    )
    data_a = np.array([[1, 2, 0], [0, 3, 4]])
    data_b = np.array([[1, 0], [0, 1], [2, 2]])
    bindings = {
        a: ft.asarray(data_a),
        b: ft.asarray(data_b),
        w: ft.asarray(np.zeros(0, dtype=np.int64)),
        c: ft.asarray(np.zeros((0, 0), dtype=np.int64)),
    }
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: val.ftype for var, val in bindings.items()
    }
    CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    program = NotationGenerator()(plan, ftypes, {}, None)
    # The workspace is reset in each row, before the loop which reduces j.
    assert declare_depths(program) == {"w": [1], "c": [0]}
    (result,) = compiler()(program).main(*bindings.values())
    np.testing.assert_array_equal(result.to_numpy(), data_a @ data_b)


@pytest.mark.parametrize(
    "compiler",
    [
        ntn.NotationInterpreter,
        NotationCompiler,
        pytest.param(
            lambda: NotationCompiler(MLIRCompiler()), marks=pytest.mark.mlir_backend
        ),
    ],
)
@pytest.mark.parametrize("columns", [2, 0])
def test_fused_overwrite_resets_once(compiler, columns):
    # Overwriting with a reduction resets the output before the fused loops,
    # unless the reduction is empty and leaves the output as it was.
    i, j = lgc.Field("i"), lgc.Field("j")
    a, d = lgc.HardAlias("a"), lgc.HardAlias("d")
    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Fuse(
                    j,
                    lgc.QueryInto(
                        lgc.Table(d, (i,)),
                        lgc.Literal(ffuncs.overwrite),
                        lgc.Table(a, (i, j)),
                    ),
                ),
            ),
            lgc.Produces((d,)),
        )
    )
    data = np.arange(2 * columns, dtype=np.float64).reshape((2, columns))
    bindings = {a: ft.asarray(data), d: ft.asarray(np.full(2, 7.0))}
    ftypes: dict[lgc.Alias, TensorFType] = {
        var: val.ftype for var, val in bindings.items()
    }
    CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
    program = NotationGenerator()(plan, ftypes, {}, None)
    (result,) = compiler()(program).main(*bindings.values())
    expected = data[:, -1] if columns else np.full(2, 7.0)
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_queries_without_loop_orders_fuse_every_field():
    i, j = lgc.Field("i"), lgc.Field("j")
    a, d = lgc.HardAlias("a"), lgc.HardAlias("d")
    plan = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.QueryInto(
                    lgc.Table(d, (i,)), lgc.Literal(ffuncs.add), lgc.Table(a, (i, j))
                ),
            ),
            lgc.Produces((d,)),
        )
    )
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ft.asarray(np.ones((2, 2))).ftype,
        d: ft.asarray(np.ones(2)).ftype,
    }
    with pytest.raises(ValueError, match="need Fuse nodes"):
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())
