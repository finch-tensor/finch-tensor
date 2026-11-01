import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import TensorFType, ffuncs, ftype
from finch.autoschedule import CompilerFormLowerer, LogicCapture, NotationGenerator
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.normalize import normalize_names
from finch.autoschedule.stages import CompilerForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsInterpreter
from finch.autoschedule.util import fuse_loop_order
from finch.compile import NotationCompiler
from finch.symbolic import PostOrderDFS

from .utils import with_workspaces


def test_protocol_alias_node():
    a = lgc.HardAlias("a")
    protocols = (lgc.DefaultProtocol(), lgc.Walk())
    view = lgc.ProtocolAlias(a, protocols)
    assert view.alias == a
    assert view.name == a.name
    assert view.unfused == a
    assert view.nfused == 0
    assert view == lgc.ProtocolAlias(a, protocols)
    assert view != lgc.ProtocolAlias(a, (lgc.Follow(), lgc.Walk()))
    assert view != a
    assert eval(repr(view), vars(lgc)) == view
    assert str(view) == "ProtocolAlias(a, (default, walk))"
    i, j = lgc.Field("i"), lgc.Field("j")
    assert str(lgc.Table(view, (i, j))) == "a[i, walk(j)]"
    fused = lgc.ProtocolAlias(lgc.FusedAlias(a, 1), (lgc.Gallop(), lgc.Follow()))
    assert fused.unfused == a
    assert fused.nfused == 1
    assert str(lgc.Table(fused, (i, j))) == "a(gallop(i))[follow(j)]"


def test_invalid_protocol_alias():
    a = lgc.HardAlias("a")
    i = lgc.Field("i")
    with pytest.raises(TypeError, match="wrap a HardAlias or FusedAlias"):
        lgc.ProtocolAlias("a", (lgc.Walk(),))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="without protocols"):
        lgc.ProtocolAlias(
            lgc.ProtocolAlias(a, (lgc.Walk(),)),  # ty: ignore[invalid-argument-type]
            (lgc.Walk(),),
        )
    with pytest.raises(ValueError, match="one protocol for each field"):
        lgc.Table(lgc.ProtocolAlias(a, (lgc.Walk(), lgc.Walk())), (i,))
    with pytest.raises(ValueError, match="table rank"):
        lgc.Table(lgc.ProtocolAlias(lgc.FusedAlias(a, 2), (lgc.Walk(),)), (i,))


def test_protocol_alias_normalization_and_stats():
    i, j = lgc.Field("i"), lgc.Field("j")
    a, out = lgc.HardAlias("a"), lgc.HardAlias("out")
    protocols = (lgc.DefaultProtocol(), lgc.Walk())
    plan = lgc.Plan(
        (
            lgc.Query(
                lgc.Table(out, (i, j)),
                lgc.Table(lgc.ProtocolAlias(a, protocols), (i, j)),
            ),
            lgc.Produces((out,)),
        )
    )
    renamed, bindings = normalize_names(plan, {a: "input"})
    assert isinstance(renamed, lgc.Plan)
    query = renamed.bodies[0]
    assert isinstance(query, lgc.Query)
    assert isinstance(query.rhs, lgc.Table)
    view = query.rhs.tns
    assert isinstance(view, lgc.ProtocolAlias)
    assert view.protocols == protocols
    assert view.alias in bindings
    factory = DenseStatsFactory()
    stats = factory(ft.asarray(np.ones((2, 3))), (i, j))
    results = StatsInterpreter(factory)(plan, {a: stats})
    assert isinstance(results, tuple)
    (result,) = results
    assert result.index_order == (i, j)


def test_protocols_only_apply_to_reads():
    i = lgc.Field("i")
    a, out = lgc.HardAlias("a"), lgc.HardAlias("out")
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ftype(ft.asarray(np.ones(2))),
        out: ftype(ft.asarray(np.ones(2))),
    }

    def validate(lhs, rhs):
        query = fuse_loop_order(
            lgc.Query(lgc.Table(lhs, (i,)), lgc.Table(rhs, (i,))), (i,)
        )
        plan = lgc.Plan((query, lgc.Produces((out,))))
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())

    validate(out, lgc.ProtocolAlias(a, (lgc.Walk(),)))
    with pytest.raises(ValueError, match="only apply to reads"):
        validate(lgc.ProtocolAlias(out, (lgc.Walk(),)), a)


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize(
    "protocol, lowered",
    [
        (lgc.Walk(), ntn.Walk()),
        (lgc.Gallop(), ntn.Gallop()),
        (lgc.Follow(), ntn.Follow()),
    ],
)
def test_protocol_lowering(compiler, protocol, lowered):
    i, j = lgc.Field("i"), lgc.Field("j")
    a, b, out = lgc.HardAlias("a"), lgc.HardAlias("b"), lgc.HardAlias("out")
    data_a = np.array([[1, 0, 2], [0, 3, 0]])
    data_b = np.array([1, 0, 7])
    product = lgc.MapJoin(
        lgc.Literal(ffuncs.mul),
        (
            lgc.Table(lgc.ProtocolAlias(a, (lgc.DefaultProtocol(), protocol)), (i, j)),
            lgc.Table(lgc.ProtocolAlias(b, (lgc.Follow(),)), (j,)),
        ),
    )
    query = lgc.Query(
        lgc.Table(out, (i,)),
        lgc.Aggregate(lgc.Literal(ffuncs.add), lgc.Literal(0), product, (j,)),
    )
    plan = lgc.Plan((fuse_loop_order(query, (i, j)), lgc.Produces((out,))))
    bindings = {a: ft.asarray(data_a), b: ft.asarray(data_b)}
    capture = LogicCapture()
    DefaultLogicFormatter(CompilerFormLowerer(capture))(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    reads = [
        node.mode
        for node in PostOrderDFS(program)
        if isinstance(node, ntn.Access) and isinstance(node.mode, ntn.Read)
    ]
    assert ntn.Read((ntn.DefaultProtocol(), lowered)) in reads
    assert ntn.Read((ntn.Follow(),)) in reads
    assert f"{lowered}(" in str(program)
    bindings[out] = capture.last_bindings[out].construct((2,))
    (result,) = compiler()(program).main(*with_workspaces(program, *bindings.values()))
    np.testing.assert_array_equal(result.to_numpy(), data_a @ data_b)
