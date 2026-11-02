import itertools

import pytest

import numpy as np

import finch as ft
import finch.finch_logic as lgc
from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import TensorFType, ffuncs, ftype
from finch.autoschedule import CompilerFormLowerer, LogicCapture, NotationGenerator
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer.loop_ordering import concordize
from finch.autoschedule.normalize import normalize_names
from finch.autoschedule.stages import CompilerForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsInterpreter
from finch.autoschedule.util import fuse_loop_order, push_fields
from finch.codegen import CCompiler, MLIRCompiler, NumbaCompiler
from finch.compile import CompilerMode, NotationCompiler, make_extent
from finch.symbolic import PostOrderDFS

from .test_sparse_update import dense_values, run_copy, tensors
from .utils import with_workspaces


def test_table_protocols():
    a = lgc.HardAlias("a")
    i, j = lgc.Field("i"), lgc.Field("j")
    protocols = (lgc.DefaultProtocol(), lgc.Walk())
    table = lgc.Table(a, (i, j), protocols)
    assert lgc.Table(a, (i, j)).protocols == ()
    assert table != lgc.Table(a, (i, j))
    assert eval(repr(table), vars(lgc)) == table
    assert str(table) == "a[i, walk(j)]"
    # Rewrites rebuild a table from its children, protocols included.
    assert table.make_term(table.head(), *table.children) == table
    fused = lgc.Table(lgc.FusedAlias(a, 1), (i, j), (lgc.Gallop(), lgc.Follow()))
    assert str(fused) == "a(gallop(i))[follow(j)]"
    with pytest.raises(ValueError, match="one protocol for each field"):
        lgc.Table(a, (i,), (lgc.Walk(), lgc.Walk()))


def test_table_protocols_survive_normalization_and_stats():
    i, j = lgc.Field("i"), lgc.Field("j")
    a, out = lgc.HardAlias("a"), lgc.HardAlias("out")
    protocols = (lgc.DefaultProtocol(), lgc.Walk())
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(out, (i, j)), lgc.Table(a, (i, j), protocols)),
            lgc.Produces((out,)),
        )
    )
    renamed, bindings = normalize_names(plan, {a: "input"})
    assert isinstance(renamed, lgc.Plan)
    query = renamed.bodies[0]
    assert isinstance(query, lgc.Query)
    assert isinstance(query.rhs, lgc.Table)
    assert query.rhs.protocols == protocols
    assert query.rhs.tns in bindings
    factory = DenseStatsFactory()
    stats = factory(ft.asarray(np.ones((2, 3))), (i, j))
    results = StatsInterpreter(factory)(plan, {a: stats})
    assert isinstance(results, tuple)
    (result,) = results
    assert result.index_order == (i, j)


def test_rewrites_keep_table_protocols():
    i, j, k, m = map(lgc.Field, "ijkm")
    a, b = lgc.HardAlias("a"), lgc.HardAlias("b")
    protocols = (lgc.Walk(), lgc.Follow())
    # Relabeling renames the fields of a table, which keep their protocols.
    relabeled = push_fields(lgc.Relabel(lgc.Table(a, (i, j), protocols), (k, m)))
    assert relabeled == lgc.Table(a, (k, m), protocols)
    # A swizzled read permutes the protocols along with its fields.
    read = lgc.Aggregate(
        lgc.Literal(ffuncs.overwrite),
        lgc.Literal(0),
        lgc.Table(a, (i, j), protocols),
        (),
    )
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(a, (i, j)), lgc.Table(lgc.Literal(0), (i, j))),
            fuse_loop_order(lgc.Query(lgc.Table(b, (j, i)), read), (j, i)),
            lgc.Produces((b,)),
        )
    )
    swizzled = [
        node.protocols
        for node in PostOrderDFS(concordize(plan, bindings={}))
        if isinstance(node, lgc.Table) and node.protocols
    ]
    assert swizzled == [(lgc.Follow(), lgc.Walk())]


def test_protocols_only_apply_to_reads():
    i = lgc.Field("i")
    a, out = lgc.HardAlias("a"), lgc.HardAlias("out")
    walk = (lgc.Walk(),)
    ftypes: dict[lgc.Alias, TensorFType] = {
        a: ftype(ft.asarray(np.ones(2))),
        out: ftype(ft.asarray(np.ones(2))),
    }

    def validate(lhs_protocols, rhs_protocols):
        query = lgc.Query(
            lgc.Table(out, (i,), lhs_protocols), lgc.Table(a, (i,), rhs_protocols)
        )
        plan = lgc.Plan((fuse_loop_order(query, (i,)), lgc.Produces((out,))))
        CompilerForm.validate_inputs(plan, ftypes, {}, DenseStatsFactory())

    validate((), walk)
    with pytest.raises(ValueError, match="only apply to reads"):
        validate(walk, ())


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
            lgc.Table(a, (i, j), (lgc.DefaultProtocol(), protocol)),
            lgc.Table(b, (j,), (lgc.Follow(),)),
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


def coiterate_program(x, y, z, op, protocols):
    """z = op(x, y), reading x and y with their protocols."""
    args = tuple(
        ntn.Variable(f"{name}_tns", tns.ftype)
        for name, tns in zip("xyz", (x, y, z), strict=True)
    )
    slots = tuple(ntn.Slot(f"{arg.name}_slot", arg.result_type) for arg in args)
    idxs = tuple(ntn.Variable(f"i{d}", ftype(np.intp)) for d in range(z.ndim))
    reads = tuple(
        ntn.Unwrap(ntn.Access(slot, ntn.Read(protos), idxs))
        for slot, protos in zip(slots[:2], protocols, strict=True)
    )
    overwrite = ntn.Literal(ffuncs.overwrite)
    body: ntn.NotationStatement = ntn.Increment(
        ntn.Access(slots[2], ntn.Update(overwrite), idxs),
        ntn.Call(ntn.Literal(op), reads),
    )
    for d in reversed(range(z.ndim)):
        ext = (ntn.Literal(np.intp(0)), ntn.Literal(np.intp(z.shape[d])))
        body = ntn.Loop(idxs[d], ntn.Call(ntn.Literal(make_extent), ext), body)
    stmts = (
        *map(ntn.Unpack, slots, args),
        ntn.Declare(
            slots[2],
            ntn.Literal(z.fill_value),
            overwrite,
            tuple(ntn.Literal(dim) for dim in z.shape),
        ),
        body,
        ntn.Freeze(slots[2], overwrite),
        *map(ntn.Repack, slots, args),
        ntn.Return(args[2]),
    )
    kernel_t = asm.AssemblyKernelFType(
        "coiterate", tuple(arg.result_type for arg in args), z.ftype
    )
    return ntn.Module(
        (ntn.Function(ntn.Variable("coiterate", kernel_t), args, ntn.Block(stmts)),)
    )


@pytest.fixture(
    params=[
        asm.AssemblyInterpreter,
        NumbaCompiler,
        pytest.param(CCompiler, marks=pytest.mark.c_backend),
        pytest.param(MLIRCompiler, marks=pytest.mark.mlir_backend),
    ]
)
def loader(request):
    return request.param()


def stored(loader, format_, data):
    source, target = tensors(format_, data)
    return run_copy(loader, source, target)


PROTOCOLS = {
    "D": (ntn.DefaultProtocol(), ntn.Follow()),
    "S": (ntn.Walk(), ntn.Gallop(), ntn.Follow()),
    "H": (ntn.Walk(), ntn.Follow()),
    "B": (ntn.Walk(), ntn.Gallop(), ntn.Follow()),
}


@pytest.mark.parametrize("op", [ffuncs.mul, ffuncs.add])
@pytest.mark.parametrize(
    "x_format, y_format", [("S", "S"), ("S", "D"), ("H", "S"), ("B", "B"), ("B", "H")]
)
def test_level_protocols(loader, op, x_format, y_format):
    data_x = np.array([0, 2, 0, 4, 5, 0, 0, 7, 0, 9])
    data_y = np.array([1, 0, 0, 3, 6, 0, 8, 7, 0, 0])
    x = stored(loader, x_format, data_x)
    y = stored(loader, y_format, data_y)
    for proto_x in PROTOCOLS[x_format]:
        for proto_y in PROTOCOLS[y_format]:
            _, z = tensors("D", data_x)
            program = coiterate_program(x, y, z, op, ((proto_x,), (proto_y,)))
            kernel = NotationCompiler(loader, mode=CompilerMode(debug=True))(
                program
            ).coiterate
            result = kernel(x, y, z)
            np.testing.assert_array_equal(
                dense_values(result.lvl),
                op(data_x, data_y),
                err_msg=f"{proto_x}, {proto_y}",
            )


def walks(format_):
    return tuple(
        ntn.DefaultProtocol() if char == "D" else ntn.Walk() for char in format_
    )


@pytest.mark.parametrize("op", [ffuncs.mul, ffuncs.add])
@pytest.mark.parametrize("x_format, y_format", [("DS", "SS"), ("SB", "DH")])
def test_level_protocols_follow_their_indices(loader, op, x_format, y_format):
    data_x = np.array([[0, 2, 0, 4], [0, 0, 0, 0], [5, 0, 0, 7], [0, 0, 9, 0]])
    data_y = np.array([[1, 0, 0, 3], [0, 6, 0, 0], [8, 0, 0, 7], [0, 0, 0, 0]])
    x = stored(loader, x_format, data_x)
    y = stored(loader, y_format, data_y)
    for protos in itertools.product(*(PROTOCOLS[char] for char in x_format)):
        _, z = tensors("DD", data_x)
        program = coiterate_program(x, y, z, op, (protos, walks(y_format)))
        kernel = NotationCompiler(loader, mode=CompilerMode(debug=True))(
            program
        ).coiterate
        result = kernel(x, y, z)
        np.testing.assert_array_equal(
            dense_values(result.lvl), op(data_x, data_y), err_msg=f"{protos}"
        )


@pytest.mark.parametrize(
    "format_, protocol",
    [("D", ntn.Walk()), ("D", ntn.Gallop()), ("H", ntn.Gallop())],
)
def test_unsupported_level_protocols(format_, protocol):
    loader = asm.AssemblyInterpreter()
    data = np.array([0, 2, 0, 4])
    x = stored(loader, format_, data)
    _, z = tensors("D", data)
    program = coiterate_program(x, x, z, ffuncs.mul, ((protocol,), walks(format_)))
    with pytest.raises(NotImplementedError, match=f"the {protocol} protocol"):
        NotationCompiler(loader)(program)
