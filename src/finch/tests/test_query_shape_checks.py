import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import ffuncs
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.util import fuse_loop_order
from finch.codegen import NumbaCompiler
from finch.compile import CompilerMode, NotationCompiler
from finch.compile.lower import AssemblyGenerator
from finch.symbolic import PostOrderDFS
from finch.tensor import BufferizedNDArray


@pytest.fixture(
    params=[
        pytest.param(("notation", None), id="notation"),
        *(
            pytest.param((backend, mode), id=f"{backend}-{name}")
            for backend in ("assembly", "numba")
            for name, mode in (
                ("safe", CompilerMode(safe=True)),
                ("debug", CompilerMode(debug=True)),
            )
        ),
    ]
)
def executor(request):
    backend, mode = request.param
    match backend:
        case "notation":
            loader = ntn.NotationInterpreter()
        case _:
            backend = {
                "assembly": asm.AssemblyInterpreter,
                "numba": NumbaCompiler,
            }[backend]()
            loader = NotationCompiler(backend, mode=mode)
    return LogicExecutor(DefaultLogicFormatter(LogicCompiler(loader)), cache=True)


def tensor(values):
    return BufferizedNDArray.from_numpy(np.asarray(values, dtype=np.int64))


def pointwise_plan():
    a, b, c = (lgc.HardAlias(name) for name in "ABC")
    i = lgc.Field("i")
    rhs = lgc.MapJoin(lgc.Literal(ffuncs.add), (lgc.Table(a, (i,)), lgc.Table(b, (i,))))
    return lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(
                    lgc.Table(c, (i,)),
                    lgc.Aggregate(
                        lgc.Literal(ffuncs.overwrite),
                        lgc.Literal(np.int64(0)),
                        rhs,
                        (),
                    ),
                ),
                (i,),
            ),
            lgc.Produces((c,)),
        )
    )


def test_shared_dimensions_checked_before_declare(executor):
    a, b, c = (lgc.HardAlias(name) for name in "ABC")
    plan = pointwise_plan()
    output = tensor([99, 99, 99, 99])
    for a_size, b_size in ((2, 3), (3, 2), (0, 1), (1, 0)):
        with pytest.raises(AssertionError, match="Finch assertion failed:.*dim"):
            executor(
                plan,
                {a: tensor(np.arange(a_size)), b: tensor(np.arange(b_size)), c: output},
            )
        assert output.shape == (4,)
        np.testing.assert_array_equal(output.to_numpy(), [99] * 4)
    for size in (3, 0, 1):
        (result,) = executor(
            plan, {a: tensor(np.arange(size)), b: tensor(np.arange(size)), c: output}
        )
        assert result.shape == (size,)
        np.testing.assert_array_equal(result.to_numpy(), 2 * np.arange(size))
    assert len(executor.cached_kernels) == 1


def test_reduced_dimensions_checked_even_with_empty_output(executor):
    a, b, c = (lgc.HardAlias(name) for name in "ABC")
    i, j, k = (lgc.Field(name) for name in "ijk")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(
                    lgc.Table(c, (i, j)),
                    lgc.Aggregate(
                        lgc.Literal(ffuncs.add),
                        lgc.Literal(np.int64(0)),
                        lgc.MapJoin(
                            lgc.Literal(ffuncs.mul),
                            (lgc.Table(a, (i, k)), lgc.Table(b, (k, j))),
                        ),
                        (k,),
                    ),
                ),
                (i, k, j),
            ),
            lgc.Produces((c,)),
        )
    )
    for rows, left_k, right_k in ((2, 3, 4), (0, 3, 4), (2, 0, 1)):
        with pytest.raises(AssertionError, match="Finch assertion failed:.*dim"):
            executor(
                plan,
                {a: tensor(np.ones((rows, left_k))), b: tensor(np.ones((right_k, 3)))},
            )
    for rows, size in ((2, 3), (0, 3), (2, 0)):
        (result,) = executor(
            plan, {a: tensor(np.ones((rows, size))), b: tensor(np.ones((size, 3)))}
        )
        np.testing.assert_array_equal(result.to_numpy(), np.full((rows, 3), size))
    assert len(executor.cached_kernels) == 1


@pytest.mark.parametrize("op", [ffuncs.add, ffuncs.overwrite])
def test_query_into_checks_shared_fields_and_allows_broadcast(executor, op):
    a, c = lgc.HardAlias("A"), lgc.HardAlias("C")
    i, j = lgc.Field("i"), lgc.Field("j")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.QueryInto(
                    lgc.Table(c, (i, j)),
                    lgc.Literal(op),
                    lgc.Table(a, (i,)),
                ),
                (i, j),
            ),
            lgc.Produces((c,)),
        )
    )
    for a_size, c_size in ((2, 3), (3, 2), (0, 1), (1, 0)):
        output = tensor(np.full((c_size, 4), 99))
        with pytest.raises(AssertionError, match="Finch assertion failed:.*dim"):
            executor(plan, {a: tensor(np.arange(a_size)), c: output})
        np.testing.assert_array_equal(output.to_numpy(), np.full((c_size, 4), 99))
    for shape in ((3, 4), (0, 4), (3, 0)):
        values = np.arange(shape[0])
        (result,) = executor(plan, {a: tensor(values), c: tensor(np.full(shape, 99))})
        expected = np.broadcast_to(values[:, None], shape)
        if op == ffuncs.add:
            expected = expected + 99
        np.testing.assert_array_equal(result.to_numpy(), expected)
    assert len(executor.cached_kernels) == 1


def test_empty_overwrite_reduction_still_checks_output_shape(executor):
    a, c = lgc.HardAlias("A"), lgc.HardAlias("C")
    i, j = lgc.Field("i"), lgc.Field("j")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.QueryInto(
                    lgc.Table(c, (i,)),
                    lgc.Literal(ffuncs.overwrite),
                    lgc.Table(a, (i, j)),
                ),
                (i, j),
            ),
            lgc.Produces((c,)),
        )
    )
    output = tensor([99, 99])
    with pytest.raises(AssertionError, match="Finch assertion failed:.*dim"):
        executor(plan, {a: tensor(np.zeros((3, 0))), c: output})
    np.testing.assert_array_equal(output.to_numpy(), [99, 99])
    (result,) = executor(plan, {a: tensor(np.zeros((2, 0))), c: output})
    np.testing.assert_array_equal(result.to_numpy(), [99, 99])


def test_shape_checks_use_current_definition(executor):
    a, b, c = (lgc.HardAlias(name) for name in "ABC")
    i = lgc.Field("i")
    plan = lgc.Plan(
        (
            fuse_loop_order(lgc.Query(lgc.Table(c, (i,)), lgc.Table(a, (i,))), (i,)),
            fuse_loop_order(lgc.Query(lgc.Table(c, (i,)), lgc.Table(b, (i,))), (i,)),
            fuse_loop_order(
                lgc.QueryInto(
                    lgc.Table(c, (i,)),
                    lgc.Literal(ffuncs.add),
                    lgc.Table(b, (i,)),
                ),
                (i,),
            ),
            lgc.Produces((c,)),
        )
    )
    for size in (5, 0, 2):
        (result,) = executor(
            plan, {a: tensor(np.arange(3)), b: tensor(np.arange(size))}
        )
        np.testing.assert_array_equal(result.to_numpy(), 2 * np.arange(size))
    assert len(executor.cached_kernels) == 1


@pytest.mark.parametrize(
    "mode", [None, CompilerMode(), CompilerMode(safe=True), CompilerMode(debug=True)]
)
def test_shape_checks_follow_loader_mode(mode):
    if mode is None:
        compiler = LogicCompiler()
        mode = compiler.ctx_load.mode
        assert mode.safe and mode.debug
    else:
        compiler = LogicCompiler(NotationCompiler(asm.AssemblyInterpreter(), mode=mode))
    bindings = {lgc.HardAlias(name): tensor([1, 2]).ftype for name in "ABC"}
    program = compiler.ctx_lower(pointwise_plan(), bindings, {}, None)
    checks = [node for node in PostOrderDFS(program) if isinstance(node, ntn.Assert)]
    assert len(checks) == int(mode.safe or mode.debug)
    lowered = AssemblyGenerator(mode=mode)(program)
    asm.AssemblyTypeChecker()(asm.LowerPackedStructSlots()(lowered))
    if checks:
        assert "assert(" in str(program)
        ntn.NotationCFGBuilder().build(program)
