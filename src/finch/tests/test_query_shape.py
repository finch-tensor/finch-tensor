import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import ffuncs
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.einsum import LogicEinsumLoader
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.util import fuse_loop_order
from finch.codegen import NumbaCompiler
from finch.compile import NotationCompiler
from finch.tensor import BufferizedNDArray, BufferizedNDArrayFType


@pytest.fixture(
    params=[
        "einsum",
        "notation",
        "assembly",
        "numba",
    ]
)
def executor(request):
    match request.param:
        case "einsum":
            loader = LogicEinsumLoader()
        case "notation":
            loader = LogicCompiler(ntn.NotationInterpreter())
        case _:
            backend = {
                "assembly": asm.AssemblyInterpreter,
                "numba": NumbaCompiler,
            }[request.param]()
            loader = LogicCompiler(NotationCompiler(backend))
    return LogicExecutor(DefaultLogicFormatter(loader), cache=True)


def test_query_shapes_flow_through_cached_kernel(executor, monkeypatch):
    a, b, c, d = (lgc.HardAlias(name) for name in "ABCD")
    i, j, k = (lgc.Field(name) for name in "ijk")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(b, (j, i, k)), lgc.Table(a, (i, j))), (j, i, k)
            ),
            fuse_loop_order(
                lgc.QueryInto(
                    lgc.Table(b, (j, i, k)),
                    lgc.Literal(ffuncs.add),
                    lgc.MapJoin(
                        lgc.Literal(ffuncs.add),
                        (lgc.Literal(np.int64(1)), lgc.Literal(np.int64(1))),
                    ),
                ),
                (j, i, k),
            ),
            fuse_loop_order(
                lgc.Query(
                    lgc.Table(c, (j, k)),
                    lgc.Aggregate(
                        lgc.Literal(ffuncs.add),
                        lgc.Literal(np.int64(0)),
                        lgc.Table(b, (j, i, k)),
                        (i,),
                    ),
                ),
                (j, i, k),
            ),
            fuse_loop_order(
                lgc.Query(lgc.Table(d, (k, j)), lgc.Table(c, (j, k))), (k, j)
            ),
            lgc.Produces((b, c, d)),
        )
    )
    allocations = []
    construct = BufferizedNDArrayFType.construct

    def record_construct(self, shape, fill_value=None):
        allocations.append(shape)
        return construct(self, shape, fill_value)

    monkeypatch.setattr(BufferizedNDArrayFType, "construct", record_construct)
    for shape in ((2, 3), (4, 1), (0, 3), (2, 0), (1, 2)):
        values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        tensor = BufferizedNDArray.from_numpy(values)
        allocations.clear()
        results = executor(plan, {a: tensor})
        assert allocations == [(0, 0, 0), (0, 0), (0, 0)]
        expected_b = (values.T + 2)[..., None]
        expected_c = expected_b.sum(axis=1)
        for result, expected in zip(
            results, (expected_b, expected_c, expected_c.T), strict=True
        ):
            assert result.shape == expected.shape
            np.testing.assert_array_equal(np.asarray(result), expected)
    assert len(executor.cached_kernels) == 1


def test_logic_interpreter_sizes_query_outputs():
    a, b = lgc.HardAlias("A"), lgc.HardAlias("B")
    i, j = lgc.Field("i"), lgc.Field("j")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(b, (j, i)), lgc.Table(a, (i, j))), (j, i)
            ),
            lgc.Produces((lgc.Table(b, (j, i)),)),
        )
    )
    executor = LogicExecutor(DefaultLogicFormatter(lgc.MockLogicLoader()), cache=True)
    for shape in ((2, 3), (0, 2), (4, 1)):
        values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        (result,) = executor(plan, {a: BufferizedNDArray.from_numpy(values)})
        assert result.shape == values.T.shape
        np.testing.assert_array_equal(result.to_numpy(), values.T)
    assert len(executor.cached_kernels) == 1


@pytest.mark.parametrize("ndim", [0, 2])
def test_query_expands_scalar_dimensions(executor, ndim):
    a, b = lgc.HardAlias("A"), lgc.HardAlias("B")
    idxs = tuple(lgc.Field(f"i_{i}") for i in range(ndim))
    plan = lgc.Plan(
        (
            fuse_loop_order(lgc.Query(lgc.Table(b, idxs), lgc.Table(a, ())), idxs),
            lgc.Produces((b,)),
        )
    )
    for value in (3, 7):
        tensor = BufferizedNDArray.from_numpy(np.array(value, dtype=np.int64))
        (result,) = executor(plan, {a: tensor})
        assert result.shape == (1,) * ndim
        np.testing.assert_array_equal(np.asarray(result), np.full(result.shape, value))
    assert len(executor.cached_kernels) == 1


def test_query_into_keeps_bound_shape(executor):
    a, b = lgc.HardAlias("A"), lgc.HardAlias("B")
    i, j = lgc.Field("i"), lgc.Field("j")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.QueryInto(
                    lgc.Table(b, (i, j)),
                    lgc.Literal(ffuncs.add),
                    lgc.Table(a, (i,)),
                ),
                (i, j),
            ),
            lgc.Produces((b,)),
        )
    )
    for shape in ((3, 4), (0, 2), (3, 0)):
        a_np = np.arange(shape[0], dtype=np.int64)
        b_np = np.ones(shape, dtype=np.int64)
        (result,) = executor(
            plan,
            {
                a: BufferizedNDArray.from_numpy(a_np),
                b: BufferizedNDArray.from_numpy(b_np.copy()),
            },
        )
        assert result.shape == shape
        np.testing.assert_array_equal(np.asarray(result), b_np + a_np[:, None])


def test_query_replaces_bound_shape(executor):
    a, b = lgc.HardAlias("A"), lgc.HardAlias("B")
    i, j = lgc.Field("i"), lgc.Field("j")
    plan = lgc.Plan(
        (
            fuse_loop_order(
                lgc.Query(lgc.Table(b, (j, i)), lgc.Table(a, (i, j))), (j, i)
            ),
            lgc.Produces((b,)),
        )
    )
    output = BufferizedNDArray.from_numpy(np.full((1, 1), 99, dtype=np.int64))
    for shape in ((2, 3), (4, 1), (0, 2)):
        values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        (result,) = executor(plan, {a: BufferizedNDArray.from_numpy(values), b: output})
        assert result.shape == values.T.shape
        np.testing.assert_array_equal(np.asarray(result), values.T)
    assert len(executor.cached_kernels) == 1
