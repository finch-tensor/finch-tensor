import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_logic as lgc
from finch import finch_notation as ntn
from finch.algebra import DynamicFill, ffuncs, is_dynamic
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.codegen import NumbaCompiler
from finch.compile import NotationCompiler
from finch.tensor import BufferizedNDArray


@pytest.fixture(params=["logic", "notation", "assembly", "numba"])
def executor(request):
    match request.param:
        case "logic":
            loader = lgc.MockLogicLoader()
        case "notation":
            loader = LogicCompiler(ntn.NotationInterpreter())
        case _:
            backend = {
                "assembly": asm.AssemblyInterpreter,
                "numba": NumbaCompiler,
            }[request.param]()
            loader = LogicCompiler(NotationCompiler(backend))
    return LogicExecutor(DefaultLogicFormatter(loader), cache=True)


def produces(executor, *tables):
    if isinstance(executor.ctx.ctx, lgc.MockLogicLoader):
        return lgc.Produces(tables)
    return lgc.Produces(tuple(table.tns for table in tables))


@pytest.mark.parametrize("ndim", [0, 1])
def test_query_fills_flow_through_cached_kernel(executor, monkeypatch, ndim):
    a, b, c = (lgc.HardAlias(name) for name in "ABC")
    idxs = tuple(lgc.Field(f"i_{i}") for i in range(ndim))
    one = lgc.Literal(np.float64(1))
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(b, idxs), lgc.Table(a, idxs)),
            lgc.Query(
                lgc.Table(c, idxs),
                lgc.Aggregate(
                    lgc.Literal(ffuncs.overwrite),
                    lgc.MapJoin(lgc.Literal(ffuncs.add), (lgc.FillTable(b), one)),
                    lgc.Reorder(
                        lgc.MapJoin(lgc.Literal(ffuncs.add), (lgc.Table(b, idxs), one)),
                        idxs,
                    ),
                    (),
                ),
            ),
            produces(executor, lgc.Table(b, idxs), lgc.Table(c, idxs)),
        )
    )

    def reject_fill_inference(*args, **kwargs):
        raise AssertionError("Output fills must be computed by the kernel")

    monkeypatch.setattr(lgc.LogicStatement, "infer_fill_value", reject_fill_inference)
    for size, fill in ((3, 2.0), (0, 7.0), (2, -4.0)):
        shape = (size,) * ndim
        values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        tensor = BufferizedNDArray.from_numpy(
            values, fill_value=DynamicFill(np.float64(fill))
        )
        copied, added = executor(plan, {a: tensor})
        for result, expected, expected_fill in (
            (copied, values, fill),
            (added, values + 1, fill + 1),
        ):
            assert is_dynamic(result.fill)
            assert result.fill_value == expected_fill
            np.testing.assert_array_equal(result.to_numpy(), expected)
    assert len(executor.cached_kernels) == 1


@pytest.mark.parametrize(
    "dtype, fills",
    [(np.bool_, (True, False)), (np.complex128, (3.5 + 7j, 1.5 + 2j))],
)
def test_query_copies_dynamic_fills(executor, dtype, fills):
    a, b = lgc.HardAlias("A"), lgc.HardAlias("B")
    i = lgc.Field("i")
    plan = lgc.Plan(
        (
            lgc.Query(lgc.Table(b, (i,)), lgc.Table(a, (i,))),
            produces(executor, lgc.Table(b, (i,))),
        )
    )
    for size, fill in zip((2, 0), fills, strict=True):
        values = np.zeros(size, dtype=dtype)
        tensor = BufferizedNDArray.from_numpy(
            values, fill_value=DynamicFill(dtype(fill))
        )
        (result,) = executor(plan, {a: tensor})
        assert is_dynamic(result.fill)
        assert result.fill_value == tensor.fill_value
        np.testing.assert_array_equal(result.to_numpy(), values)
    assert len(executor.cached_kernels) == 1
