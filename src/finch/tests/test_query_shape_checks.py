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
