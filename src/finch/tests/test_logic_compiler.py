import pytest

import numpy as np

import finch.finch_logic as logic
import finch.finch_notation as ntn
from finch import ffuncs, ftype
from finch.algebra import DynamicFill
from finch.autoschedule import (
    CompilerFormLowerer,
    LogicCapture,
    NotationGenerator,
)
from finch.autoschedule.util import fuse_loop_order
from finch.compile import NotationCompiler
from finch.finch_logic import (
    Aggregate,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
)
from finch.tensor.bufferized_ndarray import (
    BufferizedNDArray,
)

from .conftest import finch_assert_equal, reset_name_counts
from .utils import with_workspaces


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize("init", [0, 7])
@pytest.mark.parametrize(
    "kind",
    ["copy", "dynamic_copy", "pointwise", "dynamic_pointwise", "reduction", "inplace"],
)
def test_generated_init_write(kind, init, compiler):
    i, j = Field("i"), Field("j")
    src, dst = HardAlias("src"), HardAlias("dst")
    data = np.array([[5, 0], [4, 0]], dtype=np.int64)
    init = np.int64(init)
    fill = DynamicFill(init) if kind.startswith("dynamic_") else init
    if kind in ("copy", "dynamic_copy"):
        query = Query(Table(dst, (j, i)), Table(src, (i, j)))
        expected = data.T
    else:
        reduced = (j,) if kind == "reduction" else ()
        output_idxs = (i,) if reduced else (i, j)
        rhs = Aggregate(
            Literal(ffuncs.overwrite),
            Literal(fill),
            Table(src, (i, j)),
            reduced,
        )
        if kind == "inplace":
            query = QueryInto(
                Table(dst, output_idxs),
                Literal(ffuncs.overwrite),
                Table(src, (i, j)),
            )
        else:
            query = Query(Table(dst, output_idxs), rhs)
        expected = data[:, -1] if reduced else data

    # A declaration can't change a static fill, so the output has the fill it is
    # declared with, which a copy takes from its source.
    bindings = {
        src: BufferizedNDArray.from_numpy(
            data, fill_value=fill if kind in ("copy", "dynamic_copy") else 0
        ),
        dst: BufferizedNDArray.from_numpy(
            np.full(expected.shape, 9, dtype=np.int64), fill_value=fill
        ),
    }
    loop_order = (j, i) if kind in ("copy", "dynamic_copy") else (i, j)
    plan = Plan((fuse_loop_order(query, loop_order), Produces((dst,))))
    capture = LogicCapture()
    CompilerFormLowerer(capture)(
        plan,
        {var: ftype(val) for var, val in bindings.items()},
        {},
        None,
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    result = compiler()(program).main(*with_workspaces(program, *bindings.values()))
    finch_assert_equal(result[0].to_numpy(), expected)


def test_logic_compiler(file_regression):
    plan = Plan(
        bodies=(
            fuse_loop_order(
                Query(
                    lhs=Table(
                        HardAlias(name="A2"), (Field(name="i0"), Field(name="i2"))
                    ),
                    rhs=Aggregate(
                        op=logic.Literal(val=ffuncs.add),
                        init=logic.Literal(val=0),
                        arg=MapJoin(
                            op=logic.Literal(val=ffuncs.mul),
                            args=(
                                Table(
                                    HardAlias(name="A0"),
                                    (Field(name="i0"), Field(name="i1")),
                                ),
                                Table(
                                    HardAlias(name="A1"),
                                    (Field(name="i1"), Field(name="i2")),
                                ),
                            ),
                        ),
                        idxs=(Field(name="i1"),),
                    ),
                ),
                (Field(name="i0"), Field(name="i1"), Field(name="i2")),
            ),
            Produces(args=(HardAlias(name="A2"),)),
        ),
    )

    bindings = {
        HardAlias(name="A0"): BufferizedNDArray.from_numpy(np.array([[1, 2], [3, 4]])),
        HardAlias(name="A1"): BufferizedNDArray.from_numpy(np.array([[5, 6], [7, 8]])),
        HardAlias(name="A2"): BufferizedNDArray.from_numpy(np.array([[0, 0], [0, 0]])),
    }

    capture = LogicCapture()
    CompilerFormLowerer(capture)(
        plan,
        {var: ftype(val) for var, val in bindings.items()},
        {},
        None,
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)

    file_regression.check(
        reset_name_counts(str(program)),
        extension=".txt",
        basename="test_logic_compiler_program",
    )

    result = ntn.NotationInterpreter()(program).main(
        *with_workspaces(program, *bindings.values())
    )

    expected = np.matmul(
        bindings[HardAlias(name="A0")].to_numpy(),
        bindings[HardAlias(name="A1")].to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].to_numpy(), expected)


def test_logic_compiler_inplace(file_regression):
    plan = Plan(
        bodies=(
            fuse_loop_order(
                QueryInto(
                    lhs=Table(
                        HardAlias(name="A2"), (Field(name="i0"), Field(name="i2"))
                    ),
                    op=Literal(ffuncs.add),
                    rhs=MapJoin(
                        op=logic.Literal(val=ffuncs.mul),
                        args=(
                            Table(
                                HardAlias(name="A0"),
                                (Field(name="i0"), Field(name="i1")),
                            ),
                            Table(
                                HardAlias(name="A1"),
                                (Field(name="i1"), Field(name="i2")),
                            ),
                        ),
                    ),
                ),
                (Field(name="i0"), Field(name="i1"), Field(name="i2")),
            ),
            Produces(args=(HardAlias(name="A2"),)),
        ),
    )

    bindings = {
        HardAlias(name="A0"): BufferizedNDArray.from_numpy(np.array([[1, 2], [3, 4]])),
        HardAlias(name="A1"): BufferizedNDArray.from_numpy(np.array([[5, 6], [7, 8]])),
        HardAlias(name="A2"): BufferizedNDArray.from_numpy(np.array([[1, 1], [1, 1]])),
    }

    capture = LogicCapture()
    CompilerFormLowerer(capture)(
        plan,
        {var: ftype(val) for var, val in bindings.items()},
        {},
        None,
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)

    file_regression.check(
        reset_name_counts(str(program)),
        extension=".txt",
        basename="test_logic_compiler_inplace_program",
    )

    result = ntn.NotationInterpreter()(program).main(
        *with_workspaces(program, *bindings.values())
    )

    expected = np.ones_like(bindings[HardAlias(name="A2")].to_numpy()) + np.matmul(
        bindings[HardAlias(name="A0")].to_numpy(),
        bindings[HardAlias(name="A1")].to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].to_numpy(), expected)
