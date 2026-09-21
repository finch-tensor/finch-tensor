import pytest

import numpy as np

import finch.finch_logic as logic
import finch.finch_notation as ntn
from finch import ffuncs, ftype
from finch.algebra import DynamicFill
from finch.autoschedule import INTERPRET_NOTATION, NotationGenerator
from finch.compile import NotationCompiler
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from finch.symbolic import PostOrderDFS
from finch.tensor.bufferized_ndarray import (
    BufferizedNDArray,
)

from .conftest import finch_assert_equal, reset_name_counts


@pytest.mark.parametrize("compiler", [ntn.NotationInterpreter, NotationCompiler])
@pytest.mark.parametrize("init", [0, 7])
@pytest.mark.parametrize(
    "kind",
    ["copy", "dynamic_copy", "pointwise", "dynamic_pointwise", "reduction", "inplace"],
)
def test_generated_init_write(kind, init, compiler):
    i, j = Field("i"), Field("j")
    src, dst = Alias("src"), Alias("dst")
    data = np.array([[5, 0], [4, 0]], dtype=np.int64)
    init = np.int64(init)
    fill = DynamicFill(init) if kind.startswith("dynamic_") else init
    op = ffuncs.overwrite
    if kind in ("copy", "dynamic_copy"):
        query = Query(dst, Reorder(Table(src, (i, j)), (j, i)))
        expected = data.T
        if kind == "copy":
            op = ffuncs.init_write(init)
    else:
        reduced = (j,) if kind == "reduction" else ()
        output_idxs = (i,) if reduced else (i, j)
        rhs = Aggregate(
            Literal(ffuncs.overwrite),
            Literal(fill),
            Reorder(Table(src, (i, j)), (i, j)),
            reduced,
        )
        if kind == "inplace":
            rhs = MapJoin(Literal(ffuncs.overwrite), (Table(dst, output_idxs), rhs))
        query = Query(dst, Reorder(rhs, output_idxs))
        expected = data[:, -1] if reduced else data
        if kind == "pointwise":
            op = ffuncs.init_write(init)

    # Static pointwise initialization can differ from the storage format's fill.
    output_fill = 0 if kind == "pointwise" else fill
    bindings = {
        src: BufferizedNDArray.from_numpy(data),
        dst: BufferizedNDArray.from_numpy(
            np.full(expected.shape, 9, dtype=np.int64), fill_value=output_fill
        ),
    }
    plan = Plan((query, Produces((dst,))))
    program = NotationGenerator()(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )
    operators = []
    for node in PostOrderDFS(program):
        match node:
            case (
                ntn.Declare(_, _, actual, _)
                | ntn.Thaw(_, actual)
                | ntn.Freeze(_, actual)
                | ntn.Increment(ntn.Access(_, ntn.Update(actual), _), _)
            ):
                operators.append(actual)
    assert operators == [ntn.Literal(op)] * 3

    result = compiler()(program).main(*bindings.values())
    finch_assert_equal(result[0].to_numpy(), expected)


def test_logic_compiler(file_regression):
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name="A2"),
                rhs=Reorder(
                    arg=Aggregate(
                        op=logic.Literal(val=ffuncs.add),
                        init=logic.Literal(val=0),
                        arg=Reorder(
                            arg=MapJoin(
                                op=logic.Literal(val=ffuncs.mul),
                                args=(
                                    Table(
                                        Alias(name="A0"),
                                        (Field(name="i0"), Field(name="i1")),
                                    ),
                                    Table(
                                        Alias(name="A1"),
                                        (Field(name="i1"), Field(name="i2")),
                                    ),
                                ),
                            ),
                            idxs=(Field(name="i0"), Field(name="i1"), Field(name="i2")),
                        ),
                        idxs=(Field(name="i1"),),
                    ),
                    idxs=(Field(name="i0"), Field(name="i2")),
                ),
            ),
            Produces(args=(Alias(name="A2"),)),
        ),
    )

    bindings = {
        Alias(name="A0"): BufferizedNDArray.from_numpy(np.array([[1, 2], [3, 4]])),
        Alias(name="A1"): BufferizedNDArray.from_numpy(np.array([[5, 6], [7, 8]])),
        Alias(name="A2"): BufferizedNDArray.from_numpy(np.array([[0, 0], [0, 0]])),
    }

    program = NotationGenerator()(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )

    file_regression.check(
        reset_name_counts(str(program)),
        extension=".txt",
        basename="test_logic_compiler_program",
    )

    result = INTERPRET_NOTATION(plan, bindings)

    expected = np.matmul(
        bindings[Alias(name="A0")].to_numpy(),
        bindings[Alias(name="A1")].to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].to_numpy(), expected)


def test_logic_compiler_inplace(file_regression):
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name="A2"),
                rhs=Reorder(
                    arg=MapJoin(
                        op=Literal(ffuncs.add),
                        args=(
                            Table(Alias("A2"), (Field(name="i0"), Field(name="i2"))),
                            Aggregate(
                                op=logic.Literal(val=ffuncs.add),
                                init=logic.Literal(val=0),
                                arg=Reorder(
                                    arg=MapJoin(
                                        op=logic.Literal(val=ffuncs.mul),
                                        args=(
                                            Table(
                                                Alias(name="A0"),
                                                (Field(name="i0"), Field(name="i1")),
                                            ),
                                            Table(
                                                Alias(name="A1"),
                                                (Field(name="i1"), Field(name="i2")),
                                            ),
                                        ),
                                    ),
                                    idxs=(
                                        Field(name="i0"),
                                        Field(name="i1"),
                                        Field(name="i2"),
                                    ),
                                ),
                                idxs=(Field(name="i1"),),
                            ),
                        ),
                    ),
                    idxs=(Field(name="i0"), Field(name="i2")),
                ),
            ),
            Produces(args=(Alias(name="A2"),)),
        ),
    )

    bindings = {
        Alias(name="A0"): BufferizedNDArray.from_numpy(np.array([[1, 2], [3, 4]])),
        Alias(name="A1"): BufferizedNDArray.from_numpy(np.array([[5, 6], [7, 8]])),
        Alias(name="A2"): BufferizedNDArray.from_numpy(np.array([[1, 1], [1, 1]])),
    }

    program = NotationGenerator()(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )

    file_regression.check(
        reset_name_counts(str(program)),
        extension=".txt",
        basename="test_logic_compiler_inplace_program",
    )

    result = INTERPRET_NOTATION(plan, bindings)

    expected = np.ones_like(bindings[Alias(name="A2")].to_numpy()) + np.matmul(
        bindings[Alias(name="A0")].to_numpy(),
        bindings[Alias(name="A1")].to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].to_numpy(), expected)
