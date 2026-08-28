import pytest

import numpy as np

import finch as fl
from finch.algebra import ffuncs, ftype
from finch.autoschedule.compiler import NotationGenerator
from finch.finch_logic import (
    Alias,
    Field,
    FusedAlias,
    Literal,
    LogicInterpreter,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from finch.tensor.bufferized_ndarray import (
    BufferizedNDArray,
)

from .conftest import reset_name_counts


# 1. Introduced node to finch_logic
def test_fused_alias_node():
    w = Alias("w")
    fa = FusedAlias(w, 1)
    assert fa.alias == w
    assert fa.n == 1
    assert FusedAlias(Alias("w"), 1) == FusedAlias(Alias("w"), 1)
    assert FusedAlias(Alias("w"), 1) != FusedAlias(Alias("w"), 2)
    with pytest.raises(NotImplementedError):
        fa.fields()


# 2. Testing if I can use it in a plan or if it is evaluable
def test_use_fused_alias():
    i, j = Field("i"), Field("j")
    a = fl.asarray(np.array([[1, 2], [3, 4]]))
    w = FusedAlias(Alias("w"), 1)
    p = Plan(
        (
            Query(w, Table(Literal(a), (i, j))),
            Query(
                Alias("C"), MapJoin(Literal(ffuncs.mul), (Table(w, (i, j)), Literal(2)))
            ),
            Produces((Alias("C"),)),
        )
    )
    result = LogicInterpreter()(p)[0]
    expected = a.to_numpy() * 2
    assert (result.to_numpy() == expected).all()


# 3. Adding FusedAlias to compiler.py in places it can occur
def test_fused_alias_compiles(file_regression):
    i, j = Field("i"), Field("j")
    w = FusedAlias(Alias("w"), 1)
    plan = Plan(
        bodies=(
            Query(w, Reorder(Table(Alias("A0"), (i, j)), (i, j))),
            Query(Alias("A1"), Reorder(Table(w, (i, j)), (i, j))),
            Produces(args=(Alias("A1"),)),
        )
    )

    bindings = {
        Alias(name="A0"): BufferizedNDArray.from_numpy(np.array([[1, 2], [3, 4]])),
        Alias(name="w"): BufferizedNDArray.from_numpy(np.array([[0, 0], [0, 0]])),
        Alias(name="A1"): BufferizedNDArray.from_numpy(np.array([[0, 0], [0, 0]])),
    }

    program = NotationGenerator()(
        plan, {var: ftype(val) for var, val in bindings.items()}, {}, None
    )
    file_regression.check(reset_name_counts(str(program)), extension=".txt")
