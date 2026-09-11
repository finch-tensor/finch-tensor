import numpy as np

import finch as fl
from finch.algebra import ffuncs, ftype
from finch.autoschedule.compiler import LogicCompiler, NotationGenerator
from finch.finch_logic import (
    Aggregate,
    Field,
    FusedAlias,
    HardAlias,
    Literal,
    LogicInterpreter,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)


def test_ex_interpreter():
    i, j = Field("i"), Field("j")
    A_np = np.array([[1, 2], [3, 4]])
    A = fl.asarray(A_np)

    # Interpreter end

    # B = A * 2
    plan_interpreter = Plan(
        (
            Query(
                HardAlias("B"),
                MapJoin(
                    Literal(ffuncs.mul), (Table(HardAlias("A"), (i, j)), Literal(2))
                ),
            ),
            Produces((HardAlias("B"),)),
        )
    )

    # Running it through interpreter
    bindings = {HardAlias("A"): A}
    (result,) = LogicInterpreter().lower(plan_interpreter, bindings)
    print(result.to_numpy())


def test_ex_notation():
    i, j = Field("i"), Field("j")
    A_np = np.array([[1, 2], [3, 4]])
    A = fl.asarray(A_np)

    # Compiler end - needs reorder node
    B = fl.asarray(np.zeros((2, 2), dtype=int))

    plan_compiler = Plan(
        (
            Query(HardAlias("B"), Reorder(Table(HardAlias("A"), (i, j)), (i, j))),
            Produces((HardAlias("B"),)),
        )
    )
    bindings = {HardAlias("A"): ftype(A), HardAlias("B"): ftype(B)}
    module = NotationGenerator().lower(plan_compiler, bindings, {}, None)
    print(module)


# Trying MockFusedTensor
def test_fused_alias_interpreter():
    i, j, k = Field("i"), Field("j"), Field("k")
    A = fl.asarray(np.array([[1, 2, 0], [0, 3, 4]]))
    B = fl.asarray(np.array([[1, 0], [0, 1], [2, 2]]))
    scratch = FusedAlias(HardAlias("scratch"), 1)

    plan_w = Plan(
        (
            Query(
                scratch,
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                    ),
                    (k,),
                ),
            ),
            Query(HardAlias("C"), Table(scratch, (i, j))),
            Produces((HardAlias("C"),)),
        )
    )

    bindings_w = {HardAlias("A"): A, HardAlias("B"): B}
    (C,) = LogicInterpreter().lower(plan_w, bindings_w)
    print(f"C = \n{C.to_numpy()}")
    mock = bindings_w[HardAlias("scratch")]
    # Checking what got stored under scratch
    print(f"Scratch storage type : {type(mock).__name__}")
    print(f"Scratch storage shape to loop through : {mock.shape} ")
    print(f"Scratch storage actual shape : {mock.inner_shape} ")
    print(f"How many copies of scratch space do we have : {len(mock.store_tns)}\n")
    for outer_i, tns in mock.store_tns.items():
        print(f"i={outer_i[0]}, holding {tns.to_numpy()}")


def test_plan_compiled_gustavsons():
    i, j, k = Field("i"), Field("j"), Field("k")
    A = fl.asarray(np.array([[1, 2, 0], [0, 3, 4]]))
    B = fl.asarray(np.array([[1, 0], [0, 1], [2, 2]]))
    scratch_alias = HardAlias("scratch")
    fused_scratch = FusedAlias(HardAlias("scratch"), 1)

    plan_compiled = Plan(
        (
            Query(
                fused_scratch,
                Reorder(
                    Aggregate(
                        Literal(ffuncs.add),
                        Literal(0),
                        Reorder(
                            MapJoin(
                                Literal(ffuncs.mul),
                                (
                                    Table(HardAlias("A"), (i, k)),
                                    Table(HardAlias("B"), (k, j)),
                                ),
                            ),
                            (i, k, j),
                        ),
                        (k,),
                    ),
                    (i, j),
                ),
            ),
            Query(HardAlias("C"), Reorder(Table(fused_scratch, (i, j)), (i, j))),
            Produces((HardAlias("C"),)),
        )
    )

    scratch_val = fl.asarray(np.zeros((2, 2), dtype=int))
    C_val = fl.asarray(np.zeros((2, 2), dtype=int))
    binding_ftypes = {
        HardAlias("A"): ftype(A),
        HardAlias("B"): ftype(B),
        HardAlias("C"): ftype(C_val),
        scratch_alias: ftype(scratch_val),
    }
    print(NotationGenerator().lower(plan_compiled, binding_ftypes, {}, None))
    lib, *_ = LogicCompiler().lower(plan_compiled, binding_ftypes, {}, None)
    (result,) = lib.main(A, B, scratch_val, C_val)
    print(result.to_numpy())
