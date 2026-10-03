import numpy as np

import finch
from finch.algebra import ffuncs
from finch.algebra.ftypes import ftype
from finch.autoschedule import (
    DefaultLogicFactorizer,
    DefaultLoopOrderer,
    LogicCapture,
    normalize_names,
)
from finch.autoschedule.factorizer.optimize import (
    isolate_aggregates,
    lift_fields,
    optimize,
    propagate_copy_queries,
    propagate_map_queries,
    propagate_map_queries_backward,
    propagate_transpose_queries,
)
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer.loop_ordering import (
    concordize,
    heuristic_loop_order,
)
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.autoschedule.util import flatten_plans, push_fields
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
)
from finch.symbolic.gensym import _sg

from .conftest import reset_name_counts


def test_propagate_map_queries():
    plan = Plan(
        (
            Query(
                Table(Alias("A10"), ()),
                MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))),
            ),
            Query(Table(Alias("A11"), ()), Table(Alias("A10"), ())),
            Produces((Alias("A11"),)),
        )
    )
    expected = Plan(
        (
            Query(
                Table(Alias("A11"), ()),
                Relabel(MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))), ()),
            ),
            Produces((Alias("A11"),)),
        )
    )

    result = propagate_map_queries(plan)
    assert result == expected


def test_propagate_map_queries_backward():
    plan = Plan(
        (
            Query(
                Table(Alias("A0"), (Field("i0"), Field("i1"))),
                Table(Alias("A1"), (Field("i0"), Field("i1"))),
            ),
            Query(
                Table(Alias("table-1"), (Field("i0"), Field("i1"))),
                Table(Alias("A0"), (Field("i0"), Field("i1"))),
            ),
            Query(
                Table(Alias("map-join-1"), (Field("i2"), Field("i4"))),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(10), (Field("i2"),)),
                        Aggregate(
                            Literal(ffuncs.add),
                            Literal(0),
                            Table(Literal(10), (Field("i2"), Field("i3"), Field("i4"))),
                            (Field("i3"),),
                        ),
                    ),
                ),
            ),
            Query(
                Table(Alias("aggregate-1"), ()),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(10),
                    Aggregate(
                        Literal(ffuncs.add),
                        Literal(0),
                        Table(Alias("A2"), ()),
                        (Field("i5"),),
                    ),
                    (Field("i6"),),
                ),
            ),
            Query(
                Table(Alias("aggregate-2"), (Field("i10"),)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        Aggregate(
                            Literal(ffuncs.add),
                            Literal(0),
                            Table(
                                Alias("A3"),
                                (Field("i10"), Field("i7"), Field("i9"), Field("i8")),
                            ),
                            (Field("i7"), Field("i8")),
                        ),
                        (Field("i9"), Field("i10")),
                    ),
                    (Field("i9"),),
                ),
            ),
            Produces(()),
        )
    )

    expected = Plan(
        (
            Plan(()),
            Query(
                Table(Alias("table-1"), (Field("i0"), Field("i1"))),
                Table(Alias("A1"), (Field("i0"), Field("i1"))),
            ),
            Query(
                Table(Alias("map-join-1"), (Field("i2"), Field("i4"))),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Table(Literal(10), (Field("i2"),)),
                            Table(Literal(10), (Field("i2"), Field("i3"), Field("i4"))),
                        ),
                    ),
                    (Field("i3"),),
                ),
            ),
            Query(
                Table(Alias("aggregate-1"), ()),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(10),
                    Table(Alias("A2"), ()),
                    (Field("i5"), Field("i6")),
                ),
            ),
            Query(
                Table(Alias("aggregate-2"), (Field("i10"),)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        Table(
                            Alias("A3"),
                            (Field("i10"), Field("i7"), Field("i9"), Field("i8")),
                        ),
                        (Field("i9"), Field("i7"), Field("i10"), Field("i8")),
                    ),
                    (Field("i7"), Field("i8"), Field("i9")),
                ),
            ),
            Produces(()),
        )
    )

    result = propagate_map_queries_backward(plan)
    assert result == expected


def test_isolate_aggregates():
    plan = Plan(
        (
            Query(
                Table(Alias("A0"), (Field("i3"),)),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Aggregate(
                        Literal(ffuncs.mul),
                        Literal(1),
                        Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                        (Field("i2"),),
                    ),
                    (Field("i1"),),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Plan(
                (
                    Query(
                        Table(Alias(f"#A#{_sg.counter}"), (Field("i1"), Field("i3"))),
                        Aggregate(
                            Literal(ffuncs.mul),
                            Literal(1),
                            Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                            (Field("i2"),),
                        ),
                    ),
                    Query(
                        Table(Alias("A0"), (Field("i3"),)),
                        Aggregate(
                            Literal(ffuncs.add),
                            Literal(0),
                            Table(
                                Alias(f"#A#{_sg.counter}"), (Field("i1"), Field("i3"))
                            ),
                            (Field("i1"),),
                        ),
                    ),
                )
            ),
        )
    )

    result = isolate_aggregates(plan)
    assert result == expected


def test_push_fields():
    plan = Plan(
        (
            (
                Query(
                    Table(Alias("relabel-1"), (Field("B1"), Field("B2"))),
                    Relabel(
                        MapJoin(
                            Literal("+"),
                            (
                                Table(Literal("tbl1"), (Field("A1"), Field("A2"))),
                                Table(Literal("tbl2"), (Field("A2"), Field("A1"))),
                            ),
                        ),
                        (Field("B1"), Field("B2")),
                    ),
                )
            ),
            Query(
                Table(Alias("relabel-2"), (Field("B1"), Field("B3"))),
                Relabel(
                    Aggregate(
                        Literal("+"),
                        Literal(0),
                        Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                        (Field("A2"),),
                    ),
                    (Field("B1"), Field("B3")),
                ),
            ),
            Query(
                Table(Alias("reorder-1"), (Field("A3"), Field("A1"))),
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Query(
                Table(Alias("relabel-1"), (Field(name="B1"), Field(name="B2"))),
                MapJoin(
                    op=Literal(val="+"),
                    args=(
                        Table(
                            tns=Literal(val="tbl1"),
                            idxs=(Field(name="B1"), Field(name="B2")),
                        ),
                        Table(
                            tns=Literal(val="tbl2"),
                            idxs=(Field(name="B2"), Field(name="B1")),
                        ),
                    ),
                ),
            ),
            Query(
                Table(Alias("relabel-2"), (Field(name="B1"), Field(name="B3"))),
                Aggregate(
                    op=Literal(val="+"),
                    init=Literal(val=0),
                    arg=Table(
                        tns=Literal(val=""),
                        idxs=(Field(name="B1"), Field(name="A2"), Field(name="B3")),
                    ),
                    idxs=(Field(name="A2"),),
                ),
            ),
            Query(
                Table(Alias("reorder-1"), (Field("A3"), Field("A1"))),
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Reorder(
                        Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                        (Field("A3"), Field("A2"), Field("A1")),
                    ),
                    (Field("A2"),),
                ),
            ),
        )
    )

    result = push_fields(plan)
    assert result == expected


def test_propagate_copy_queries():
    plan = Plan(
        (
            Query(
                Table(Alias("A0"), (Field("i0"),)), Table(Alias("A0"), (Field("i0"),))
            ),
            Query(
                Table(Alias("A1"), (Field("i1"),)), Table(Alias("A2"), (Field("i1"),))
            ),
            Query(
                Table(Alias("A1"), (Field("i1"),)), Table(Literal(0), (Field("i1"),))
            ),
            Produces((Alias("A1"),)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Query(
                Table(Alias("A2"), (Field("i1"),)), Table(Literal(0), (Field("i1"),))
            ),
            Produces((Alias("A2"),)),
        )
    )

    result = propagate_copy_queries(plan, {})
    assert result == expected


def test_propagate_transpose_queries():
    plan = Plan(
        (
            Query(
                Table(Alias("A1"), (Field("j1"), Field("j2"))),
                Relabel(
                    Table(
                        Alias("XD"),
                        (Field("i1"), Field("i2")),
                    ),
                    (Field("j1"), Field("j2")),
                ),
            ),
            Query(
                Table(Alias("A2"), (Field("j2"), Field("j1"))),
                Table(Alias("A1"), (Field("j1"), Field("j2"))),
            ),
            Produces((Alias("A2"),)),
        )
    )

    expected = Plan(
        (
            Query(
                Table(Alias("A2"), (Field("j2"), Field("j1"))),
                Table(Alias("XD"), (Field("j1"), Field("j2"))),
            ),
            Produces((Alias("A2"),)),
        )
    )

    result = propagate_transpose_queries(plan)
    assert result == expected


def test_lift_fields():
    plan = Plan(
        (
            Query(
                Table(Alias("A_#"), (Field("i1"),)),
                Aggregate(
                    Literal("*"),
                    Literal(1),
                    Table(Literal(2), (Field("i1"), Field("i2"))),
                    (Field("i2"),),
                ),
            ),
            Query(
                Table(Alias("A0"), (Field("i1"), Field("i2"))),
                MapJoin(
                    Literal("*"),
                    (
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        Table(Literal(4), (Field("i1"), Field("i2"))),
                    ),
                ),
            ),
            Query(
                Table(Alias("A0"), (Field("i1"), Field("i2"))),
                MapJoin(
                    Literal("*"),
                    (
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        Table(Literal(4), (Field("i1"), Field("i2"))),
                    ),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Query(
                Table(Alias("A_#"), (Field("i1"),)),
                Aggregate(
                    Literal("*"),
                    Literal(1),
                    Reorder(
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        (Field("i1"), Field("i2")),
                    ),
                    (Field("i2"),),
                ),
            ),
            Query(
                Table(Alias("A0"), (Field("i1"), Field("i2"))),
                Reorder(
                    MapJoin(
                        Literal("*"),
                        (
                            Table(Literal(2), (Field("i1"), Field("i2"))),
                            Table(Literal(4), (Field("i1"), Field("i2"))),
                        ),
                    ),
                    (Field("i1"), Field("i2")),
                ),
            ),
            Query(
                Table(Alias("A0"), (Field("i1"), Field("i2"))),
                Reorder(
                    MapJoin(
                        Literal("*"),
                        (
                            Table(Literal(2), (Field("i1"), Field("i2"))),
                            Table(Literal(4), (Field("i1"), Field("i2"))),
                        ),
                    ),
                    (Field("i1"), Field("i2")),
                ),
            ),
        )
    )

    result = lift_fields(plan)
    assert result == expected


def test_normalize_names():
    plan = Plan(
        (
            Query(
                Table(Alias("A0"), (Field("##foo#8"),)),
                Table(Alias("A0"), (Field("##foo#8"),)),
            ),
            Query(
                Table(Alias("A1"), (Field("##foo#1"),)),
                Table(Alias("A1"), (Field("##foo#1"),)),
            ),
            Query(
                Table(Alias("A2"), (Field("#2#foo"),)),
                Table(Alias("A2"), (Field("#2#foo"),)),
            ),
            Query(Table(Alias("##foo#9"), ()), Table(Alias("##foo#9"), ())),
            Query(
                Table(Alias("A4"), (Field("#10#A"),)),
                Table(Alias("A4"), (Field("#10#A"),)),
            ),
            Query(Table(Alias("bar"), ()), Table(Alias("bar"), ())),
            Query(Table(Alias("A5"), (Field("j"),)), Table(Alias("A5"), (Field("j"),))),
            Query(Table(Alias("##test#0"), ()), Table(Alias("##test#0"), ())),
        )
    )

    expected = Plan(
        (
            Query(Table(Alias("A"), (Field("i"),)), Table(Alias("A"), (Field("i"),))),
            Query(
                Table(Alias("A_2"), (Field("i_2"),)),
                Table(Alias("A_2"), (Field("i_2"),)),
            ),
            Query(
                Table(Alias("A_3"), (Field("i_3"),)),
                Table(Alias("A_3"), (Field("i_3"),)),
            ),
            Query(Table(Alias("A_4"), ()), Table(Alias("A_4"), ())),
            Query(
                Table(Alias("A_5"), (Field("i_4"),)),
                Table(Alias("A_5"), (Field("i_4"),)),
            ),
            Query(Table(Alias("A_6"), ()), Table(Alias("A_6"), ())),
            Query(
                Table(Alias("A_7"), (Field("i_5"),)),
                Table(Alias("A_7"), (Field("i_5"),)),
            ),
            Query(Table(Alias("A_8"), ()), Table(Alias("A_8"), ())),
        )
    )

    result, bindings = normalize_names(plan, {})
    assert result == expected


def test_concordize():
    # A0 is read transposed under the loop order of A1, so it needs a
    # swizzled copy. A2 reads A0 in storage order and needs none.
    def overwrite(arg, loop_order):
        return Aggregate(
            Literal(ffuncs.overwrite), Literal(0), Reorder(arg, loop_order), ()
        )

    i0, i1 = Field("i0"), Field("i1")
    plan = Plan(
        (
            Query(Table(Alias("A0"), (i0, i1)), Table(Literal(0), (i0, i1))),
            Query(
                Table(Alias("A1"), (i1, i0)),
                overwrite(Table(Alias("A0"), (i0, i1)), (i1, i0)),
            ),
            Query(
                Table(Alias("A2"), (i0, i1)),
                overwrite(Table(Alias("A0"), (i0, i1)), (i0, i1)),
            ),
            Produces((Alias("A1"), Alias("A2"))),
        )
    )

    expected = Plan(
        (
            Query(Table(Alias("A0"), (i0, i1)), Table(Literal(0), (i0, i1))),
            Query(
                Table(Alias("A0_4"), (Field("i_1"), Field("i_0"))),
                Table(Alias("A0"), (Field("i_0"), Field("i_1"))),
            ),
            Query(
                Table(Alias("A1"), (i1, i0)),
                overwrite(Table(Alias("A0_4"), (i1, i0)), (i1, i0)),
            ),
            Query(
                Table(Alias("A2"), (i0, i1)),
                overwrite(Table(Alias("A0"), (i0, i1)), (i0, i1)),
            ),
            Produces((Alias("A1"), Alias("A2"))),
        )
    )

    result = concordize(plan, bindings={})
    assert result == expected


def test_heuristic_loop_order():
    plan = Plan(
        (
            Query(
                Table(Alias("C"), (Field("i0"), Field("i2"))),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        MapJoin(
                            Literal(ffuncs.mul),
                            (
                                Reorder(
                                    Table(Alias("A"), (Field("i0"), Field("i1"))),
                                    (Field("i0"), Field("i1")),
                                ),
                                Reorder(
                                    Table(Alias("B"), (Field("i1"), Field("i2"))),
                                    (Field("i1"), Field("i2")),
                                ),
                            ),
                        ),
                        (Field("i0"), Field("i2"), Field("i1")),
                    ),
                    (Field("i1"),),
                ),
            ),
            Produces((Alias("C"),)),
        )
    )

    expected = Plan(
        (
            Query(
                Table(Alias("C"), (Field("i0"), Field("i2"))),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        Reorder(
                            MapJoin(
                                Literal(ffuncs.mul),
                                (
                                    Reorder(
                                        Table(
                                            Alias("A"),
                                            (Field("i0"), Field("i1")),
                                        ),
                                        (Field("i0"), Field("i1")),
                                    ),
                                    Reorder(
                                        Table(
                                            Alias("B"),
                                            (Field("i1"), Field("i2")),
                                        ),
                                        (Field("i1"), Field("i2")),
                                    ),
                                ),
                            ),
                            (Field("i0"), Field("i2"), Field("i1")),
                        ),
                        # The contracted index `i1` is shared by both inputs,
                        # so it is placed outermost for an outer-product order.
                        (Field("i1"), Field("i0"), Field("i2")),
                    ),
                    (Field("i1"),),
                ),
            ),
            Produces((Alias("C"),)),
        )
    )

    result = heuristic_loop_order(plan)
    assert result == expected


def test_flatten_plans():
    plan = Plan(
        (
            Plan(
                (
                    Query(
                        Table(Alias("A0"), (Field("i0"),)),
                        Table(Alias("A0"), (Field("i0"),)),
                    ),
                    Query(
                        Table(Alias("A1"), (Field("i0"),)),
                        Table(Alias("A1"), (Field("i0"),)),
                    ),
                )
            ),
            Query(Table(Alias("A2"), ()), Table(Alias("A2"), ())),
            Plan(
                (
                    Plan(
                        (
                            Query(
                                Table(Alias("A3"), (Field("i3"),)),
                                Table(Alias("A3"), (Field("i3"),)),
                            ),
                            Produces((Alias("A4"),)),
                        )
                    ),
                )
            ),
            Query(
                Table(Alias("A5"), (Field("i4"),)), Table(Alias("A5"), (Field("i4"),))
            ),
            Query(
                Table(Alias("A6"), (Field("i0"),)), Table(Alias("A6"), (Field("i0"),))
            ),
        )
    )

    expected = Plan(
        (
            Query(
                Table(Alias("A0"), (Field("i0"),)), Table(Alias("A0"), (Field("i0"),))
            ),
            Query(
                Table(Alias("A1"), (Field("i0"),)), Table(Alias("A1"), (Field("i0"),))
            ),
            Query(Table(Alias("A2"), ()), Table(Alias("A2"), ())),
            Query(
                Table(Alias("A3"), (Field("i3"),)), Table(Alias("A3"), (Field("i3"),))
            ),
            Produces((Alias("A4"),)),
        )
    )

    result = flatten_plans(plan)
    assert result == expected


def test_scheduler_e2e_matmul(file_regression):
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(
                Table(Alias("AB"), (i, k, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(Alias("A"), (i, k)), Table(Alias("B"), (k, j))),
                ),
            ),
            Query(
                Table(Alias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)
                ),
            ),
            Produces((Alias("C"),)),
        )
    )

    plan_opt, bindings = optimize(
        plan,
        {
            Alias("A"): ftype(finch.asarray(a)),
            Alias("B"): ftype(finch.asarray(b)),
        },
    )

    file_regression.check(
        str(plan_opt), extension=".txt", basename="test_scheduler_e2e_matmul_plan"
    )


def test_scheduler_e2e_sddmm(file_regression):
    s = np.array([[2, 4], [6, 0]])
    a = np.array([[1, 2], [3, 2]])
    b = np.array([[9, 8], [6, 5]])
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(
                Table(Alias("AB"), (i, j, k)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Reorder(Table(Alias("A"), (i, j)), (i, j)),
                        Reorder(Table(Alias("B"), (k, j)), (j, k)),
                    ),
                ),
            ),
            # matmul
            Query(
                Table(Alias("C"), (i, j)),
                Aggregate(
                    Literal(ffuncs.add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)
                ),
            ),
            # elemwise
            Query(
                Table(Alias("RES"), (i, j)),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Reorder(Table(Alias("C"), (i, j)), (i, j)),
                        Reorder(Table(Alias("S"), (j, i)), (i, j)),
                    ),
                ),
            ),
            Produces((Alias("RES"),)),
        )
    )

    capture = LogicCapture()
    scheduler = DefaultLogicFactorizer(
        DefaultLoopOrderer(DefaultLogicFormatter(capture))
    )
    bindings = {
        Alias("S"): finch.asarray(s),
        Alias("A"): finch.asarray(a),
        Alias("B"): finch.asarray(b),
    }
    binding_ftypes = {var: val.ftype for var, val in bindings.items()}
    stats_factory = DenseStatsFactory()
    stats = {}
    scheduler(plan, binding_ftypes, stats, stats_factory)
    plan_opt = capture.last_prgm

    file_regression.check(
        reset_name_counts(str(plan_opt)),
        extension=".txt",
        basename="test_scheduler_e2e_sddmm_plan",
    )


def test_scheduler_inplace(file_regression):
    plan = Plan(
        bodies=(
            Query(
                lhs=Table(Alias(name="A2"), (Field(name="i0"), Field(name="i2"))),
                rhs=MapJoin(
                    op=Literal(ffuncs.add),
                    args=(
                        Aggregate(
                            op=Literal(val=ffuncs.add),
                            init=Literal(val=0),
                            arg=Reorder(
                                arg=MapJoin(
                                    op=Literal(val=ffuncs.mul),
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
                        MapJoin(
                            op=Literal(ffuncs.add),
                            args=(
                                Table(
                                    Alias("A2"),
                                    (Field(name="i0"), Field(name="i2")),
                                ),
                                Table(
                                    Alias("A1"),
                                    (Field(name="i0"), Field(name="i2")),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            Plan(
                bodies=(Produces(args=(Alias(name="A2"),)),),
            ),
        ),
    )
    capture = LogicCapture()
    scheduler = DefaultLogicFactorizer(
        DefaultLoopOrderer(DefaultLogicFormatter(capture))
    )

    bindings = {
        Alias(name="A0"): finch.asarray(np.array([[1, 2], [3, 4]])),
        Alias(name="A1"): finch.asarray(np.array([[5, 6], [7, 8]])),
        Alias(name="A2"): finch.asarray(np.array([[1, 1], [1, 1]])),
    }

    binding_ftypes = {var: val.ftype for var, val in bindings.items()}
    stats_factory = DenseStatsFactory()
    stats = {}
    scheduler(plan, binding_ftypes, stats, stats_factory)
    plan_opt = capture.last_prgm

    file_regression.check(
        reset_name_counts(str(plan_opt)),
        extension=".txt",
        basename="test_scheduler_inplace",
    )
