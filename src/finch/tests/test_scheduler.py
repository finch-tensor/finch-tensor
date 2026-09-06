import numpy as np

import finch
from finch.algebra import ffuncs
from finch.algebra.ftypes import ftype
from finch.autoschedule import (
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    LogicCapture,
    normalize_names,
)
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_ordering import concordize, heuristic_loop_order
from finch.autoschedule.optimize import (
    isolate_aggregates,
    lift_fields,
    optimize,
    propagate_copy_queries,
    propagate_map_queries,
    propagate_map_queries_backward,
    propagate_transpose_queries,
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
                Alias("A10"),
                MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))),
            ),
            Query(Alias("A11"), Table(Alias("A10"), ())),
            Produces((Alias("A11"),)),
        )
    )
    expected = Plan(
        (
            Query(
                Alias("A11"),
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
            Query(Alias("A0"), Table(Alias("A1"), ())),
            Query(Alias("table-1"), Table(Alias("A0"), (Field("i0"), Field("i1")))),
            Query(
                Alias("map-join-1"),
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
                Alias("aggregate-1"),
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
                Alias("aggregate-2"),
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
            Query(Alias("table-1"), Table(Alias("A1"), (Field("i0"), Field("i1")))),
            Query(
                Alias("map-join-1"),
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
                Alias("aggregate-1"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(10),
                    Table(Alias("A2"), ()),
                    (Field("i5"), Field("i6")),
                ),
            ),
            Query(
                Alias("aggregate-2"),
                Reorder(
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
                    (Field("i10"),),
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
                Alias("A0"),
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
                        Alias(f"#A#{_sg.counter}"),
                        Aggregate(
                            Literal(ffuncs.mul),
                            Literal(1),
                            Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                            (Field("i2"),),
                        ),
                    ),
                    Query(
                        Alias("A0"),
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
                    Alias("relabel-1"),
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
                Alias("relabel-2"),
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
                Alias("reorder-1"),
                Reorder(
                    Aggregate(
                        Literal("+"),
                        Literal(0),
                        Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                        (Field("A2"),),
                    ),
                    (Field("A3"), Field("A1")),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Query(
                Alias("relabel-1"),
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
                Alias("relabel-2"),
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
                Alias("reorder-1"),
                Reorder(
                    Aggregate(
                        Literal("+"),
                        Literal(0),
                        Reorder(
                            Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                            (Field("A3"), Field("A2"), Field("A1")),
                        ),
                        (Field("A2"),),
                    ),
                    (Field("A3"), Field("A1")),
                ),
            ),
        )
    )

    result = push_fields(plan)
    assert result == expected


def test_propagate_copy_queries():
    plan = Plan(
        (
            Query(Alias("A0"), Table(Alias("A0"), (Field("i0"),))),
            Query(Alias("A1"), Table(Alias("A2"), (Field("i1"),))),
            Query(Alias("A1"), Table(Literal(0), (Field("i1"),))),
            Produces((Alias("A1"),)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Query(Alias("A2"), Table(Literal(0), (Field("i1"),))),
            Produces((Alias("A2"),)),
        )
    )

    result = propagate_copy_queries(plan, {})
    assert result == expected


def test_propagate_transpose_queries():
    plan = Plan(
        (
            Query(
                Alias("A1"),
                Relabel(
                    Table(
                        Alias("XD"),
                        (Field("i1"), Field("i2")),
                    ),
                    (Field("j1"), Field("j2")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("A1"), (Field("j1"), Field("j2"))),
                    (Field("j2"), Field("j1")),
                ),
            ),
            Produces((Alias("A2"),)),
        )
    )

    expected = Plan(
        (
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("XD"), (Field("j1"), Field("j2"))),
                    (Field("j2"), Field("j1")),
                ),
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
                Alias("A_#"),
                Aggregate(
                    Literal("*"),
                    Literal(1),
                    Table(Literal(2), (Field("i1"), Field("i2"))),
                    (Field("i2"),),
                ),
            ),
            Query(
                Alias("A0"),
                MapJoin(
                    Literal("*"),
                    (
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        Table(Literal(4), (Field("i1"), Field("i2"))),
                    ),
                ),
            ),
            Query(
                Alias("A0"),
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
                Alias("A_#"),
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
                Alias("A0"),
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
                Alias("A0"),
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
                Alias("A0"),
                Table(Alias("A0"), (Field("##foo#8"),)),
            ),
            Query(
                Alias("A1"),
                Table(Alias("A1"), (Field("##foo#1"),)),
            ),
            Query(
                Alias("A2"),
                Table(Alias("A2"), (Field("#2#foo"),)),
            ),
            Query(
                Alias("##foo#9"),
                Table(Alias("##foo#9"), ()),
            ),
            Query(
                Alias("A4"),
                Table(Alias("A4"), (Field("#10#A"),)),
            ),
            Query(
                Alias("bar"),
                Table(Alias("bar"), ()),
            ),
            Query(
                Alias("A5"),
                Table(Alias("A5"), (Field("j"),)),
            ),
            Query(
                Alias("##test#0"),
                Table(Alias("##test#0"), ()),
            ),
        )
    )

    expected = Plan(
        (
            Query(
                Alias("A"),
                Table(Alias("A"), (Field("i"),)),
            ),
            Query(
                Alias("A_2"),
                Table(Alias("A_2"), (Field("i_2"),)),
            ),
            Query(
                Alias("A_3"),
                Table(Alias("A_3"), (Field("i_3"),)),
            ),
            Query(
                Alias("A_4"),
                Table(Alias("A_4"), ()),
            ),
            Query(
                Alias("A_5"),
                Table(Alias("A_5"), (Field("i_4"),)),
            ),
            Query(
                Alias("A_6"),
                Table(Alias("A_6"), ()),
            ),
            Query(
                Alias("A_7"),
                Table(Alias("A_7"), (Field("i_5"),)),
            ),
            Query(
                Alias("A_8"),
                Table(Alias("A_8"), ()),
            ),
        )
    )

    result, bindings = normalize_names(plan, {})
    assert result == expected


def test_concordize():
    plan = Plan(
        (
            Query(Alias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                Alias("A1"),
                Reorder(
                    Table(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
            ),
            Produces((Alias("A1"), Alias("A2"))),
        )
    )

    expected = Plan(
        (
            Query(Alias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                Alias("A0_4"),
                Reorder(
                    Table(Alias("A0"), (Field("i_0"), Field("i_1"))),
                    (Field("i_1"), Field("i_0")),
                ),
            ),
            Query(
                Alias("A0_5"),
                Reorder(
                    Table(Alias("A0"), (Field("i_0"), Field("i_1"))),
                    (Field("i_0"), Field("i_1")),
                ),
            ),
            Query(
                Alias("A1"),
                Reorder(
                    Table(Alias("A0_4"), (Field("i1"), Field("i0"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("A0_5"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
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
                Alias("C"),
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
                Alias("C"),
                Reorder(
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
                    (Field("i0"), Field("i2")),
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
                    Query(Alias("A0"), Table(Alias("A0"), (Field("i0"),))),
                    Query(Alias("A1"), Table(Alias("A1"), (Field("i0"),))),
                )
            ),
            Query(Alias("A2"), Table(Alias("A2"), ())),
            Plan(
                (
                    Plan(
                        (
                            Query(Alias("A3"), Table(Alias("A3"), (Field("i3"),))),
                            Produces((Alias("A4"),)),
                        )
                    ),
                )
            ),
            Query(Alias("A5"), Table(Alias("A5"), (Field("i4"),))),
            Query(Alias("A6"), Table(Alias("A6"), (Field("i0"),))),
        )
    )

    expected = Plan(
        (
            Query(Alias("A0"), Table(Alias("A0"), (Field("i0"),))),
            Query(Alias("A1"), Table(Alias("A1"), (Field("i0"),))),
            Query(Alias("A2"), Table(Alias("A2"), ())),
            Query(Alias("A3"), Table(Alias("A3"), (Field("i3"),))),
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
                Alias("AB"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(Alias("A"), (i, k)), Table(Alias("B"), (k, j))),
                ),
            ),
            Query(
                Alias("C"),
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
                Alias("AB"),
                Reorder(
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Reorder(Table(Alias("A"), (i, j)), (i, j)),
                            Reorder(Table(Alias("B"), (k, j)), (j, k)),
                        ),
                    ),
                    (i, j, k),
                ),
            ),
            # matmul
            Query(
                Alias("C"),
                Aggregate(
                    Literal(ffuncs.add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)
                ),
            ),
            # elemwise
            Query(
                Alias("RES"),
                Reorder(
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Reorder(Table(Alias("C"), (i, j)), (i, j)),
                            Reorder(Table(Alias("S"), (j, i)), (i, j)),
                        ),
                    ),
                    (i, j),
                ),
            ),
            Produces((Alias("RES"),)),
        )
    )

    capture = LogicCapture()
    scheduler = DefaultLogicOptimizer(
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
                lhs=Alias(name="A2"),
                rhs=Reorder(
                    arg=MapJoin(
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
                    idxs=(Field(name="i0"), Field(name="i2")),
                ),
            ),
            Plan(
                bodies=(Produces(args=(Alias(name="A2"),)),),
            ),
        ),
    )
    capture = LogicCapture()
    scheduler = DefaultLogicOptimizer(
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
