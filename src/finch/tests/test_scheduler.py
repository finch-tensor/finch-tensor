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
    Field,
    HardAlias,
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
                HardAlias("A10"),
                MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))),
            ),
            Query(HardAlias("A11"), Table(HardAlias("A10"), ())),
            Produces((HardAlias("A11"),)),
        )
    )
    expected = Plan(
        (
            Query(
                HardAlias("A11"),
                Relabel(MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))), ()),
            ),
            Produces((HardAlias("A11"),)),
        )
    )

    result = propagate_map_queries(plan)
    assert result == expected


def test_propagate_map_queries_backward():
    plan = Plan(
        (
            Query(HardAlias("A0"), HardAlias("A1")),
            Table(HardAlias("A0"), (Field("i0"), Field("i1"))),
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
                    Table(Literal(10), (Field("i4"),)),
                ),
            ),
            Aggregate(
                Literal(ffuncs.add),
                Literal(10),
                Aggregate(
                    Literal(ffuncs.add), Literal(0), HardAlias("A2"), (Field("i5"),)
                ),
                (Field("i6"),),
            ),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                Reorder(
                    Aggregate(
                        Literal(ffuncs.add),
                        Literal(0),
                        Table(
                            HardAlias("A3"),
                            (Field("i10"), Field("i7"), Field("i9"), Field("i8")),
                        ),
                        (Field("i7"), Field("i8")),
                    ),
                    (Field("i9"), Field("i10")),
                ),
                (Field("i9"),),
            ),
            Produces(()),
        )
    )

    expected = Plan(
        (
            Plan(()),
            Relabel(HardAlias("A1"), (Field("i0"), Field("i1"))),
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(10), (Field("i2"),)),
                        Table(Literal(10), (Field("i2"), Field("i3"), Field("i4"))),
                        Table(Literal(10), (Field("i4"),)),
                    ),
                ),
                (Field("i3"),),
            ),
            Aggregate(
                Literal(ffuncs.add),
                Literal(10),
                HardAlias("A2"),
                (Field("i5"), Field("i6")),
            ),
            Reorder(
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        Table(
                            HardAlias("A3"),
                            (Field("i10"), Field("i7"), Field("i9"), Field("i8")),
                        ),
                        (
                            Field("i9"),
                            Field("i7"),
                            Field("i10"),
                            Field("i8"),
                        ),
                    ),
                    (Field("i7"), Field("i8"), Field("i9")),
                ),
                (Field("i10"),),
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
                HardAlias("A0"),
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
                        HardAlias(f"#A#{_sg.counter}"),
                        Aggregate(
                            Literal(ffuncs.mul),
                            Literal(1),
                            Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                            (Field("i2"),),
                        ),
                    ),
                    Query(
                        HardAlias("A0"),
                        Aggregate(
                            Literal(ffuncs.add),
                            Literal(0),
                            Table(
                                HardAlias(f"#A#{_sg.counter}"),
                                (Field("i1"), Field("i3")),
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
            Relabel(
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
                (Field("B1"), Field("B3")),
            ),
            Reorder(
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
                (Field("A3"), Field("A1")),
            ),
        )
    )

    expected = Plan(
        (
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
            Aggregate(
                op=Literal(val="+"),
                init=Literal(val=0),
                arg=Table(
                    tns=Literal(val=""),
                    idxs=(Field(name="B1"), Field(name="A2"), Field(name="B3")),
                ),
                idxs=(Field(name="A2"),),
            ),
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
        )
    )

    result = push_fields(plan)
    assert result == expected


def test_propagate_copy_queries():
    plan = Plan(
        (
            Query(HardAlias("A0"), Table(HardAlias("A0"), (Field("i0"),))),
            Query(HardAlias("A1"), Table(HardAlias("A2"), (Field("i1"),))),
            Query(HardAlias("A1"), Table(Literal(0), (Field("i1"),))),
            Produces((HardAlias("A1"),)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Query(HardAlias("A2"), Table(Literal(0), (Field("i1"),))),
            Produces((HardAlias("A2"),)),
        )
    )

    result = propagate_copy_queries(plan, {})
    assert result == expected


def test_propagate_transpose_queries():
    plan = Plan(
        (
            Query(
                HardAlias("A1"),
                Relabel(
                    Table(
                        HardAlias("XD"),
                        (Field("i1"), Field("i2")),
                    ),
                    (Field("j1"), Field("j2")),
                ),
            ),
            Query(
                HardAlias("A2"),
                Reorder(
                    Table(HardAlias("A1"), (Field("j1"), Field("j2"))),
                    (Field("j2"), Field("j1")),
                ),
            ),
            Produces((HardAlias("A2"),)),
        )
    )

    expected = Plan(
        (
            Query(
                HardAlias("A2"),
                Reorder(
                    Table(HardAlias("XD"), (Field("j1"), Field("j2"))),
                    (Field("j2"), Field("j1")),
                ),
            ),
            Produces((HardAlias("A2"),)),
        )
    )

    result = propagate_transpose_queries(plan)
    assert result == expected


def test_lift_fields():
    plan = Plan(
        (
            Aggregate(
                Literal("*"),
                Literal(1),
                Table(Literal(2), (Field("i1"), Field("i2"))),
                (Field("i2"),),
            ),
            Query(
                HardAlias("A0"),
                MapJoin(
                    Literal("*"),
                    (
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        Table(Literal(4), (Field("i1"), Field("i2"))),
                    ),
                ),
            ),
            Query(
                HardAlias("A0"),
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
            Aggregate(
                Literal("*"),
                Literal(1),
                Reorder(
                    Table(Literal(2), (Field("i1"), Field("i2"))),
                    (Field("i1"), Field("i2")),
                ),
                (Field("i2"),),
            ),
            Query(
                HardAlias("A0"),
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
                HardAlias("A0"),
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
            Field("##foo#8"),
            Field("##foo#1"),
            Field("#2#foo"),
            HardAlias("##foo#9"),
            Field("#10#A"),
            HardAlias("bar"),
            Field("j"),
            HardAlias("##test#0"),
        )
    )

    expected = Plan(
        (
            Field("i"),
            Field("i_2"),
            Field("i_3"),
            HardAlias("A"),
            Field("i_4"),
            HardAlias("A_2"),
            Field("i_5"),
            HardAlias("A_3"),
        )
    )

    result, bindings = normalize_names(plan, {})
    assert result == expected


def test_concordize():
    plan = Plan(
        (
            Query(HardAlias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                HardAlias("A1"),
                Reorder(
                    Table(HardAlias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                HardAlias("A2"),
                Reorder(
                    Table(HardAlias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
            ),
            Produces((HardAlias("A1"), HardAlias("A2"))),
        )
    )

    expected = Plan(
        (
            Query(HardAlias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                HardAlias("A0_4"),
                Reorder(
                    Table(HardAlias("A0"), (Field("i_0"), Field("i_1"))),
                    (Field("i_1"), Field("i_0")),
                ),
            ),
            Query(
                HardAlias("A0_5"),
                Reorder(
                    Table(HardAlias("A0"), (Field("i_0"), Field("i_1"))),
                    (Field("i_0"), Field("i_1")),
                ),
            ),
            Query(
                HardAlias("A1"),
                Reorder(
                    Table(HardAlias("A0_4"), (Field("i1"), Field("i0"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                HardAlias("A2"),
                Reorder(
                    Table(HardAlias("A0_5"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
            ),
            Produces((HardAlias("A1"), HardAlias("A2"))),
        )
    )

    result = concordize(plan, bindings={})
    assert result == expected


def test_heuristic_loop_order():
    plan = Plan(
        (
            Query(
                HardAlias("C"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        MapJoin(
                            Literal(ffuncs.mul),
                            (
                                Reorder(
                                    Table(HardAlias("A"), (Field("i0"), Field("i1"))),
                                    (Field("i0"), Field("i1")),
                                ),
                                Reorder(
                                    Table(HardAlias("B"), (Field("i1"), Field("i2"))),
                                    (Field("i1"), Field("i2")),
                                ),
                            ),
                        ),
                        (Field("i0"), Field("i2"), Field("i1")),
                    ),
                    (Field("i1"),),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    expected = Plan(
        (
            Query(
                HardAlias("C"),
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
                                                HardAlias("A"),
                                                (Field("i0"), Field("i1")),
                                            ),
                                            (Field("i0"), Field("i1")),
                                        ),
                                        Reorder(
                                            Table(
                                                HardAlias("B"),
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
            Produces((HardAlias("C"),)),
        )
    )

    result = heuristic_loop_order(plan)
    assert result == expected


def test_flatten_plans():
    plan = Plan(
        (
            Plan(
                (
                    Field("i0"),
                    Field("i1"),
                )
            ),
            HardAlias("A0"),
            Plan(
                (
                    Plan(
                        (
                            Field("i3"),
                            Produces((HardAlias("A1"),)),
                        )
                    ),
                )
            ),
            Field("i4"),
            HardAlias("A2"),
        )
    )

    expected = Plan(
        (
            Field("i0"),
            Field("i1"),
            HardAlias("A0"),
            Field("i3"),
            Produces((HardAlias("A1"),)),
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
                HardAlias("AB"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(HardAlias("A"), (i, k)), Table(HardAlias("B"), (k, j))),
                ),
            ),
            Query(
                HardAlias("C"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
                ),
            ),
            Produces((HardAlias("C"),)),
        )
    )

    plan_opt, bindings = optimize(
        plan,
        {
            HardAlias("A"): ftype(finch.asarray(a)),
            HardAlias("B"): ftype(finch.asarray(b)),
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
                HardAlias("AB"),
                Reorder(
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Reorder(Table(HardAlias("A"), (i, j)), (i, j)),
                            Reorder(Table(HardAlias("B"), (k, j)), (j, k)),
                        ),
                    ),
                    (i, j, k),
                ),
            ),
            # matmul
            Query(
                HardAlias("C"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(HardAlias("AB"), (i, k, j)),
                    (k,),
                ),
            ),
            # elemwise
            Query(
                HardAlias("RES"),
                Reorder(
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Reorder(Table(HardAlias("C"), (i, j)), (i, j)),
                            Reorder(Table(HardAlias("S"), (j, i)), (i, j)),
                        ),
                    ),
                    (i, j),
                ),
            ),
            Produces((HardAlias("RES"),)),
        )
    )

    capture = LogicCapture()
    scheduler = DefaultLogicOptimizer(
        DefaultLoopOrderer(DefaultLogicFormatter(capture))
    )
    bindings = {
        HardAlias("S"): finch.asarray(s),
        HardAlias("A"): finch.asarray(a),
        HardAlias("B"): finch.asarray(b),
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
                lhs=HardAlias(name="A2"),
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
                                                HardAlias(name="A0"),
                                                (Field(name="i0"), Field(name="i1")),
                                            ),
                                            Table(
                                                HardAlias(name="A1"),
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
                                        HardAlias("A2"),
                                        (Field(name="i0"), Field(name="i2")),
                                    ),
                                    Table(
                                        HardAlias("A1"),
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
                bodies=(Produces(args=(HardAlias(name="A2"),)),),
            ),
        ),
    )
    capture = LogicCapture()
    scheduler = DefaultLogicOptimizer(
        DefaultLoopOrderer(DefaultLogicFormatter(capture))
    )

    bindings = {
        HardAlias(name="A0"): finch.asarray(np.array([[1, 2], [3, 4]])),
        HardAlias(name="A1"): finch.asarray(np.array([[5, 6], [7, 8]])),
        HardAlias(name="A2"): finch.asarray(np.array([[1, 1], [1, 1]])),
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
