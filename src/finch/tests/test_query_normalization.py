from finch import ffuncs
from finch.autoschedule.factorizer.galley_factorizer.query_normalization import (
    merge_queries,
    preprocess_plan_for_galley,
)
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


def _contains_reorder(expr) -> bool:
    if isinstance(expr, Reorder):
        return True
    if isinstance(expr, MapJoin):
        return any(_contains_reorder(arg) for arg in expr.args)
    if isinstance(expr, Aggregate):
        return _contains_reorder(expr.arg)
    return False


def test_merge_queries_inlines_alias_tables_and_keeps_produced_aliases():
    """
    INPUT:
      A1 = Table(A, i, j)
      A2 = Table(A1, j, i)
      return A2

    EXPECTED AFTER merge_queries:
      A2[j, i] = Table(A, j, i)   # A1 inlined; direct ref to
                                    base tensor w/ reordered fields
      return A2
    """
    i = Field("i")
    j = Field("j")

    A_lit = Literal("A")
    q1 = Query(Table(Alias("A1"), (i, j)), Table(A_lit, (i, j)))
    q2 = Query(Table(Alias("A2"), (j, i)), Table(Alias("A1"), (j, i)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    assert isinstance(preprocessed, Plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies

    assert isinstance(out_query, Query)
    assert out_query.lhs == Table(Alias("A2"), (j, i))
    assert out_query.rhs == Table(A_lit, (j, i))

    assert isinstance(out_produces, Produces)
    assert out_produces.args == (Alias("A2"),)


def test_normalize_reorders_strips_all_reorders():
    """
    INPUT:
      out[j] = Reorder(Aggregate(add, 0, Reorder(Table(A, i, j), j, i), i), j)
      (Aggregate reduces i; inner Reorder swaps Table to (j,i); outer asks for (j,))

    EXPECTED AFTER normalize_reorders_in_plan:
      out[j] = Aggregate(add, 0, Table(A, i, j), i)
    """
    i = Field("i")
    j = Field("j")

    A_lit = Literal("A")
    inner = Reorder(Table(A_lit, (i, j)), (j, i))
    agg = Aggregate(Literal(ffuncs.add), Literal(0), inner, (i,))
    original_rhs = Reorder(agg, (j,))
    q = Query(Table(Alias("out"), original_rhs.fields()), original_rhs)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, norm_produces = preprocessed.bodies
    assert isinstance(norm_q, Query)

    assert norm_q.lhs == Table(Alias("out"), (j,))
    assert norm_q.rhs == Aggregate(
        Literal(ffuncs.add), Literal(0), Table(A_lit, (i, j)), (i,)
    )

    assert isinstance(norm_produces, Produces)
    assert norm_produces.args == (Alias("out"),)


def test_preprocess_plan_for_galley_produces_canonical_queries():
    """
    INPUT:
      A1 = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
      # A@B -> (i,k)
      A2[i, k] = Table(A1, i, k)
      return A2

    EXPECTED AFTER preprocess_plan_for_galley:
      Single merged query for A2. A1 inlined;
      A2[i, k] = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")

    A_lit = Literal("A")
    B_lit = Literal("B")

    mj = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(A_lit, (i, j)),
            Table(B_lit, (j, k)),
        ),
    )
    agg = Aggregate(Literal(ffuncs.add), Literal(0), mj, (j,))
    q1 = Query(Table(Alias("A1"), agg.fields()), agg)
    q2 = Query(Table(Alias("A2"), (i, k)), Table(Alias("A1"), (i, k)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    # Expect a single query for A2 plus Produces(A2)
    assert isinstance(preprocessed, Plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies
    assert isinstance(out_query, Query)
    assert out_query.lhs.tns == Alias("A2")

    assert out_query.lhs.idxs == (i, k)
    assert isinstance(out_query.rhs, Aggregate)
    assert not _contains_reorder(out_query.rhs)

    assert isinstance(out_produces, Produces)
    assert out_produces.args == (Alias("A2"),)


def test_merge_queries_chain_of_three_aliases():
    """
    INPUT:
      A1 = Table(A, i, j)
      A2 = Table(A1, j, i)
      A3 = Table(A2, i, j)
      return A3

    EXPECTED AFTER merge_queries:
      A3[i, j] = Table(A, i, j)
      return A3
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    q1 = Query(Table(Alias("A1"), (i, j)), Table(A_lit, (i, j)))
    q2 = Query(Table(Alias("A2"), (j, i)), Table(Alias("A1"), (j, i)))
    q3 = Query(Table(Alias("A3"), (i, j)), Table(Alias("A2"), (i, j)))
    plan = Plan((q1, q2, q3, Produces((Alias("A3"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies

    assert isinstance(out_query, Query)
    assert isinstance(out_produces, Produces)

    assert out_query.lhs == Table(Alias("A3"), (i, j))
    assert out_query.rhs == Table(A_lit, (i, j))


def test_merge_queries_produces_multiple_aliases():
    """
    INPUT:
      A1 = Table(A, i, j)
      A2 = Table(A1, j, i)
      return A1, A2

    EXPECTED AFTER merge_queries:
      A1[i, j] = Table(A, i, j)
      A2[j, i] = Table(A1, j, i)   # A1 is a produced alias,
                                     so it is NOT inlined
      return A1, A2
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    q1 = Query(Table(Alias("A1"), (i, j)), Table(A_lit, (i, j)))
    q2 = Query(Table(Alias("A2"), (j, i)), Table(Alias("A1"), (j, i)))
    plan = Plan((q1, q2, Produces((Alias("A1"), Alias("A2")))))

    preprocessed = preprocess_plan_for_galley(plan)
    assert len(preprocessed.bodies) == 3
    out_q1, out_q2, out_produces = preprocessed.bodies

    assert isinstance(out_q1, Query)
    assert isinstance(out_q2, Query)
    assert isinstance(out_produces, Produces)

    assert out_q1.lhs == Table(Alias("A1"), (i, j))
    assert out_q1.rhs == Table(A_lit, (i, j))

    assert out_q2.lhs == Table(Alias("A2"), (j, i))
    assert out_q2.rhs == Table(Alias("A1"), (j, i))


def test_normalize_reorders_strips_transpose_reorder():
    """
    INPUT:
      out[j, i] = Reorder(Table(A, i, j), j, i)   # swap i,j

    EXPECTED AFTER normalize_reorders_in_plan:
      out[j, i] = Table(A, i, j)   # the lhs keeps the swapped order
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    original_rhs = Reorder(Table(A_lit, (i, j)), (j, i))
    q = Query(Table(Alias("out"), original_rhs.fields()), original_rhs)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, _ = preprocessed.bodies

    assert isinstance(norm_q, Query)
    assert norm_q.lhs == Table(Alias("out"), (j, i))
    assert norm_q.rhs == Table(A_lit, (i, j))


def test_normalize_reorders_nested_reorders_collapse():
    """
    INPUT:
      out[i, j] = Reorder(Reorder(Table(A, i, j), j, i), i, j)   # swap, swap back

    EXPECTED AFTER normalize_reorders_in_plan:
      out[i, j] = Table(A, i, j)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    inner = Reorder(Table(A_lit, (i, j)), (j, i))
    outer = Reorder(inner, (i, j))
    q = Query(Table(Alias("out"), outer.fields()), outer)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, _ = preprocessed.bodies

    assert isinstance(norm_q, Query)
    assert norm_q.lhs == Table(Alias("out"), (i, j))
    assert norm_q.rhs == Table(A_lit, (i, j))


def test_normalize_reorders_aggregate_drops_outer_reorder():
    """
    INPUT:
      out[i] = Reorder(Aggregate(add, 0, Table(A, i, j), j), i)
      Aggregate reduces j, natural output (i,). Reorder asks for (i,) - same.

    EXPECTED AFTER normalize_reorders_in_plan:
      out[i] = Aggregate(add, 0, Table(A, i, j), j)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    agg = Aggregate(Literal(ffuncs.add), Literal(0), Table(A_lit, (i, j)), (j,))
    original_rhs = Reorder(agg, (i,))
    q = Query(Table(Alias("out"), original_rhs.fields()), original_rhs)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, _ = preprocessed.bodies

    assert isinstance(norm_q, Query)
    assert norm_q.lhs == Table(Alias("out"), (i,))
    assert norm_q.rhs == agg


def test_preprocess_plan_chain_with_reorder_and_aggregate():
    """
    INPUT:
      A1 = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
      A2[k, i] = Table(A1, i, k)   # swap output to (k, i)
      return A2

    EXPECTED AFTER preprocess_plan_for_galley:
      A2[k, i] = single merged query with no Reorders
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")
    A_lit = Literal("A")
    B_lit = Literal("B")

    mj = MapJoin(
        Literal(ffuncs.mul),
        (Table(A_lit, (i, j)), Table(B_lit, (j, k))),
    )
    agg = Aggregate(Literal(ffuncs.add), Literal(0), mj, (j,))
    q1 = Query(Table(Alias("A1"), agg.fields()), agg)
    q2 = Query(Table(Alias("A2"), (k, i)), Table(Alias("A1"), (i, k)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies
    assert isinstance(out_query, Query)
    assert isinstance(out_produces, Produces)

    assert out_query.lhs == Table(Alias("A2"), (k, i))
    assert not _contains_reorder(out_query.rhs)


def test_preprocess_plan_single_table_no_change():
    """
    INPUT:
      out = Table(A, i, j)
      return out

    EXPECTED AFTER preprocess_plan_for_galley:
      out[i, j] = Table(A, i, j)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    q = Query(Table(Alias("out"), (i, j)), Table(A_lit, (i, j)))
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert len(preprocessed.bodies) == 2
    out_query, _ = preprocessed.bodies
    assert out_query == q


def test_preprocess_plan_A_at_B_at_C():
    """
    INPUT (A @ B @ C - chain of two matrix multiplications):
      A1 = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
      # A@B -> (i,k)
      A2 = Aggregate(add, 0, MapJoin(mul, [Table(A1,i,k), Table(C,k,l)]), k)
      # (A@B)@C -> (i,l)
      return A2

    EXPECTED AFTER preprocess_plan_for_galley:
      Single merged query for A2. push_aggregates_up distributes mul over add,
      yielding one Aggregate with reduction (j,k) over
      MapJoin(mul, [MapJoin(mul, A, B), C]).
      No interior Reorders.
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")
    l_ = Field("l")
    A_lit = Literal("A")
    B_lit = Literal("B")
    C_lit = Literal("C")

    # A @ B: (i, j) @ (j, k) -> (i, k)
    ab = MapJoin(
        Literal(ffuncs.mul),
        (Table(A_lit, (i, j)), Table(B_lit, (j, k))),
    )
    q1 = Query(
        Table(Alias("A1"), (i, k)),
        Aggregate(Literal(ffuncs.add), Literal(0), ab, (j,)),
    )

    # (A @ B) @ C: (i, k) @ (k, l) -> (i, l)
    abc = MapJoin(
        Literal(ffuncs.mul),
        (Table(Alias("A1"), (i, k)), Table(C_lit, (k, l_))),
    )
    q2 = Query(
        Table(Alias("A2"), (i, l_)),
        Aggregate(Literal(ffuncs.add), Literal(0), abc, (k,)),
    )
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies
    assert isinstance(out_query, Query)
    assert isinstance(out_produces, Produces)
    assert out_query.lhs == Table(Alias("A2"), (i, l_))

    # Single Aggregate with both reduction indices after push_aggregates_up.
    # Internal fields may get fresh names when inlining (e.g. j -> gensym), so
    # we only assert the count, not the exact names.
    assert isinstance(out_query.rhs, Aggregate)
    assert len(out_query.rhs.idxs) == 2


def test_merge_queries_inlines_mapjoin_aggregate_chain():
    """
    INPUT (matmul then sum over one axis):
      A = Table(X, i, i_2)
      A_2 = Table(Y, i_3, i_4)
      A_3 = MapJoin(mul, Table(A, i_11, i_12), Table(A_2, i_12, i_13))
      A_4 = Aggregate(add, 0, Table(A_3, i_16, i_17, i_18), i_17)
      return A_4

    EXPECTED AFTER merge_queries:
      A_4[i_16, i_18] = Aggregate(add, 0, MapJoin(mul, Table(X, i_16, i_17),
        Table(Y, i_17, i_18)), i_17)
      A_3 inlined into A_4; A and A_2 inlined to base tensors w/ alpha-renamed idxs.
    """
    i = Field("i")
    i_2 = Field("i_2")
    i_3 = Field("i_3")
    i_4 = Field("i_4")
    i_11 = Field("i_11")
    i_12 = Field("i_12")
    i_13 = Field("i_13")
    i_16 = Field("i_16")
    i_17 = Field("i_17")
    i_18 = Field("i_18")

    X_lit = Literal("X")
    Y_lit = Literal("Y")

    q1 = Query(Table(Alias("A"), (i, i_2)), Table(X_lit, (i, i_2)))
    q2 = Query(Table(Alias("A_2"), (i_3, i_4)), Table(Y_lit, (i_3, i_4)))
    a3_rhs = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(Alias("A"), (i_11, i_12)),
            Table(Alias("A_2"), (i_12, i_13)),
        ),
    )
    q3 = Query(Table(Alias("A_3"), a3_rhs.fields()), a3_rhs)
    q4 = Query(
        Table(Alias("A_4"), (i_16, i_18)),
        Aggregate(
            Literal(ffuncs.add),
            Literal(0),
            Table(Alias("A_3"), (i_16, i_17, i_18)),
            (i_17,),
        ),
    )
    plan = Plan((q1, q2, q3, q4, Produces((Alias("A_4"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert isinstance(preprocessed, Plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies

    assert isinstance(out_query, Query)
    assert out_query.lhs.tns == Alias("A_4")
    assert isinstance(out_produces, Produces)
    assert out_produces.args == (Alias("A_4"),)

    assert out_query.lhs.idxs == (i_16, i_18)
    rhs = out_query.rhs
    assert isinstance(rhs, Aggregate)
    assert rhs.idxs == (i_17,)
    assert isinstance(rhs.arg, MapJoin)
    assert isinstance(rhs.arg.args[0], Table)
    assert rhs.arg.args[0].tns == X_lit
    assert rhs.arg.args[0].idxs == (i_16, i_17)
    assert isinstance(rhs.arg.args[1], Table)
    assert rhs.arg.args[1].tns == Y_lit
    assert rhs.arg.args[1].idxs == (i_17, i_18)


def _collect_reduce_idxs(expr):
    """Recursively collect all reduction indices from Aggregate nodes."""
    result = []
    if isinstance(expr, Aggregate):
        result.extend(expr.idxs)
        result.extend(_collect_reduce_idxs(expr.arg))
    elif isinstance(expr, MapJoin):
        for arg in expr.args:
            result.extend(_collect_reduce_idxs(arg))
    elif isinstance(expr, Reorder):
        result.extend(_collect_reduce_idxs(expr.arg))
    elif isinstance(expr, Table):
        pass
    return result


def test_merge_queries_same_alias_inlined_twice_unique_internal_fields():
    """
    When the same alias is inlined multiple times (e.g. B @ B where B = A @ A),
    internal fields (reduction indices) must get fresh names at each call site
    to avoid index collisions.

    INPUT:
      A = Table(X, i, j)
      B = Aggregate(add, 0, MapJoin(mul, Table(A,i,k), Table(A,k,j)), k)   # A @ A
      C = Aggregate(add, 0, MapJoin(mul, Table(B,i,m), Table(B,m,j)), m)   # B @ B
      return C

    After merge_queries, the two inlined copies of B's RHS must use distinct
    contraction indices; otherwise the computation would be wrong.
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")
    m = Field("m")
    X_lit = Literal("X")

    # B = A @ A: (i,k) @ (k,j) -> (i,j), contracts over k
    a_rhs = Table(X_lit, (i, j))
    q_a = Query(Table(Alias("A"), a_rhs.fields()), a_rhs)
    b_rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0),
        MapJoin(
            Literal(ffuncs.mul),
            (Table(Alias("A"), (i, k)), Table(Alias("A"), (k, j))),
        ),
        (k,),
    )
    q_b = Query(Table(Alias("B"), b_rhs.fields()), b_rhs)
    # C = B @ B: (i,m) @ (m,j) -> (i,j), contracts over m
    c_rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0),
        MapJoin(
            Literal(ffuncs.mul),
            (Table(Alias("B"), (i, m)), Table(Alias("B"), (m, j))),
        ),
        (m,),
    )
    q_c = Query(Table(Alias("C"), c_rhs.fields()), c_rhs)
    plan = Plan((q_a, q_b, q_c, Produces((Alias("C"),))))

    merged = merge_queries(plan)
    assert len(merged.bodies) == 2
    out_query = merged.bodies[0]
    assert isinstance(out_query, Query)
    assert out_query.lhs.tns == Alias("C")

    reduce_idxs = _collect_reduce_idxs(out_query.rhs)
    names = [f.name for f in reduce_idxs]
    assert len(names) == len(set(names)), (
        f"Duplicate reduction index names after inlining same alias twice: {names}"
    )
