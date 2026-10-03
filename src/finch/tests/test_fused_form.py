import pytest

import finch.finch_logic as lgc
from finch.algebra import as_fill, ffuncs, ftype
from finch.autoschedule.normalize import normalize_names
from finch.autoschedule.stages import FusedForm
from finch.autoschedule.tensor_stats import DenseStatsFactory
from finch.symbolic import PostOrderDFS

i, j, k = map(lgc.Field, ("i", "j", "k"))
a, b, scratch = map(lgc.HardAlias, ("a", "b", "scratch"))
fused = lgc.FusedAlias(scratch, 1)


def test_fuse_node():
    query = lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (j, i)))
    node = lgc.Fuse(i, lgc.Plan((query,)))
    assert node.children == [i, node.body]
    assert node.make_term(node.head(), *node.children) == node
    assert eval(repr(node), vars(lgc)) == node
    assert query in list(PostOrderDFS(node))
    assert node.infer_shape({a: (2, 3)}) == {a: (2, 3), scratch: (3, 2)}
    assert node.infer_element_type({a: ftype(float)})[scratch] == ftype(float)
    assert node.infer_fill_value({a: as_fill(0.0)})[scratch] == as_fill(0.0)

    renamed, _ = normalize_names(node, {})
    match renamed:
        case lgc.Fuse(idx, lgc.Plan((lgc.Query(lgc.Table(_, (outer, _)), _),))):
            assert idx == outer
        case _:
            pytest.fail("Normalization did not preserve the Fuse node")
    FusedForm.validate_inputs(renamed, {}, {}, DenseStatsFactory())


def test_fuse_printer():
    node = lgc.Plan(
        (
            lgc.Fuse(
                i,
                lgc.Plan(
                    (
                        lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i, j))),
                        lgc.Fuse(
                            j,
                            lgc.QueryInto(
                                lgc.Table(b, (i, j)),
                                lgc.Literal(ffuncs.add),
                                lgc.Table(fused, (i, j)),
                            ),
                        ),
                    )
                ),
            ),
            lgc.Produces((b,)),
        )
    )
    assert str(node) == (
        "fuse(i):\n"
        "    scratch(i)[j] = a[i, j]\n"
        "    fuse(j):\n"
        "        b[i, j] <<add>>= scratch(i)[j]\n"
        "return b\n"
    )
    FusedForm.validate_inputs(node, {}, {}, DenseStatsFactory())


@pytest.mark.parametrize(
    "term",
    [
        lgc.Plan(),
        lgc.Query(lgc.Table(lgc.FusedAlias(scratch, 0), (i,)), lgc.Table(a, (i,))),
        lgc.Fuse(i, lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i, j)))),
        # The hard-alias table may be on the lhs.
        lgc.Fuse(i, lgc.Query(lgc.Table(b, (i, j)), lgc.Table(fused, (i, j)))),
        # Descendant tables provide the extents of both enclosing Fuse nodes.
        lgc.Fuse(
            i,
            lgc.Fuse(
                j,
                lgc.Query(
                    lgc.Table(lgc.FusedAlias(scratch, 2), (i, j, k)),
                    lgc.Table(a, (i, j, k)),
                ),
            ),
        ),
        # Scratch storage can be written and read within its fused dimension.
        lgc.Fuse(
            i,
            lgc.Plan(
                (
                    lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i, j))),
                    lgc.Query(lgc.Table(b, (i, j)), lgc.Table(fused, (i, j))),
                )
            ),
        ),
        # Read/write restrictions do not extend beyond the Fuse body.
        lgc.Plan(
            (
                lgc.Fuse(i, lgc.Query(lgc.Table(b, (i,)), lgc.Table(a, (i,)))),
                lgc.Fuse(j, lgc.Query(lgc.Table(a, (j,)), lgc.Table(b, (j,)))),
            )
        ),
        # A tensor not accessed by the fused field may be read and written.
        lgc.Fuse(
            i,
            lgc.Plan(
                (
                    lgc.Query(lgc.Table(scratch, (j,)), lgc.Table(a, (i, j))),
                    lgc.Query(lgc.Table(b, (i, j)), lgc.Table(scratch, (j,))),
                )
            ),
        ),
    ],
)
def test_fused_form_accepts(term):
    FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())


@pytest.mark.parametrize(
    "term",
    [
        lgc.Fuse(i, lgc.Plan()),
        lgc.Fuse(i, lgc.Query(lgc.Table(fused, (i, j)), lgc.Literal(0))),
        lgc.Fuse(i, lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (j,)))),
        # A hard-alias table outside the body cannot supply its fused extent.
        lgc.Plan(
            (
                lgc.Query(lgc.Table(b, (i,)), lgc.Table(a, (i,))),
                lgc.Fuse(i, lgc.Query(lgc.Table(fused, (i,)), lgc.Literal(0))),
            )
        ),
        # Every nesting level needs a hard-alias occurrence of its own field.
        lgc.Fuse(
            i,
            lgc.Fuse(j, lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i,)))),
        ),
    ],
)
def test_fused_form_requires_hard_alias_extent(term):
    with pytest.raises(ValueError, match="must occur in a HardAlias table"):
        FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())


@pytest.mark.parametrize(
    "fields,idxs,n",
    [
        ((), (i,), 1),
        ((i,), (j,), 1),
        ((i,), (i, j), 2),
        ((i, j), (j, i), 2),
        ((i, j), (i, k), 2),
        ((i, j), (j, k), 1),
    ],
)
@pytest.mark.parametrize("on_lhs", [True, False])
def test_fused_form_requires_matching_nesting(fields, idxs, n, on_lhs):
    table = lgc.Table(lgc.FusedAlias(scratch, n), idxs)
    hard = lgc.Table(a, (i, j, k))
    term = lgc.Query(table, hard) if on_lhs else lgc.Query(hard, table)
    for idx in reversed(fields):
        term = lgc.Fuse(idx, term)
    with pytest.raises(ValueError, match="must match the enclosing Fuse fields"):
        FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())


@pytest.mark.parametrize("view", [scratch, lgc.FusedAlias(scratch, 0)])
@pytest.mark.parametrize("read_idxs", [(i,), (j,)])
@pytest.mark.parametrize("reverse", [False, True])
def test_fused_form_rejects_read_write_across_queries(view, read_idxs, reverse):
    bodies = (
        lgc.Query(lgc.Table(view, (i,)), lgc.Table(a, (i,))),
        lgc.Query(lgc.Table(b, (i,)), lgc.Table(view, read_idxs)),
    )
    term = lgc.Fuse(i, lgc.Plan(bodies[::-1] if reverse else bodies))
    with pytest.raises(ValueError, match="cannot occur on both lhs and rhs"):
        FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())


@pytest.mark.parametrize("query_into", [False, True])
def test_fused_form_rejects_read_write_in_one_query(query_into):
    lhs = lgc.Table(a, (i,))
    rhs = lgc.Table(a, (j,))
    body = (
        lgc.QueryInto(lhs, lgc.Literal(ffuncs.add), rhs)
        if query_into
        else lgc.Query(lhs, rhs)
    )
    with pytest.raises(ValueError, match="cannot occur on both lhs and rhs"):
        FusedForm.validate_inputs(lgc.Fuse(i, body), {}, {}, DenseStatsFactory())


def test_fused_form_rejects_read_write_across_alias_views():
    term = lgc.Fuse(
        i,
        lgc.Plan(
            (
                lgc.Query(lgc.Table(fused, (i,)), lgc.Table(scratch, (i,))),
                lgc.Query(lgc.Table(scratch, (i,)), lgc.Table(a, (i,))),
            )
        ),
    )
    with pytest.raises(ValueError, match="cannot occur on both lhs and rhs"):
        FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())


def test_fused_form_rejects_read_write_in_stored_dimension_of_fused_alias():
    term = lgc.Fuse(
        i,
        lgc.Fuse(
            j,
            lgc.Plan(
                (
                    lgc.Query(lgc.Table(fused, (i, j)), lgc.Table(a, (i, j))),
                    lgc.Query(lgc.Table(b, (i, j)), lgc.Table(fused, (i, j))),
                )
            ),
        ),
    )
    with pytest.raises(ValueError, match="Fuse field j accesses a non-fused dimension"):
        FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())


def test_fused_form_rejects_read_write_across_nested_blocks():
    term = lgc.Fuse(
        i,
        lgc.Plan(
            (
                lgc.Query(lgc.Table(scratch, (i, j)), lgc.Table(a, (i, j))),
                lgc.Fuse(
                    j,
                    lgc.Query(lgc.Table(b, (i, j)), lgc.Table(scratch, (i, j))),
                ),
            )
        ),
    )
    with pytest.raises(ValueError, match="Fuse field i accesses a non-fused dimension"):
        FusedForm.validate_inputs(term, {}, {}, DenseStatsFactory())
