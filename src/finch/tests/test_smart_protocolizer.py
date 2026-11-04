import numpy as np

import finch as fl
from finch import ffuncs
from finch.algebra import TensorFType, ftype
from finch.autoschedule import NotationGenerator
from finch.autoschedule.protocolizer import smart_protocolize
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.util import fuse_loop_order
from finch.codegen import NumbaCompiler, NumpyBuffer
from finch.compile import NotationCompiler
from finch.finch_assembly import AssemblySimplify, LowerPackedStructSlots
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Follow,
    Gallop,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
    TensorStats,
    Walk,
)
from finch.symbolic import PostOrderDFS
from finch.tensor import ElementLevel, FiberTensor, SparseListLevel, element


def _sparse_vector(n, idx, vals):
    fmt = element(np.float64(0), ftype(np.float64), ftype(np.intp))
    return FiberTensor(
        SparseListLevel(
            ElementLevel(fmt, NumpyBuffer(np.asarray(vals, dtype=np.float64))),
            np.intp(n),
            NumpyBuffer(np.array([0, len(idx)], dtype=np.intp)),
            NumpyBuffer(np.asarray(idx, dtype=np.intp)),
        )
    )


def _dot(tensors):
    # c[] = sum_i of the product of the tensors at i, looping over i.
    i = Field("i")
    c = HardAlias("c")
    names = tuple(HardAlias(name) for name in "abd"[: len(tensors)])
    product = MapJoin(Literal(ffuncs.mul), tuple(Table(x, (i,)) for x in names))
    rhs = Aggregate(Literal(ffuncs.add), Literal(0.0), product, (i,))
    plan = Plan((fuse_loop_order(Query(Table(c, ()), rhs), (i,)), Produces((c,))))
    output = fl.asarray(np.zeros(()))
    bindings: dict[Alias, TensorFType] = {
        **{x: t.ftype for x, t in zip(names, tensors, strict=True)},
        c: output.ftype,
    }
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {
        x: sf(t, (i,)) for x, t in zip(names, tensors, strict=True)
    }
    return plan, bindings, stats, sf, output


def _protocols(prgm):
    found = {}
    for node in PostOrderDFS(prgm):
        match node:
            case Table(Alias() as tns, _, protocols) if protocols:
                found[tns.name] = protocols[0]
    return found


def test_smart_protocolize_gallops_through_long_lists():
    rng = np.random.default_rng(0)
    n = 10000
    long = _sparse_vector(n, np.sort(rng.choice(n, 1000, replace=False)), np.ones(1000))
    short = _sparse_vector(n, [17, 4242], [2.0, 3.0])
    plan, bindings, stats, sf, _ = _dot((long, short))

    # Galloping through the long list visits only where the short one is.
    prgm = smart_protocolize(plan, bindings, stats, sf)
    assert _protocols(prgm) == {"a": Gallop(), "b": Walk()}

    # Without gallops, the long list is looked up where the short one is,
    # rather than walked.
    prgm = smart_protocolize(plan, bindings, stats, sf, max_gallops=0)
    assert _protocols(prgm) == {"a": Follow(), "b": Walk()}


def test_smart_protocolize_follows_dense_inputs():
    n = 100
    sparse = _sparse_vector(n, [3, 50], [1.0, 2.0])
    dense = fl.asarray(np.arange(n, dtype=np.float64))
    plan, bindings, stats, sf, _ = _dot((sparse, dense))
    # A dense level can only be looked up.
    assert _protocols(smart_protocolize(plan, bindings, stats, sf))["b"] == Follow()


def test_smart_protocolize_compiles_its_choices():
    rng = np.random.default_rng(1)
    n = 10000
    idx = np.sort(rng.choice(n, 1000, replace=False))
    vals = rng.random(1000)
    long = _sparse_vector(n, idx, vals)
    short = _sparse_vector(n, [idx[3], idx[500], 9999], [2.0, 3.0, 4.0])
    plan, bindings, stats, sf, output = _dot((long, short))
    prgm = smart_protocolize(plan, bindings, stats, sf)
    assert Gallop() in _protocols(prgm).values()

    program = NotationGenerator()(prgm, bindings, {}, None)
    compiler = NotationCompiler(
        NumbaCompiler(), ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify())
    )
    (result,) = compiler(program).main(long, short, output)
    dense_long = np.zeros(n)
    dense_long[idx] = vals
    expected = 2.0 * dense_long[idx[3]] + 3.0 * dense_long[idx[500]]
    expected += 4.0 * dense_long[9999]
    np.testing.assert_allclose(result.to_numpy(), expected)
