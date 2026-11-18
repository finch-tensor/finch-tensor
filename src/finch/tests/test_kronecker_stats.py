from functools import reduce

import pytest

import numpy as np

import finch as fl
from finch import ffuncs
from finch.autoschedule.cost_models import (
    STORAGE_BASIS,
    DenseLevelOption,
    ElementLevelOption,
    LevelOption,
    PigeonBasis,
    ProtocolBasis,
    SparseHashLevelOption,
    SparseListLevelOption,
)
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.memo_stats import MemoStatsFactory
from finch.autoschedule.util import fuse_loop_order
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
from scripts.fitting.kronecker_stats import (
    KroneckerStatsFactory,
    KroneckerTensor,
    block_shapes,
    kronecker_rmat,
)

BITS = 2
i, j, k = Field("i"), Field("j"), Field("k")
LG = {i: 5, j: 6, k: 3}


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def random_tensor(rng, fields, density=(0.3, 1.0)):
    lg_shape = [LG[idx] for idx in fields]
    seeds = []
    for _ in block_shapes(lg_shape, BITS):
        S = np.ones(())
        for _ in fields:
            s = rng.uniform(-0.8, 0.8)
            S = np.multiply.outer(S, [0.5 - s / 2, 0.5 + s / 2])
        seeds.append(S * rng.uniform(*density))
    return kronecker_rmat(lg_shape, seeds, rng, block_bits=BITS)


def dense(tensor) -> np.ndarray:
    return reduce(lambda x, y: np.kron(x, y), tensor.blocks).astype(bool)


def test_block_shapes_split_digits_from_the_top():
    assert block_shapes([10, 3, 0], 4) == [(16, 8, 1), (16, 1, 1), (4, 1, 1)]
    assert block_shapes([], 4) == []


def test_kronecker_rmat_draws_blocks_and_lists_their_product(rng):
    tensor = random_tensor(rng, (i, j))
    assert [block.shape for block in tensor.blocks] == [(4, 4), (4, 4), (2, 4)]
    support = dense(tensor)
    assert support.shape == tensor.shape == (32, 64)
    assert tensor.nnz == support.sum()
    coords = tensor.coords()
    for got, expected in zip(coords, np.nonzero(support), strict=True):
        np.testing.assert_array_equal(got, expected)
    # The checksum tells draws apart by their blocks alone.
    again = random_tensor(np.random.default_rng(42), (i, j))
    assert again.checksum() == tensor.checksum()
    flipped = [block.copy() for block in tensor.blocks]
    flipped[-1].flat[0] ^= True
    changed = KroneckerTensor(tensor.lg_shape, flipped, BITS)
    assert changed.checksum() != tensor.checksum()


def test_stats_count_expressions_exactly(rng):
    sf = KroneckerStatsFactory(BITS)
    A, B = random_tensor(rng, (i, j)), random_tensor(rng, (j, k))
    a, b = dense(A)[:, :, None], dense(B)[None, :, :]
    sa, sb = sf(A, (i, j)), sf(B, (j, k))
    join = sf.mapjoin(ffuncs.mul, sa, sb)
    union = sf.mapjoin(ffuncs.add, sa, sb)
    assert join.index_order == union.index_order == (i, j, k)
    assert join.nnz == (a & b).sum()
    assert union.nnz == (a | b).sum()
    projected = sf.aggregate(ffuncs.add, None, (j,), union)
    assert projected.nnz == (a | b).any(axis=1).sum()
    assert sf.aggregate(ffuncs.mul, None, (j,), sa).nnz == dense(A).all(axis=1).sum()
    # The largest slice of a union counts each slice by inclusion-exclusion.
    assert union.estimate_non_fill_values(over=(i,)) == (a | b).sum(axis=(1, 2)).max()
    assert union.estimate_non_fill_values(over=(j, k)) == (a | b).sum(axis=0).max()
    assert join.estimate_non_fill_values(over=(k,)) == (a & b).sum(axis=(0, 1)).max()
    reordered = sf.reorder(sa, (j, Field("x"), i))
    assert reordered.dim_sizes[Field("x")] == 1
    assert reordered.nnz == sa.nnz
    assert reordered.estimate_non_fill_values(over=(j,)) == dense(A).sum(0).max()
    relabeled = sf.relabel(sb, (i, j))
    assert relabeled.estimate_non_fill_values(over=(i,)) == dense(B).sum(1).max()
    assert sf.aggregate(ffuncs.add, None, (i, j, k), union).nnz == 1
    # A union with itself is one term.
    assert len(sf.mapjoin(ffuncs.add, sa, sa).terms) == 1
    with pytest.raises(NotImplementedError, match="union"):
        sf.aggregate(ffuncs.mul, None, (j,), union)


def test_stats_take_scalars_and_empty_tensors_and_reject_others():
    sf = KroneckerStatsFactory(BITS)
    assert sf(fl.asarray(np.array(3.0)), ()).nnz == 1
    assert sf(fl.asarray(np.array(0.0)), ()).nnz == 0
    lg_shape = (LG[i], LG[j])
    blocks = [np.zeros(shape, bool) for shape in block_shapes(lg_shape, BITS)]
    empty = sf(KroneckerTensor(lg_shape, blocks, BITS), (i, j))
    assert empty.nnz == empty.estimate_non_fill_values(over=(i,)) == 0
    with pytest.raises(TypeError, match="KroneckerTensor"):
        sf(fl.asarray(np.ones(4)), (i,))


def matmul(protocols):
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    product = MapJoin(
        Literal(ffuncs.mul),
        (Table(a, (i, k), protocols[0]), Table(b, (k, j), protocols[1])),
    )
    query = Query(
        Table(c, (i, j)), Aggregate(Literal(ffuncs.add), Literal(0.0), product, (k,))
    )
    return Plan((fuse_loop_order(query, (i, k, j)), Produces((c,))))


def elementwise_sum(protocols):
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    total = MapJoin(
        Literal(ffuncs.add),
        (Table(a, (i, j), protocols[0]), Table(b, (i, j), protocols[1])),
    )
    query = Query(
        Table(c, (i, j)), Aggregate(Literal(ffuncs.add), Literal(0.0), total, ())
    )
    return Plan((fuse_loop_order(query, (i, j)), Produces((c,))))


@pytest.mark.parametrize(
    "program, fields, protocols",
    [
        # The root of a is dense, so it is followed, and the hash level of b
        # is never galloped.
        (matmul, ((i, k), (k, j)), ((Follow(), Gallop()), (Follow(), Walk()))),
        (matmul, ((i, k), (k, j)), ((Follow(), Walk()), (Gallop(), Follow()))),
        (elementwise_sum, ((i, j), (i, j)), ((Follow(), Walk()), (Walk(), Follow()))),
    ],
)
def test_features_match_exact_stats_of_the_tensors(rng, program, fields, protocols):
    a, b, c = HardAlias("a"), HardAlias("b"), HardAlias("c")
    tensors = {a: random_tensor(rng, fields[0]), b: random_tensor(rng, fields[1])}
    kronecker, exact = KroneckerStatsFactory(BITS), ExactStatsFactory()
    # The fitting pipeline memoizes the Kronecker stats, which counts the same.
    memo = MemoStatsFactory(KroneckerStatsFactory(BITS))
    stats: dict[str, dict[Alias, TensorStats]] = {
        "kronecker": {
            tns: kronecker(tensor, idxs)
            for (tns, tensor), idxs in zip(tensors.items(), fields, strict=True)
        },
        "memo": {
            tns: memo(tensor, idxs)
            for (tns, tensor), idxs in zip(tensors.items(), fields, strict=True)
        },
        "exact": {
            tns: exact(fl.asarray(dense(tensor).astype(float)), idxs)
            for (tns, tensor), idxs in zip(tensors.items(), fields, strict=True)
        },
    }
    lst, hsh = SparseListLevelOption(), SparseHashLevelOption()
    formats: dict[Alias, list[LevelOption]] = {
        a: [DenseLevelOption(), lst, ElementLevelOption()],
        b: [lst, hsh, ElementLevelOption()],
        c: [DenseLevelOption(), hsh, ElementLevelOption()],
    }
    plan = program(protocols)
    for basis in (PigeonBasis(), ProtocolBasis(), STORAGE_BASIS):
        expected = basis.get_features(plan, formats, stats["exact"], exact)
        np.testing.assert_array_equal(
            basis.get_features(plan, formats, stats["kronecker"], kronecker),
            expected,
        )
        np.testing.assert_array_equal(
            basis.get_features(plan, formats, stats["memo"], memo), expected
        )
