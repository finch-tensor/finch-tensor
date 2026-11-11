"""
Tensors whose supports are Kronecker products of small blocks, for timing the
programs which fit cost models, and the exact stats of their features. A
dimension of size 2 ** n is split into digits of block_bits bits from the top,
the last of which may have fewer, and block t holds the support over the t-th
digit of each dimension, so a coordinate is in the support when each of its
digits is in its block. Joins, projections, and reorders of such supports act
block by block, and a union is a union of them, so the stats count the
non-fill values of any expression of the inputs exactly from their blocks,
without the tensors.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import cached_property, reduce
from typing import Any

import numpy as np

from finch.algebra import (
    FinchOperator,
    StaticFillFType,
    ftype,
    is_annihilator,
    is_idempotent,
    is_identity,
)
from finch.autoschedule.tensor_stats import BaseTensorStats, BaseTensorStatsFactory
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.finch_logic import Field, StatsFactory

Term = tuple[np.ndarray, ...]


def lg_size(size: float) -> int:
    n = int(size)
    if n <= 0 or n & (n - 1) or n != size:
        raise ValueError(f"Expected a power of two, got {size}")
    return n.bit_length() - 1


def block_shapes(lg_shape: Sequence[int], block_bits: int) -> list[tuple[int, ...]]:
    """
    The shape of each block of a tensor of shape 2 ** lg_shape, in which a
    dimension without a t-th digit has extent 1.
    """
    n_blocks = max((-(-lg // block_bits) for lg in lg_shape), default=0)
    return [
        tuple(2 ** min(block_bits, max(lg - t * block_bits, 0)) for lg in lg_shape)
        for t in range(n_blocks)
    ]


def rmat_block(S, lg_block: Sequence[int], rng) -> np.ndarray:
    """
    A boolean array of shape 2 ** lg_block whose S.sum() * 2 ** sum(lg_block)
    true cells are the first distinct cells which balls dropped by R-MAT with
    the seed S land in.
    """
    S = np.asarray(S, dtype=float)
    shape = tuple(2**lg for lg in lg_block)
    size = math.prod(shape)
    nnz = round(S.sum() * size)
    block = np.zeros(size, dtype=bool)
    if nnz:
        # At each level a ball picks an entry of S, and each axis with bits
        # left takes that entry's bit along it.
        P = np.ones(())
        bits = []
        for level in range(max(lg_block, default=0)):
            done = tuple(d for d, lg in enumerate(lg_block) if lg <= level)
            P = np.multiply.outer(P, S.sum(axis=done) / S.sum())
            bits += [(d, level) for d, lg in enumerate(lg_block) if level < lg]
        # The bits of an axis, from the top, make its coordinate.
        order = sorted(range(len(bits)), key=bits.__getitem__)
        P = P.transpose(order).reshape(shape)
        # The order in which balls first land in cells is the order of
        # exponential draws over the probabilities of the cells.
        with np.errstate(divide="ignore"):
            keys = rng.exponential(size=size) / P.ravel()
        block[np.argpartition(keys, nnz - 1)[:nnz]] = True
    return block.reshape(shape)


class KroneckerTensor:
    """
    A tensor of shape 2 ** lg_shape whose support is the Kronecker product of
    blocks, boolean arrays of the shapes block_shapes gives, and whose values
    are of dtype, indexed by idx_type.
    """

    def __init__(
        self,
        lg_shape: Sequence[int],
        blocks: Sequence[np.ndarray],
        block_bits: int,
        dtype=np.float64,
        idx_type=np.intp,
    ):
        self.lg_shape = tuple(lg_shape)
        self.blocks = tuple(np.asarray(block, dtype=bool) for block in blocks)
        self.block_bits = block_bits
        self.dtype = np.dtype(dtype).type
        self.idx_type = idx_type
        expected = block_shapes(self.lg_shape, block_bits)
        if [block.shape for block in self.blocks] != expected:
            raise ValueError(f"Expected blocks of shapes {expected}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(2**lg for lg in self.lg_shape)

    @property
    def ndim(self) -> int:
        return len(self.lg_shape)

    @property
    def nnz(self) -> int:
        return math.prod(int(block.sum()) for block in self.blocks)

    def checksum(self) -> str:
        """A hash of the blocks, which tells whether two draws of them agree."""
        digest = hashlib.sha256()
        for block in self.blocks:
            digest.update(np.array(block.shape, dtype=np.int64).tobytes())
            digest.update(np.packbits(block).tobytes())
        return digest.hexdigest()

    def transpose(self, perm: Sequence[int]) -> KroneckerTensor:
        """The tensor which stores its dimensions in the order of perm."""
        if list(perm) == list(range(self.ndim)):
            return self
        return KroneckerTensor(
            [self.lg_shape[d] for d in perm],
            [block.transpose(perm) for block in self.blocks],
            self.block_bits,
            self.dtype,
            self.idx_type,
        )

    def coords(self) -> tuple[np.ndarray, ...]:
        """The coordinates of the support, in lexicographic order."""
        if not self.lg_shape:
            return ()
        # The flat index of a cell is the sum of the digits of its
        # coordinates, each shifted past the bits below it in its dimension
        # and the bits of the dimensions after it.
        after = [sum(self.lg_shape[d + 1 :]) for d in range(self.ndim)]
        flat = np.zeros(1, dtype=np.int64)
        for t, block in enumerate(self.blocks):
            below = [max(lg - (t + 1) * self.block_bits, 0) for lg in self.lg_shape]
            digits = np.nonzero(block)
            offsets = sum(
                digits[d].astype(np.int64) << (below[d] + after[d])
                for d in range(self.ndim)
            )
            flat = np.add.outer(flat, offsets).ravel()
        return np.unravel_index(np.sort(flat), self.shape)


def kronecker_rmat(
    lg_shape: Sequence[int],
    seeds: Sequence[Any],
    rng,
    block_bits: int,
    dtype=np.float64,
    idx_type=np.intp,
) -> KroneckerTensor:
    """
    A KroneckerTensor of shape 2 ** lg_shape whose blocks are drawn by
    rmat_block, each with its own seed of seeds, whose sum is the density of
    its block.
    """
    shapes = block_shapes(lg_shape, block_bits)
    if len(seeds) != len(shapes):
        raise ValueError(f"Expected a seed for each of {len(shapes)} blocks")
    blocks = [
        rmat_block(S, [lg_size(n) for n in shape], rng)
        for S, shape in zip(seeds, shapes, strict=True)
    ]
    return KroneckerTensor(lg_shape, blocks, block_bits, dtype, idx_type)


def support_tensor(stats: KroneckerStats, dtype, idx_type) -> KroneckerTensor:
    """
    The KroneckerTensor of the support of stats, stored in its index order,
    which must be one term or none, as the supports of sums of products are.
    """
    if len(stats.terms) > 1:
        raise ValueError(f"Expected one term, got a union of {len(stats.terms)}")
    lg_shape = [lg_size(stats.dim_sizes[idx]) for idx in stats.index_order]
    shapes = block_shapes(lg_shape, stats.block_bits)
    # A block of extent 1 along a digit is the same along all of it.
    blocks = [
        np.broadcast_to(block, shape)
        for block, shape in zip(
            stats.terms[0] if stats.terms else [np.zeros(s, bool) for s in shapes],
            shapes,
            strict=True,
        )
    ]
    return KroneckerTensor(lg_shape, blocks, stats.block_bits, dtype, idx_type)


def stats_block_shapes(stats: BaseTensorStats, block_bits: int):
    lg_shape = [lg_size(stats.dim_sizes[idx]) for idx in stats.index_order]
    return block_shapes(lg_shape, block_bits)


def meet(terms: Iterable[Term]) -> Term:
    """The intersection of terms, block by block."""
    return tuple(reduce(np.logical_and, blocks) for blocks in zip(*terms, strict=True))


def count(term: Term, shapes) -> int:
    # An axis of extent 1 where the block's digit has more is the same along it.
    return math.prod(
        int(block.sum())
        * math.prod(n // m for n, m in zip(shape, block.shape, strict=True))
        for block, shape in zip(term, shapes, strict=True)
    )


def slice_counts(block: np.ndarray, shape, fixed: tuple[int, ...]) -> np.ndarray:
    """The count of each slice of block which fixes the axes of fixed, flattened."""
    free = tuple(d for d in range(block.ndim) if d not in fixed)
    scale = math.prod(shape[d] // block.shape[d] for d in free)
    counts = block.sum(axis=free, dtype=np.int64) * scale
    return np.broadcast_to(counts, tuple(shape[d] for d in fixed)).ravel()


class KroneckerStats(NumericStats):
    """
    The support of a tensor as a union of terms, each the Kronecker product of
    one block for each digit of its fields, as KroneckerTensor stores it. A
    block has an axis for each field of index_order, of the extent of the
    field's digit, or 1 where the block is the same along the digit.
    """

    def __init__(self, base: BaseTensorStats, terms: Iterable[Term], block_bits: int):
        super().__init__(base)
        self.block_bits = block_bits
        # Repeated terms and empty ones add nothing to the union.
        unique: dict[tuple, Term] = {}
        for term in terms:
            if all(block.any() for block in term):
                key = tuple((block.shape, block.tobytes()) for block in term)
                unique.setdefault(key, term)
        self.terms = list(unique.values())

    @property
    def block_shapes(self) -> list[tuple[int, ...]]:
        return stats_block_shapes(self, self.block_bits)

    def subsets(self) -> list[tuple[int, Term]]:
        """The sign and intersection of each subset of terms, by inclusion-exclusion."""
        return [
            ((-1) ** (k + 1), meet(subset))
            for k in range(1, len(self.terms) + 1)
            for subset in itertools.combinations(self.terms, k)
        ]

    @cached_property
    def nnz(self) -> float:
        shapes = self.block_shapes
        return float(sum(sign * count(term, shapes) for sign, term in self.subsets()))

    def estimate_non_fill_values(self, over: Iterable[Field] = ()) -> float:
        fixed = tuple(d for d, idx in enumerate(self.index_order) if idx in set(over))
        if not fixed or not self.terms:
            return self.nnz
        # The count of a slice is a signed sum over subsets of products of the
        # counts of its digits in each block, so the largest is found among
        # the distinct products of the distinct counts of each block.
        subsets = self.subsets()
        signs = np.array([sign for sign, _ in subsets], dtype=np.int64)
        products = np.ones((1, len(subsets)), dtype=np.int64)
        for t, shape in enumerate(self.block_shapes):
            table = np.stack(
                [slice_counts(term[t], shape, fixed) for _, term in subsets], axis=1
            )
            products = products[:, None, :] * np.unique(table, axis=0)[None, :, :]
            products = np.unique(products.reshape(-1, len(subsets)), axis=0)
            if len(subsets) == 1:
                products = products.max(axis=0, keepdims=True)
        return float(np.max(products @ signs))

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        return np.log2(np.array(sizes + [self.nnz + 1]))


class KroneckerStatsFactory(
    BaseTensorStatsFactory[KroneckerStats], StatsFactory[KroneckerStats]
):
    """The exact stats of KroneckerTensors with digits of block_bits, and scalars."""

    def __init__(self, block_bits: int):
        super().__init__(KroneckerStats)
        self.block_bits = block_bits

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> KroneckerStats:
        match tensor:
            case KroneckerTensor(block_bits=self.block_bits):
                dim_sizes = OrderedDict(
                    (idx, float(n)) for idx, n in zip(fields, tensor.shape, strict=True)
                )
                base = BaseTensorStats(
                    fields,
                    dim_sizes,
                    StaticFillFType(tensor.dtype(0)),
                    dict.fromkeys(fields, ftype(tensor.idx_type)),
                )
                return KroneckerStats(base, [tensor.blocks], self.block_bits)
            case _ if tensor.ndim == 0:
                base = super().__call__(tensor, fields)
                terms = [()] if tensor.item() != tensor.fill_value else []
                return KroneckerStats(base, terms, self.block_bits)
        raise TypeError(
            f"Expected a scalar or a KroneckerTensor with {self.block_bits} bit "
            f"digits, got {tensor}"
        )

    def aligned(self, stats: KroneckerStats, base: BaseTensorStats) -> list[Term]:
        """
        The terms of stats with an axis for each field of base, and a block for
        each of its digits.
        """
        ndim = len(base.index_order)
        perm = [
            stats.index_order.index(idx)
            for idx in base.index_order
            if idx in stats.index_order
        ]
        n_blocks = len(stats_block_shapes(base, self.block_bits))
        terms = []
        for term in stats.terms:
            blocks = []
            for t in range(n_blocks):
                if t < len(term):
                    shape = tuple(
                        term[t].shape[stats.index_order.index(idx)]
                        if idx in stats.index_order
                        else 1
                        for idx in base.index_order
                    )
                    blocks.append(term[t].transpose(perm).reshape(shape))
                else:
                    blocks.append(np.ones((1,) * ndim, dtype=bool))
            terms.append(tuple(blocks))
        return terms

    def _mapjoin_join(
        self, op: FinchOperator, *join_args: KroneckerStats
    ) -> KroneckerStats:
        base = self._mapjoin_defs(op, *join_args)
        terms = [
            meet(combination)
            for combination in itertools.product(
                *(self.aligned(s, base) for s in join_args)
            )
        ]
        return KroneckerStats(base, terms, self.block_bits)

    def _mapjoin_union(
        self, op: FinchOperator, *union_args: KroneckerStats
    ) -> KroneckerStats:
        base = self._mapjoin_defs(op, *union_args)
        terms = [term for s in union_args for term in self.aligned(s, base)]
        return KroneckerStats(base, terms, self.block_bits)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: KroneckerStats,
    ) -> KroneckerStats:
        base = self.aggregate_def(op, init, reduce_indices, stats)
        n_blocks = len(stats_block_shapes(base, self.block_bits))
        ndim = len(base.index_order)
        axes = tuple(
            d for d, idx in enumerate(stats.index_order) if idx in reduce_indices
        )
        f = stats.fill_type
        # As in ExactStats, an output is non-fill where any of its inputs is,
        # unless the init joins each fill of a sum, so every output is, or the
        # fill annihilates op, so an output is non-fill where all inputs are.
        if is_identity(op.ftype, f) or not is_annihilator(op.ftype, f):
            if (
                not is_identity(op.ftype, f)
                and init is not None
                and not is_idempotent(op.ftype)
            ):
                full = tuple(np.ones((1,) * ndim, dtype=bool) for _ in range(n_blocks))
                return KroneckerStats(base, [full], self.block_bits)
            reduced = [tuple(b.any(axis=axes) for b in term) for term in stats.terms]
        else:
            if len(stats.terms) > 1:
                raise NotImplementedError(
                    "An aggregate which needs all inputs non-fill can't reduce a union"
                )
            reduced = [tuple(b.all(axis=axes) for b in term) for term in stats.terms]
        # The blocks of digits of only the reduced fields are now single cells,
        # which hold for the whole term or none of it.
        terms = [
            term[:n_blocks] for term in reduced if all(b.all() for b in term[n_blocks:])
        ]
        return KroneckerStats(base, terms, self.block_bits)

    def relabel(
        self, stats: KroneckerStats, relabel_indices: tuple[Field, ...]
    ) -> KroneckerStats:
        base = self.relabel_def(stats, relabel_indices)
        return KroneckerStats(base, stats.terms, self.block_bits)

    def reorder(
        self, stats: KroneckerStats, reorder_indices: tuple[Field, ...]
    ) -> KroneckerStats:
        base = self.reorder_def(stats, reorder_indices)
        # Only fields of size 1, which have no digits, are dropped.
        dropped = tuple(
            d for d, idx in enumerate(stats.index_order) if idx not in reorder_indices
        )
        kept = [idx for idx in stats.index_order if idx in reorder_indices]
        squeezed = KroneckerStats(
            BaseTensorStats.from_fields(
                kept,
                OrderedDict((idx, stats.dim_sizes[idx]) for idx in kept),
                stats.fill_type,
            ),
            [tuple(np.squeeze(b, axis=dropped) for b in term) for term in stats.terms],
            self.block_bits,
        )
        return KroneckerStats(base, self.aligned(squeezed, base), self.block_bits)
