"""
Sampling statistics whose samples are NumPy coordinate lists instead of Finch
tensors.

``SamplingStatsFactory`` keeps each sample (a "sketch") as a Finch tensor and
combines sketches with Finch kernels, so every new sketch-program shape costs a
Julia compile; on subgraph queries those compiles dominate planning. Here an
input tensor is read into coordinates once, with Finch.jl's ``ffindnz`` (one
small kernel per input format), and every join, union, aggregate, relabel,
reorder and remask after that runs in NumPy.

A sample is ``CooSample(fields, coords, vals)``: ``coords`` holds one row of
coordinates per field and one column per stored entry, every column distinct.
Sampling keeps a coordinate of a field when a hash of the field's seed and the
coordinate falls below the field's probability, so a lower probability under
the same seed keeps a subset, and joins of samples drawn with one seed per
field keep matching coordinates.

The estimators are ``SamplingStats``'s: ``CooSamplingStats`` only replaces how
the sample is scanned and how sampling coverage is measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from finch.algebra import FinchOperator, is_annihilator, is_identity
from finch.finch_logic import Field

from .sampling_stats import SamplingStats, SamplingStatsFactory
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_STEP = np.uint64(0xD1B54A32D192ED03)


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64's finalizer, elementwise on uint64 (overflow wraps)."""
    x = x + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MIX1
    x = (x ^ (x >> np.uint64(27))) * _MIX2
    return x ^ (x >> np.uint64(31))


def _uniform(seed: int, idx: np.ndarray) -> np.ndarray:
    """A value in [0, 1) for each coordinate, fixed by the seed."""
    with np.errstate(over="ignore"):
        h = _mix(np.uint64(seed) + idx.astype(np.uint64) * _STEP)
    return (h >> np.uint64(11)).astype(np.float64) * 2.0**-53


def _keep(seed: int, idx: np.ndarray, prob: float) -> np.ndarray:
    if prob >= 1.0:
        return np.ones(idx.shape, dtype=bool)
    return _uniform(seed, idx) < prob


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooSample:
    fields: tuple[Field, ...]
    coords: np.ndarray  # (len(fields), nnz) int64, distinct columns
    vals: np.ndarray  # (nnz,) float64

    @property
    def nnz(self) -> int:
        return int(self.vals.shape[0])

    def row(self, field: Field) -> np.ndarray:
        return self.coords[self.fields.index(field)]

    def take(self, sel) -> CooSample:
        return CooSample(self.fields, self.coords[:, sel], self.vals[sel])

    def nonzero(self) -> CooSample:
        return self.take(self.vals != 0)

    def permute(self, fields: tuple[Field, ...]) -> CooSample:
        """The same entries with rows in the order of ``fields``, which must
        hold exactly this sample's fields."""
        order = [self.fields.index(f) for f in fields]
        return CooSample(tuple(fields), self.coords[order], self.vals)


def _empty(fields: tuple[Field, ...]) -> CooSample:
    return CooSample(
        tuple(fields),
        np.empty((len(fields), 0), dtype=np.int64),
        np.empty(0, dtype=np.float64),
    )


def _group_ids(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct columns of ``rows`` and, for each column, its group's index."""
    if rows.shape[0] == 0:
        return np.empty((0, 1), dtype=np.int64), np.zeros(rows.shape[1], np.int64)
    keys, inverse = np.unique(rows, axis=1, return_inverse=True)
    return keys, inverse.reshape(-1)


def _group_sum(
    fields: tuple[Field, ...], coords: np.ndarray, vals: np.ndarray
) -> CooSample:
    """Entries with equal coordinates summed; zero sums dropped."""
    keys, inverse = _group_ids(coords)
    sums = np.bincount(inverse, weights=vals, minlength=keys.shape[1])
    keep = sums != 0
    return CooSample(fields, keys[:, keep], sums[keep])


def _join_plan(left: CooSample, right: CooSample, shared: tuple[Field, ...]):
    """For left * right on the shared fields: the left and right entry of each
    output entry, without materializing the output yet. Its length is the
    output size, so this also predicts it."""
    if not shared:
        li = np.repeat(np.arange(left.nnz), right.nnz)
        ri = np.tile(np.arange(right.nnz), left.nnz)
        return li, ri
    lkey = np.stack([left.row(f) for f in shared])
    rkey = np.stack([right.row(f) for f in shared])
    _, inverse = _group_ids(np.concatenate([lkey, rkey], axis=1))
    kl, kr = inverse[: left.nnz], inverse[left.nnz :]
    groups = int(inverse.max()) + 1 if inverse.size else 0
    order = np.argsort(kr, kind="stable")
    counts = np.bincount(kr, minlength=groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if groups else counts
    matches = counts[kl]
    li = np.repeat(np.arange(left.nnz), matches)
    first = np.repeat(np.cumsum(matches) - matches, matches)
    ri = order[np.repeat(starts[kl], matches) + (np.arange(li.size) - first)]
    return li, ri


def _join(left: CooSample, right: CooSample, li, ri) -> CooSample:
    extra = tuple(f for f in right.fields if f not in left.fields)
    coords = np.concatenate(
        [left.coords[:, li], np.stack([right.row(f)[ri] for f in extra])]
        if extra
        else [left.coords[:, li]],
        axis=0,
    )
    return CooSample(left.fields + extra, coords, left.vals[li] * right.vals[ri])


def coo_of_tensor(tensor: Any, fields: tuple[Field, ...]) -> CooSample:
    """The non-fill entries of any finch tensor, read through Finch.jl."""
    from finch.compile_jl.interop import tensor_to_jl
    from finch.compile_jl.julia import jl
    from finch.compile_jl.runtime import JuliaOwnedTensor

    raw = (
        tensor.raw_julia_obj
        if isinstance(tensor, JuliaOwnedTensor)
        else tensor_to_jl(tensor)
    )
    *idx, val = (np.asarray(x) for x in jl.Finch.ffindnz(raw))
    # Finch.jl's axes are finch-lite's reversed, and its coordinates 1-based.
    coords = (
        np.stack([i.astype(np.int64) - 1 for i in reversed(idx)])
        if idx
        else np.empty((0, val.shape[0]), dtype=np.int64)
    )
    vals = np.asarray(val)
    keep = vals != tensor.fill_value
    # Like SamplingStatsFactory's sketches, every non-fill entry counts once.
    return CooSample(tuple(fields), coords[:, keep], np.ones(int(keep.sum())))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class CooSamplingStats(SamplingStats):
    """SamplingStats whose sample is a CooSample (``self.sample``)."""

    sample: CooSample

    def __init__(self, base: BaseTensorStats, sample: CooSample, **kwargs):
        super().__init__(base, sketch=None, **kwargs)  # type: ignore[arg-type]
        self.sample = sample

    def scan(self, needs_freq: bool) -> tuple[float, float, float, dict | None]:
        if self.scan_cache is None or (needs_freq and self.scan_cache[3] is None):
            vals, cts = np.unique_counts(self.sample.vals)
            n = float(np.dot(vals.astype(float), cts))
            d_n = float(cts[vals > 0].sum())
            f_1 = float(cts[vals == 1].sum())
            frequencies = (
                {int(v): float(c) for v, c in zip(vals, cts, strict=True) if v > 0}
                if needs_freq
                else None
            )
            self.scan_cache = (n, d_n, f_1, frequencies or None)
        return self.scan_cache

    def coverage_correction(self) -> float:
        _, d_n_raw, _, _ = self.scan(needs_freq=False)
        coverage = 1.0
        for field, prob in zip(self.index_order, self.sample_probs, strict=True):
            size = int(self.dim_sizes[field])
            seed = self.seeds_ref.get((field, size))
            if seed is None or size == 0:
                continue
            fraction = float(_keep(seed, np.arange(size), prob).mean())
            if fraction > 0:
                coverage *= fraction
        return d_n_raw / coverage if coverage > 0 else d_n_raw


class CooSamplingStatsFactory(SamplingStatsFactory):
    """SamplingStatsFactory over CooSamples: the same estimates, no kernels
    after reading the inputs. Pairwise joins are capped: a join predicted to
    exceed the budget samples its fields down before it is materialized."""

    def __init__(self, sample_nnz: int = 10_000, estimator: str = "sh3"):
        super().__init__(sample_nnz, estimator)
        BaseTensorStatsFactory.__init__(self, CooSamplingStats)

    # --- sampling -------------------------------------------------------

    def _seed(self, field: Field, size: int) -> int:
        key = (field, size)
        if key not in self._seeds:
            self._seeds[key] = int(self._rng.integers(0, 1 << 64, dtype=np.uint64))
        return self._seeds[key]

    def _downsample(self, sample, old_probs, new_probs, dim_sizes, seeds):
        """Entries of the sample whose coordinate survives each field's lowered
        probability."""
        keep = np.ones(sample.nnz, dtype=bool)
        for f in sample.fields:
            if new_probs[f] != old_probs[f]:
                size = int(dim_sizes[f])
                seed = seeds.get((f, size))
                if seed is None:
                    seed = self._seed(f, size)
                    seeds[f, size] = seed
                keep &= _keep(seed, sample.row(f), new_probs[f])
        return sample if keep.all() else sample.take(keep)

    def _make(self, base, sample, probs, **kwargs) -> CooSamplingStats:
        return CooSamplingStats(
            base,
            sample,
            sample_probs=[probs[f] for f in base.index_order],
            estimator=self.estimator,
            **kwargs,
        )

    def _with_probs(self, stats: CooSamplingStats, probs: dict) -> CooSamplingStats:
        old = dict(zip(stats.index_order, stats.sample_probs, strict=True))
        if all(probs[f] == old[f] for f in stats.index_order):
            return stats
        result = self.copy(stats)
        result.sample = self._downsample(
            stats.sample, old, probs, stats.dim_sizes, result.seeds_ref
        )
        result.sample_probs = [probs[f] for f in stats.index_order]
        result.scan_cache = None
        return result

    def remask(self, stats: CooSamplingStats) -> CooSamplingStats:
        """Halve the probability of the field with the most distinct sampled
        coordinates until the sample fits the budget."""
        if math.prod(int(stats.dim_sizes[f]) for f in stats.index_order) <= (
            self.sample_nnz
        ):
            return stats
        while stats.sample.nnz > self.sample_nnz and stats.index_order:
            sample = stats.sample
            field = max(stats.index_order, key=lambda f: np.unique(sample.row(f)).size)
            probs = dict(zip(stats.index_order, stats.sample_probs, strict=True))
            probs[field] *= 0.5
            stats = self._with_probs(stats, probs)
        return stats

    # --- the factory interface -----------------------------------------

    def __call__(
        self,
        tensor: Any,
        fields: tuple[Field, ...],
        sample_probs: list[float] | None = None,
    ) -> CooSamplingStats:
        base = BaseTensorStatsFactory.__call__(self, tensor, fields)
        probs = dict(zip(fields, sample_probs or [1.0] * len(fields), strict=True))
        for f in fields:
            self._seed(f, int(base.dim_sizes[f]))
        sample = coo_of_tensor(tensor, tuple(fields))
        ones = dict.fromkeys(fields, 1.0)
        sample = self._downsample(sample, ones, probs, base.dim_sizes, self._seeds)
        return self.remask(self._make(base, sample, probs, seeds_ref=self._seeds))

    def _align(self, *args: CooSamplingStats):
        probs: dict[Field, float] = {}
        for arg in args:
            for f, p in zip(arg.index_order, arg.sample_probs, strict=True):
                probs[f] = min(probs.get(f, p), p)
        aligned = tuple(
            self._with_probs(arg, {f: probs[f] for f in arg.index_order})
            for arg in args
        )
        return aligned, probs

    def _mapjoin_join(self, op: FinchOperator, *join_args: CooSamplingStats):
        """N(C)_i = N(A)_j * N(B)_k, joined two samples at a time."""
        if len(join_args) == 1:
            return self.remask(self.copy(join_args[0]))
        base = BaseTensorStatsFactory._mapjoin_defs(op, *join_args)
        join_args, probs = self._align(*join_args)
        seeds = self._seeds
        remaining = [arg.sample for arg in join_args]
        result = remaining.pop(
            max(range(len(remaining)), key=lambda n: len(remaining[n].fields))
        )
        while remaining:
            joined = set(result.fields)
            other = remaining.pop(
                max(
                    range(len(remaining)),
                    key=lambda n: (
                        len(joined & set(remaining[n].fields)),
                        len(remaining[n].fields),
                    ),
                )
            )
            left, right = (
                (result, other)
                if len(result.fields) >= len(other.fields)
                else (other, result)
            )
            shared = tuple(f for f in left.fields if f in set(right.fields))
            li, ri = _join_plan(left, right, shared)
            if li.size > self.sample_nnz:
                # Sample the fields down before materializing the join: each
                # halving halves the expected output.
                halvings = math.ceil(math.log2(li.size / self.sample_nnz))
                fields = list(dict.fromkeys(left.fields + right.fields))
                new = dict(probs)
                for _ in range(halvings):
                    f = max(fields, key=lambda f: (new[f], base.dim_sizes[f]))
                    new[f] *= 0.5
                left = self._downsample(left, probs, new, base.dim_sizes, seeds)
                right = self._downsample(right, probs, new, base.dim_sizes, seeds)
                probs = new
                li, ri = _join_plan(left, right, shared)
            result = _join(left, right, li, ri)
        return self.remask(
            self._make(
                base,
                result.permute(tuple(base.index_order)),
                probs,
                remainder_size=math.prod(a.remainder_size for a in join_args),
                remainder_prob=math.prod(a.remainder_prob for a in join_args),
                seeds_ref=self._seeds,
            )
        )

    def _mapjoin_union(self, op: FinchOperator, *union_args: CooSamplingStats):
        """
        N(C)_i = sum_{juk\\i}[N(A)_j *prod_{l in k\\j}n(B)_l +
        N(B)_k *prod_{l in j\\k}n(A)_l ] - N(A)_j*N(B)_k
        """
        base = BaseTensorStatsFactory._mapjoin_defs(op, *union_args)
        union_args, probs = self._align(*union_args)
        out = tuple(base.index_order)
        out_set = set(out)
        coords, vals = [], []
        for arg in union_args:
            arg_fields = set(arg.index_order)
            other_free = 1.0
            for other in union_args:
                if other is arg:
                    continue
                for f in other.index_order:
                    if f not in arg_fields and f not in out_set:
                        other_free *= other.dim_sizes.get(f, 1.0)
                other_free *= other.remainder_size
            # Broadcast over the output fields the argument lacks, at their
            # sampled coordinates.
            sample = arg.sample
            for f in out:
                if f in arg_fields:
                    continue
                size = int(base.dim_sizes[f])
                kept = np.flatnonzero(
                    _keep(self._seed(f, size), np.arange(size), probs[f])
                )
                sample = CooSample(
                    sample.fields + (f,),
                    np.concatenate(
                        [
                            np.repeat(sample.coords, kept.size, axis=1),
                            np.tile(kept, sample.nnz)[None, :],
                        ]
                    ),
                    np.repeat(sample.vals, kept.size),
                )
            sample = sample.permute(out)
            coords.append(sample.coords)
            vals.append(sample.vals * other_free)
        if len(union_args) >= 2:
            inter = union_args[0].sample
            for arg in union_args[1:]:
                shared = tuple(f for f in inter.fields if f in set(arg.sample.fields))
                inter = _join(inter, arg.sample, *_join_plan(inter, arg.sample, shared))
            inter = inter.permute(out) if set(inter.fields) == out_set else None
            if inter is not None:
                coords.append(inter.coords)
                vals.append(-inter.vals)
        sample = _group_sum(out, np.concatenate(coords, axis=1), np.concatenate(vals))
        return self.remask(
            self._make(
                base,
                sample,
                probs,
                remainder_size=math.prod(a.remainder_size for a in union_args),
                remainder_prob=math.prod(a.remainder_prob for a in union_args),
                seeds_ref=self._seeds,
            )
        )

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: CooSamplingStats,
    ):
        """
        op is identity on fill : N(B)_i = sum_j N(A)_j
        op annihilates fill: N(B)_i = prod(l in k)n_l * min_k 1[N(A)_j > 0]
        otherwise : N(B)_i = prod(l in k)n_l * exists(N(A)_j)
        """
        base = self.aggregate_def(op, init, reduce_indices, stats)
        reduce_set = set(reduce_indices) & set(stats.index_order)
        keep_fields = tuple(f for f in stats.index_order if f not in reduce_set)
        sample = stats.sample.nonzero()
        rows = (
            np.stack([sample.row(f) for f in keep_fields])
            if keep_fields
            else (np.empty((0, sample.nnz), dtype=np.int64))
        )
        prod_n = math.prod(int(stats.dim_sizes[f]) for f in reduce_set)
        if is_identity(op.ftype, stats.fill_type):
            result = _group_sum(keep_fields, rows, sample.vals)
        else:
            keys, inverse = _group_ids(rows)
            present = np.bincount(inverse, minlength=keys.shape[1])
            if is_annihilator(op.ftype, stats.fill_type):
                # The minimum over the reduced fields is nonzero only where
                # every reduced coordinate is present.
                full = present >= prod_n
            else:
                full = present > 0
            result = CooSample(
                keep_fields, keys[:, full], np.full(int(full.sum()), float(prod_n))
            )
        probs = dict(zip(stats.index_order, stats.sample_probs, strict=True))
        return self.remask(
            self._make(
                base,
                result.permute(tuple(base.index_order)),
                probs,
                remainder_size=stats.remainder_size
                * math.prod(stats.dim_sizes[f] for f in reduce_set),
                remainder_prob=stats.remainder_prob
                * math.prod(probs[f] for f in reduce_set),
                seeds_ref=stats.seeds_ref,
            )
        )

    def relabel(self, stats: CooSamplingStats, relabel_indices: tuple[Field, ...]):
        base = self.relabel_def(stats, relabel_indices)
        seeds = dict(stats.seeds_ref)
        for old, new in zip(stats.index_order, relabel_indices, strict=True):
            size = int(stats.dim_sizes[old])
            if (old, size) in stats.seeds_ref:
                seeds[new, size] = stats.seeds_ref[old, size]
        sample = CooSample(
            tuple(relabel_indices), stats.sample.coords, stats.sample.vals
        )
        probs = dict(zip(relabel_indices, stats.sample_probs, strict=True))
        return self.remask(
            self._make(
                base,
                sample,
                probs,
                remainder_size=stats.remainder_size,
                remainder_prob=stats.remainder_prob,
                seeds_ref=seeds,
            )
        )

    def reorder(self, stats: CooSamplingStats, reorder_indices: tuple[Field, ...]):
        """Permute fields; dropped fields have size 1 and added ones size 1, so
        their coordinate is always 0."""
        base = self.reorder_def(stats, reorder_indices)
        sample = stats.sample
        rows = [
            sample.row(f) if f in sample.fields else np.zeros(sample.nnz, np.int64)
            for f in reorder_indices
        ]
        coords = np.stack(rows) if rows else np.empty((0, sample.nnz), dtype=np.int64)
        old = dict(zip(stats.index_order, stats.sample_probs, strict=True))
        probs = {f: old.get(f, 1.0) for f in reorder_indices}
        return self.remask(
            self._make(
                base,
                CooSample(tuple(reorder_indices), coords, sample.vals),
                probs,
                remainder_size=stats.remainder_size,
                remainder_prob=stats.remainder_prob,
                seeds_ref=stats.seeds_ref,
            )
        )
