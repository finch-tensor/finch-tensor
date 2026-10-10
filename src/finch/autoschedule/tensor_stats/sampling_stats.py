from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from finch.algebra import FinchOperator, ffuncs, is_annihilator, is_identity
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
from finch.finch_logic.nodes import LogicExpression
from finch.finch_logic.tensor_stats import StatsFactory
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    RandomMaskTensor,
    SparseByteMapLevel,
    SparseCOOLevel,
    SparseHashLevel,
    SparseListLevel,
)

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory
from .util import get_lp_norms

SAMPLING_ESTIMATORS = ("uj1", "sj1", "uj2", "schlosser", "sh2", "sh3", "good1", "silly")


def mask_table(field: Field, mask: RandomMaskTensor) -> Table:
    return Table(Literal(mask), (field,))


def compute_sketch(sketch: LogicExpression) -> Table:
    from finch.autoschedule.default_schedulers import NON_RECURSIVE_SCHEDULER

    fields = sketch.fields()
    out = HardAlias("sketch_out")
    prgm = Plan((Query(Table(out, fields), sketch), Produces((out,))))
    (result,) = NON_RECURSIVE_SCHEDULER(prgm)
    return Table(Literal(result), fields)


def _dgood1(d_n: float, frequencies: dict | None, n: float, N: float) -> float:

    if d_n == 0:
        return 0.0
    if not frequencies:
        return d_n
    if n >= N:
        return d_n
    max_i = int(max(frequencies.keys()))
    total = d_n
    coef = 1.0
    for i in range(1, max_i + 1):
        j = i - 1
        denom = n - j
        if denom <= 0:
            break
        coef *= (N - n + j) / denom
        f_i = frequencies.get(i, 0.0)
        if f_i:
            sign = 1.0 if (i % 2) == 1 else -1.0
            total += sign * coef * f_i
        if not math.isfinite(total) or not math.isfinite(coef):
            return d_n

    if not (d_n <= total <= N):
        return d_n
    return float(total)


def _dsilly(d_n: float, q: float) -> float:
    if d_n == 0:
        return 0.0
    if q <= 0:
        return d_n
    return float(d_n / min(q, 1.0))


def _duj1(d_n: float, f_1: float, q: float, n: float) -> float:
    """
    Using un-smoothened first order jackknife estimator
    D_uj1 = (1-(1-q)*f_1/n)^{-1} * d_n
    """
    if d_n == 0:
        return 0.0
    if q <= 0 or n <= 0:
        return d_n
    denom = (n - f_1 + q * f_1) / n
    if denom <= 0:
        return d_n
    return d_n / denom


def _dsj1(d_n: float, q: float, N: float) -> float:
    """
    Smoothened first order jacknife
    D * (1-(1-q)^(N/D)) = d_n

    d_n = positions observed in the sample
    q = product of reduced-dimension sampling probabilities
    N = population size (nonzero contributions)
    """
    if d_n == 0:
        return 0.0
    if q <= 0.0 or q >= 1.0:
        return d_n

    def equation(D):
        # D must be >=dn and <=N
        if D <= 0:
            return -d_n
        return -D * math.expm1((N / D) * math.log1p(-q)) - d_n

    lo = d_n
    hi = N

    if equation(lo) >= 0:
        return lo
    if equation(hi) <= 0:
        return hi

    for _ in range(100):
        mid = (lo + hi) / 2
        if equation(mid) < 0:
            lo = mid
        else:
            hi = mid
        if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-8):
            break
    return (lo + hi) / 2


def _gamma2(d_n: float, frequencies: dict | None, n: float, N: float) -> float:
    """
    gamma^2 = max(0,D/n^2*sum_i[i*(i-1)*f_i]+ D/N - 1)

    d_n : estimated population distinct count (D_uj1)
    frequencies : {i:f_i} -> historgam of sketch counts
    n : np.sum(sketch) -> total sample size
    N : total population
    """

    if d_n <= 0 or n <= 0:
        return 0.0
    if frequencies:
        vals = np.array(list(frequencies.keys()), dtype=float)
        cts = np.array(list(frequencies.values()), dtype=float)
        full_sum = float(np.sum(vals * (vals - 1) * cts))
    else:
        full_sum = 0

    return max(0.0, (d_n / max(n, 1.0) ** 2) * full_sum + d_n / max(N, 1.0) - 1.0)


def _duj2(
    d_n: float, f_1: float, frequencies: dict | None, q: float, n: float, N: float
) -> float:
    """
    Unsmoothened second-order jackknife
    d_n : positions observed
    f_1 : positions seen once
    frequencies :  {i:f_i} -> historgam of sketch counts
    q : product of reduced-dimension sampling probabilities
    n : np.sum(sketch) -> total sample size
    N : total population size

    The gamma^2 containing term account for variance in multiplicty
    """
    if d_n == 0:
        return 0.0
    if q <= 0.0:
        return d_n
    if q >= 1.0:
        return d_n

    D_uj1 = _duj1(d_n, f_1, q, n)
    gamma2 = _gamma2(D_uj1, frequencies, n, N)

    ln1mq = math.log1p(-q)
    rhs = d_n - f_1 * (1.0 - q) * (ln1mq / q) * gamma2

    estimate = D_uj1 * (rhs / d_n)
    return max(float(estimate), d_n)


def _dsh(d_n: float, f_1: float, frequencies: dict | None, q: float, n: float) -> float:
    """
    Schlosser estimator - uses all frequency counts

    K_Sh = n * sum((1-q)^i * f_i) / sum(i*q*(1-q)^(i-1) * f_i)

    num : higher i, smaller weight -> estimates number of positions missed
    denom : expected total sample size contribution per missed position

    Using D = d_n + K*f_1/n
    """

    if d_n == 0:
        return 0.0
    if q <= 0.0 or q >= 1.0 or f_1 == 0 or not frequencies:
        return d_n
    vals = np.array(list(frequencies.keys()), dtype=float)
    cts = np.array(list(frequencies.values()), dtype=float)
    weights = np.exp((vals - vals.min()) * math.log1p(-q)) * cts
    ratio = float(np.sum(weights) / np.sum(vals * weights))
    return d_n + f_1 * ((1.0 - q) / q) * ratio


def _dsh2(
    d_n: float, f_1: float, frequencies: dict | None, q: float, n: float, N: float
) -> float:
    """
    Modified Schlosser - corrects the bias in K_Sh
    N_bar = N/D_uj1

    Correction factor = q*(1+q)^(N_bar-1) / ((1+q)^N_bar - 1)

    Initial estimate for D = D_uj1
    """
    if d_n == 0:
        return 0.0
    if q <= 0.0 or q >= 1.0 or f_1 == 0 or not frequencies:
        return d_n

    D_uj1 = _duj1(d_n, f_1, q, n)
    N_bar = N / max(D_uj1, 1.0)
    denom = -math.expm1(-N_bar * math.log1p(q))
    if denom <= 0:
        return d_n

    vals = np.array(list(frequencies.keys()), dtype=float)
    cts = np.array(list(frequencies.values()), dtype=float)
    weights = np.exp((vals - vals.min()) * math.log1p(-q)) * cts
    ratio = float(np.sum(weights) / np.sum(vals * weights))
    return d_n + f_1 * ((1.0 - q) / (1.0 + q)) * ratio / denom


def _dsh3(d_n: float, f_1: float, frequencies: dict | None, q: float, n: float):
    """
    Further modified Schlosser
    num1 = sum(i * q^2 * (1-q^2)^(i-1) * f_i)
    den1 = sum((1-q)^i * ((1+q)^i - 1) * f_i)
    """
    if d_n == 0:
        return 0.0
    if q <= 0.0 or q >= 1.0 or f_1 == 0 or not frequencies:
        return d_n

    vals = np.array(list(frequencies.keys()), dtype=float)
    cts = np.array(list(frequencies.values()), dtype=float)
    shifted = vals - vals.min()
    weights = np.exp(shifted * math.log1p(-q)) * cts
    weights2 = np.exp(shifted * math.log1p(-(q * q))) * cts
    # (1-q)^i * ((1+q)^i - 1) = (1-q^2)^i * (1-(1+q)^(-i)).
    denom = float(np.sum(weights2 * -np.expm1(-vals * math.log1p(q))))
    if denom <= 0:
        return d_n
    ratio = float(np.sum(weights) / np.sum(vals * weights))
    correction = ((1.0 - q) / (1.0 + q)) * float(np.sum(vals * weights2)) / denom
    return d_n + f_1 * correction * ratio * ratio


class SamplingStatsFactory(
    BaseTensorStatsFactory["SamplingStats"], StatsFactory["SamplingStats"]
):
    def __init__(self, sample_nnz: int = 10_000, estimator: str = "sh3"):
        super().__init__(SamplingStats)
        if sample_nnz < 1:
            raise ValueError("sample_nnz must be positive")
        if estimator not in SAMPLING_ESTIMATORS:
            raise ValueError(f"Unknown estimator: {estimator!r}")
        self.sample_nnz = sample_nnz
        self.estimator = estimator
        self._seeds: dict[tuple[Field, int], int] = {}
        self._rng = np.random.default_rng()

    def _get_mask(self, field: Field, size: int, prob: float) -> RandomMaskTensor:
        seed_key = (field, size)
        if seed_key not in self._seeds:
            self._seeds[seed_key] = int(self._rng.integers(0, 1 << 64, dtype=np.uint64))
        return RandomMaskTensor(size, prob, seed=self._seeds[seed_key], dtype=np.intp)

    def __call__(
        self,
        tensor: Any,
        fields: tuple[Field, ...],
        sample_probs: list[float] | None = None,
    ) -> SamplingStats:
        base = super().__call__(tensor, fields)
        fill = base.fill_value.value
        sample_probs = (
            [1.0] * len(fields) if sample_probs is None else list(sample_probs)
        )

        # Reuse each field's seed across tensors so joins sample the same
        # coordinates. An entry survives only if every dimension is kept.
        masks = [
            self._get_mask(field, int(base.dim_sizes[field]), prob)
            for field, prob in zip(fields, sample_probs, strict=True)
        ]
        non_fill = MapJoin(
            Literal(ffuncs.ne), (Table(Literal(tensor), fields), Literal(fill))
        )
        mask_tables = [
            mask_table(field, mask) for field, mask in zip(fields, masks, strict=True)
        ]

        sketch = compute_sketch(MapJoin(Literal(ffuncs.mul), (non_fill, *mask_tables)))

        return self.remask(
            SamplingStats(
                base,
                sketch=sketch,
                sample_probs=sample_probs,
                estimator=self.estimator,
                seeds_ref=self._seeds,
            )
        )

    def _mask_sketch(self, stats: SamplingStats, probs: list[float]) -> SamplingStats:
        result = self.copy(stats)
        masks = []
        for field, old_prob, prob in zip(
            stats.index_order, stats.sample_probs, probs, strict=True
        ):
            if prob == old_prob:
                continue
            size = int(stats.dim_sizes[field])
            seed = stats.seeds_ref.get((field, size))
            if seed is None:
                mask = self._get_mask(field, size, prob)
                result.seeds_ref[field, size] = self._seeds[field, size]
            else:
                mask = RandomMaskTensor(size, prob, seed=seed, dtype=np.intp)
            masks.append(mask_table(field, mask))
        if masks:
            result.sketch = compute_sketch(
                MapJoin(Literal(ffuncs.mul), (stats.sketch, *masks))
            )
            result.sample_probs = list(probs)
            result.scan_cache = None
        return result

    def remask(self, stats: SamplingStats) -> SamplingStats:
        """Halve the largest nonzero projection until the sketch fits the budget."""
        if (
            math.prod(int(stats.dim_sizes[f]) for f in stats.index_order)
            <= self.sample_nnz
        ):
            return stats
        while stats.scan(needs_freq=False)[1] > self.sample_nnz:
            match stats.sketch:
                case Table(Literal(tensor), fields):
                    norms = get_lp_norms(tensor, fields, (0,))
                case _:
                    raise TypeError("remask requires a materialized sketch")
            dim = max(
                range(len(stats.index_order)),
                key=lambda idx: norms[stats.index_order[idx]][0],
            )
            probs = list(stats.sample_probs)
            probs[dim] *= 0.5
            stats = self._mask_sketch(stats, probs)
        return stats

    def _align_mapjoin(
        self, *args: SamplingStats
    ) -> tuple[tuple[SamplingStats, ...], dict[Field, float]]:
        probs: dict[Field, float] = {}
        for arg in args:
            for field, prob in zip(arg.index_order, arg.sample_probs, strict=True):
                probs[field] = min(probs.get(field, prob), prob)
        aligned = tuple(
            self._mask_sketch(arg, [probs[f] for f in arg.index_order]) for arg in args
        )
        return aligned, probs

    def _mapjoin_join(
        self, op: FinchOperator, *join_args: SamplingStats
    ) -> SamplingStats:
        """
        N(C)_i =  N(A)_j * N(B)_k
        """

        if len(join_args) == 1:
            return self.remask(self.copy(join_args[0]))

        base_stats = super()._mapjoin_defs(op, *join_args)
        join_args, probs = self._align_mapjoin(*join_args)
        result_sketch = compute_sketch(
            MapJoin(Literal(ffuncs.mul), tuple(arg.sketch for arg in join_args))
        )

        return self.remask(
            SamplingStats(
                base_stats,
                sketch=result_sketch,
                sample_probs=[probs[f] for f in base_stats.index_order],
                remainder_size=math.prod(arg.remainder_size for arg in join_args),
                remainder_prob=math.prod(arg.remainder_prob for arg in join_args),
                estimator=self.estimator,
                seeds_ref=self._seeds,
            )
        )

    def _mapjoin_union(self, op: FinchOperator, *union_args: SamplingStats):
        """
        N(C)_i = sum_{juk\\i}[N(A)_j *prod_{l in k\\j}n(B)_l +
        N(B)_k *prod_{l in j\\k}n(A)_l ] - N(A)_j*N(B)_k
        """
        base_stats = super()._mapjoin_defs(op, *union_args)
        union_args, probs = self._align_mapjoin(*union_args)
        output_indices = set(base_stats.index_order)

        terms = []
        for arg in union_args:
            arg_indices = set(arg.index_order)
            other_free_size = 1.0
            for other in union_args:
                if other is arg:
                    continue
                for f in other.index_order:
                    if f not in arg_indices and f not in output_indices:
                        other_free_size *= other.dim_sizes.get(f, 1.0)
                other_free_size *= other.remainder_size
            missing_masks = [
                mask_table(
                    field,
                    self._get_mask(
                        field, int(base_stats.dim_sizes[field]), probs[field]
                    ),
                )
                for field in base_stats.index_order
                if field not in arg_indices
            ]
            terms.append(
                MapJoin(
                    Literal(ffuncs.mul),
                    (arg.sketch, Literal(other_free_size), *missing_masks),
                )
            )

        result = terms[0]
        for term in terms[1:]:
            result = MapJoin(Literal(ffuncs.add), (result, term))

        if len(union_args) >= 2:
            inter = MapJoin(
                Literal(ffuncs.mul), tuple(arg.sketch for arg in union_args)
            )

            result = MapJoin(Literal(ffuncs.sub), (result, inter))

        return self.remask(
            SamplingStats(
                base_stats,
                sketch=compute_sketch(result),
                sample_probs=[probs[f] for f in base_stats.index_order],
                remainder_size=math.prod(arg.remainder_size for arg in union_args),
                remainder_prob=math.prod(arg.remainder_prob for arg in union_args),
                estimator=self.estimator,
                seeds_ref=self._seeds,
            )
        )

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: SamplingStats,
    ):
        """
        op is identity on fill : N(B)_i = sum_j N(A)_j
        op annihilates fill: N(B)_i = prod(l in k)n_l * min_k 1[N(A)_j > 0]
        otherwise : N(B)_i = prod(l in k)n_l * exists(N(A)_j)
        """

        base_stats = self.aggregate_def(op, init, reduce_indices, stats)
        reduce_set = set(reduce_indices) & set(stats.index_order)
        reduce_fields = tuple(reduce_set)

        # check is_annihilator
        if is_identity(op.ftype, stats.fill_value):
            new_sketch = (
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(np.intp(0)),
                    stats.sketch,
                    reduce_fields,
                )
                if reduce_fields
                else stats.sketch
            )

        elif is_annihilator(op.ftype, stats.fill_value):
            exists: LogicExpression = MapJoin(
                Literal(ffuncs.gt), (stats.sketch, Literal(0.0))
            )
            if reduce_fields:
                exists = Aggregate(
                    Literal(ffuncs.min), Literal(np.float64(1.0)), exists, reduce_fields
                )
            prod_n = math.prod(int(stats.dim_sizes[f]) for f in reduce_set)
            new_sketch = MapJoin(Literal(ffuncs.mul), (exists, Literal(prod_n)))
        else:
            prod_n = math.prod(int(stats.dim_sizes[f]) for f in reduce_set)
            exists = MapJoin(Literal(ffuncs.gt), (stats.sketch, Literal(0.0)))
            if reduce_fields:
                exists = Aggregate(
                    Literal(ffuncs.max), Literal(np.float64(0.0)), exists, reduce_fields
                )
            new_sketch = MapJoin(Literal(ffuncs.mul), (exists, Literal(prod_n)))

        probs = dict(zip(stats.index_order, stats.sample_probs, strict=True))

        return self.remask(
            SamplingStats(
                base_stats,
                sketch=compute_sketch(new_sketch),
                sample_probs=[probs[f] for f in base_stats.index_order],
                remainder_size=stats.remainder_size
                * math.prod(stats.dim_sizes[f] for f in reduce_fields),
                remainder_prob=stats.remainder_prob
                * math.prod(probs[f] for f in reduce_fields),
                estimator=self.estimator,
                seeds_ref=stats.seeds_ref,
            )
        )

    def relabel(
        self, stats: SamplingStats, relabel_indices: tuple[Field, ...]
    ) -> SamplingStats:
        base_stats = self.relabel_def(stats, relabel_indices)
        seeds = dict(stats.seeds_ref)
        for old, new in zip(stats.index_order, relabel_indices, strict=True):
            size = int(stats.dim_sizes[old])
            if (old, size) in stats.seeds_ref:
                seeds[new, size] = stats.seeds_ref[old, size]
        return self.remask(
            SamplingStats(
                base_stats,
                sketch=compute_sketch(Relabel(stats.sketch, relabel_indices)),
                sample_probs=stats.sample_probs,
                remainder_size=stats.remainder_size,
                remainder_prob=stats.remainder_prob,
                estimator=self.estimator,
                seeds_ref=seeds,
            )
        )

    def reorder(
        self, stats: SamplingStats, reorder_indices: tuple[Field, ...]
    ) -> SamplingStats:
        base_stats = self.reorder_def(stats, reorder_indices)
        probs = dict(zip(stats.index_order, stats.sample_probs, strict=True))
        return self.remask(
            SamplingStats(
                base_stats,
                sketch=compute_sketch(Reorder(stats.sketch, reorder_indices)),
                sample_probs=[probs.get(f, 1.0) for f in base_stats.index_order],
                remainder_size=stats.remainder_size,
                remainder_prob=stats.remainder_prob,
                estimator=self.estimator,
                seeds_ref=stats.seeds_ref,
            )
        )


class SamplingStats(NumericStats):
    """
    sketch : materialized table over bound dimensions
    remainder_size : product of sizes of dimensions aggregated out of the sketch
    remainder_prob : product of their sampling probabilities
    sample_probs : per-dimension probabilities in index_order
    """

    sketch: LogicExpression
    remainder_size: float
    remainder_prob: float
    sample_probs: list[float]

    def __init__(
        self,
        base: BaseTensorStats,
        sketch: LogicExpression,
        sample_probs: list[float],
        estimator: str = "uj1",
        remainder_size: float = 1.0,
        remainder_prob: float = 1.0,
        seeds_ref: dict[tuple[Field, int], int] | None = None,
    ):

        super().__init__(base)
        self.sketch = sketch
        if len(sample_probs) != len(self.index_order):
            raise ValueError("Expected one sampling probability per dimension")
        self.sample_probs = list(sample_probs)
        self.remainder_size = float(remainder_size)
        self.remainder_prob = float(remainder_prob)
        self.estimator = estimator
        self.seeds_ref = seeds_ref if seeds_ref is not None else {}
        self.scan_cache: tuple[float, float, float, dict | None] | None = None

    def scan(self, needs_freq: bool) -> tuple[float, float, float, dict | None]:
        from finch.compile_jl.runtime import JuliaOwnedTensor

        if self.scan_cache is None or (needs_freq and self.scan_cache[3] is None):
            match self.sketch:
                case Table(Literal(tensor), _):
                    pass
                case _:
                    raise TypeError("scan requires a materialized sketch")
            match tensor:
                case JuliaOwnedTensor():
                    tensor = tensor._as_tensor()
            match tensor:
                case FiberTensor(lvl=level):
                    while True:
                        match level:
                            case ElementLevel():
                                values = level.val.arr
                                break
                            case (
                                DenseLevel(lvl=child)
                                | SparseListLevel(lvl=child)
                                | SparseCOOLevel(lvl=child)
                                | SparseHashLevel(lvl=child)
                                | SparseByteMapLevel(lvl=child)
                            ):
                                level = child
                            case _:
                                raise TypeError(
                                    f"Unsupported sketch level: {type(level).__name__}"
                                )
                case _:
                    values = np.asarray(tensor).reshape(-1)
            vals, cts = np.unique_counts(values)
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
        from finch.autoschedule.default_schedulers import NON_RECURSIVE_SCHEDULER

        _, d_n_raw, _, _ = self.scan(needs_freq=False)
        coverage = 1.0
        for field, prob in zip(self.index_order, self.sample_probs, strict=True):
            size = int(self.dim_sizes[field])
            seed = self.seeds_ref.get((field, size))
            if seed is None or size == 0:
                continue
            mask = RandomMaskTensor(size, prob, seed=seed, dtype=np.intp)
            out = HardAlias("sampled_count")
            query = Query(
                Table(out, ()),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(np.intp(0)),
                    mask_table(field, mask),
                    (field,),
                ),
            )
            (count,) = NON_RECURSIVE_SCHEDULER(Plan((query, Produces((out,)))))
            actual_fraction = float(np.asarray(count)[()]) / size
            if actual_fraction <= 0:
                continue
            coverage *= actual_fraction
        if coverage <= 0:
            return d_n_raw
        return d_n_raw / coverage

    def estimate_non_fill_values(self, over: Iterable[Field] = ()) -> float:
        """
        Correct reduced-dimension sampling, then scale to all output coordinates.
        Non-fill values are assumed to be spread evenly over the slices which fix
        the fields of `over`.
        """
        q_output = math.prod(self.sample_probs)
        q = self.remainder_prob
        needs_freq = (
            self.estimator in ("uj2", "schlosser", "sh2", "sh3", "good1")
            and 0.0 < q < 1.0
        )
        n, d_n, f_1, frequencies = self.scan(needs_freq)
        if d_n == 0 or q_output <= 0:
            return 0.0
        bound_size = math.prod(int(self.dim_sizes[f]) for f in self.index_order)
        # The formulas estimate classes among the retained output coordinates.
        # Their population consists of nonzero contributions, not all tensor cells.
        N = n / q if q > 0 else n

        if self.estimator == "uj1":
            formula_est = _duj1(d_n, f_1, q, n)

        elif self.estimator == "good1":
            formula_est = _dgood1(d_n, frequencies, n, N)

        elif self.estimator == "sj1":
            formula_est = _dsj1(d_n, q, N)

        elif self.estimator == "uj2":
            formula_est = _duj2(d_n, f_1, frequencies, q, n, N)
        elif self.estimator == "schlosser":
            formula_est = _dsh(d_n, f_1, frequencies, q, n)

        elif self.estimator == "sh2":
            formula_est = _dsh2(d_n, f_1, frequencies, q, n, N)

        elif self.estimator == "sh3":
            formula_est = _dsh3(d_n, f_1, frequencies, q, n)

        elif self.estimator == "silly":
            formula_est = d_n

        else:
            raise ValueError(
                f"Unknown estimator: {self.estimator!r}."
                f"Choose from: {', '.join(SAMPLING_ESTIMATORS)}"
            )

        total = min(bound_size, max(d_n, formula_est) / q_output)
        slices = self.get_dim_space_size(tuple(set(over) & set(self.index_order)))
        return float(total) / slices

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[f]) for f in self.index_order]
        nnz = self.estimate_non_fill_values()
        size_part = np.log2(np.array(sizes))
        nnz_part = np.log2(np.array(nnz) + 1)
        return np.concatenate([size_part, nnz_part])
