"""
Output format selection matching Galley's physical optimizer.

This is a direct port of `select_output_format` from
`src/Galley/PhysicalOptimizer/format-selector.jl` in Finch.jl.  The Julia
version stores levels innermost-index-first and therefore walks the output
indices back to front; here `index_order[0]` is the outermost level, so the
walk runs front to back and the resulting formats need no reversal.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from enum import Enum
from typing import TypeVar

from finch import finch_logic as lgc
from finch.algebra import AbstractFill, FType, TensorFType, ffuncs, ftype, ftypes
from finch.autoschedule.stages import LogicFormatter
from finch.autoschedule.tensor_stats import StatsInterpreter
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.finch_logic import Field, LogicLoader, MockLogicLoader, StatsFactory
from finch.tensor import (
    dense,
    element,
    fiber_tensor,
    sparse_bytemap,
    sparse_hash,
    sparse_list,
)
from finch.tensor.fiber_tensor import FiberTensorFType
from finch.util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)

NS = TypeVar("NS", bound=NumericStats)

# A level is stored densely when at least this fraction of its slots is
# expected to be occupied, and as a bytemap down to the second threshold.
DENSE_SPARSITY_THRESHOLD = 0.5
BYTEMAP_SPARSITY_THRESHOLD = 0.05
# Neither dense nor bytemap levels are worth their memory beyond this many
# allocated slots.
MAX_DENSE_MEMORY_FOOTPRINT = 3 * 10**10


class LevelFormat(Enum):
    """The level formats Galley's format selector chooses between."""

    DENSE = "dense"
    BYTEMAP = "bytemap"
    HASH = "hash"
    SPARSE_LIST = "sparse_list"

    def build(self, lvl, dimension_type: FType):
        match self:
            case LevelFormat.DENSE:
                return dense(lvl, dimension_type)
            case LevelFormat.BYTEMAP:
                return sparse_bytemap(lvl, dimension_type)
            case LevelFormat.HASH:
                return sparse_hash(lvl, dimension_type, single_writer=True)
            case LevelFormat.SPARSE_LIST:
                return sparse_list(lvl, dimension_type)


def fully_compat_with_loop_prefix(
    tensor_order: tuple[Field, ...], loop_prefix: tuple[Field, ...]
) -> bool:
    """
    Whether `tensor_order` can be written sequentially under `loop_prefix`.

    Both are given outermost first.  A tensor deeper than the loop prefix is
    still compatible; a mismatch at any shared position is not.
    """
    for i, idx in enumerate(tensor_order):
        if i >= len(loop_prefix):
            return True
        if idx != loop_prefix[i]:
            return False
    return True


def estimate_nnz(
    stats: NS, stats_factory: StatsFactory[NS], indices: tuple[Field, ...]
) -> float:
    """
    Estimate the number of non-fill values of `stats` projected onto `indices`.
    """
    reduce_fields = tuple(idx for idx in stats.index_order if idx not in indices)
    if reduce_fields:
        stats = stats_factory.aggregate(ffuncs.or_, False, reduce_fields, stats)
    return stats.estimate_non_fill_values()


def select_output_format(
    stats: NS,
    stats_factory: StatsFactory[NS],
    loop_order: tuple[Field, ...],
    output_indices: tuple[Field, ...],
) -> tuple[LevelFormat, ...]:
    """
    Choose a level format for each of `output_indices`, outermost level first.

    `loop_order` is the loop order the output is built under, also outermost
    first.  Levels which cannot be filled in loop order need random writes and
    so fall back to a hash level rather than a sparse list.
    """
    formats: list[LevelFormat] = []
    for level, idx in enumerate(output_indices):
        prefix = output_indices[: level + 1]
        needs_random_writes = not fully_compat_with_loop_prefix(prefix, loop_order)
        # Division is used rather than conditional indices because a lower
        # estimate is the conservative one here.
        prev_nnz = max(estimate_nnz(stats, stats_factory, prefix[:-1]), 1.0)
        new_nnz = estimate_nnz(stats, stats_factory, prefix)
        dim_size = stats.get_dim_size(idx)
        approx_nnz_per = new_nnz / prev_nnz
        # Scale the sparsity down a bit so that formats stay conservative.
        approx_sparsity = approx_nnz_per / dim_size if dim_size else float("inf")
        dense_memory_footprint = prev_nnz * dim_size

        if (
            approx_sparsity > DENSE_SPARSITY_THRESHOLD
            and dense_memory_footprint < MAX_DENSE_MEMORY_FOOTPRINT
        ):
            fmt = LevelFormat.DENSE
        elif (
            approx_sparsity > BYTEMAP_SPARSITY_THRESHOLD
            and dense_memory_footprint < MAX_DENSE_MEMORY_FOOTPRINT
            # TODO: Check out finch double bytemap bug
            and (not formats or formats[-1] is not LevelFormat.BYTEMAP)
        ):
            fmt = LevelFormat.BYTEMAP
        elif needs_random_writes:
            fmt = LevelFormat.HASH
        else:
            fmt = LevelFormat.SPARSE_LIST
        formats.append(fmt)
    return tuple(formats)


def query_loop_order(rhs: lgc.LogicExpression) -> tuple[Field, ...]:
    """
    The loop order a query's right hand side is evaluated under.

    Loop orders are carried by the `Reorder` node interior to an aggregate; a
    transpose query is looped in the order its input is stored.
    """
    match rhs:
        case lgc.Reorder(lgc.Table(_, idxs), _):
            return idxs
        case lgc.Reorder(lgc.Aggregate(_, _, lgc.Reorder(_, idxs), _), _):
            return idxs
        case lgc.Reorder(
            lgc.MapJoin(_, (lgc.Table(), lgc.Aggregate(_, _, lgc.Reorder(_, idxs), _))),
            _,
        ):
            return idxs
        case lgc.Reorder(_, idxs):
            return idxs
        case _:
            return rhs.fields()


class GalleyFormatter(LogicFormatter):
    """
    A formatter which picks output formats the way Galley's physical optimizer
    does, choosing per level between dense, bytemap, hash, and sparse list
    based on the estimated sparsity of the output and the loop order it is
    built under.
    """

    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader

    def get_tensor_ftype(
        self,
        fill_value: AbstractFill,
        shape_type: tuple[FType, ...],
        stats: NumericStats,
        stats_factory: StatsFactory,
        loop_order: tuple[Field, ...],
    ) -> FiberTensorFType:
        output_indices = stats.index_order
        if len(shape_type) != len(output_indices):
            raise ValueError(
                f"Got {len(shape_type)} shape dimensions for "
                f"{len(output_indices)} stats dimensions."
            )

        formats = select_output_format(stats, stats_factory, loop_order, output_indices)
        logger.debug("Galley formats for %s: %s", output_indices, formats)

        lvl = element(fill_value, ftype(fill_value))
        for level in reversed(range(len(output_indices))):
            lvl = formats[level].build(lvl, shape_type[level])
        return fiber_tensor(lvl)

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, NS],
        stats_factory: StatsFactory[NS],
    ):
        bindings = bindings.copy()
        stats_bindings: OrderedDict[lgc.Alias, NS] = OrderedDict(stats)
        stats_interpreter = StatsInterpreter(stats_factory=stats_factory)
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement) -> lgc.LogicStatement:
            match node:
                case lgc.Plan(bodies):
                    return lgc.Plan(tuple(formatter(body) for body in bodies))
                case lgc.Query(lhs, rhs):
                    rhs_stats = stats_interpreter(rhs, stats_bindings)
                    if not isinstance(rhs_stats, NumericStats):
                        raise TypeError("GalleyFormatter requires NumericStats.")
                    stats_bindings[lhs] = rhs_stats  # ty: ignore[invalid-assignment]

                    if lhs not in bindings:
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftypes.intp
                            for dim in shape_types[lhs]
                        )
                        bindings[lhs] = self.get_tensor_ftype(
                            fill_values[lhs],
                            shape_type,
                            rhs_stats,
                            stats_factory,
                            query_loop_order(rhs),
                        )

                    match rhs:
                        case lgc.Reorder():
                            return node
                        case _:
                            return lgc.Query(lhs, lgc.Reorder(rhs, rhs.fields()))
                case lgc.Produces():
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        prgm = formatter(prgm)

        logger.debug(prgm)

        return self.ctx(prgm, bindings, stats_bindings, stats_factory)
