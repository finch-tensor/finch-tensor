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
from finch.algebra import (
    AbstractFillFType,
    FType,
    TensorFType,
    ffuncs,
    ftype,
    ftypes,
)
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
# A workspace written out of order is a byte map up to this many slots, few
# enough to reset and copy out cheaply (16 MB of float64), such as a row of a
# product over a graph of a million vertices.
WORKSPACE_LIMIT = 2_000_000


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
                return sparse_hash(lvl, dimension_type)
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
        stats = stats_factory.aggregate(
            ffuncs.or_, False, reduce_fields, stats_factory.support(stats)
        )
    return stats.estimate_non_fill_values()


def select_output_format(
    stats: NS,
    stats_factory: StatsFactory[NS],
    loop_order: tuple[Field, ...],
    output_indices: tuple[Field, ...],
    nfused: int = 0,
) -> tuple[LevelFormat, ...]:
    """
    Choose a level format for each of `output_indices` after the first
    `nfused`, outermost level first.

    `loop_order` is the loop order the output is built under, also outermost
    first.  Levels which cannot be filled in loop order need random writes and
    so fall back to a hash level rather than a sparse list.

    The first `nfused` indices are those a workspace is fused over: it stores
    one slice of the rest at a time, so the density of each stored level is
    estimated within the average slice, and a dense level takes the memory of
    one slice. A workspace of one level of at most WORKSPACE_LIMIT slots, such
    as a row of a sparse product, which is written out of order is a byte map
    rather than a hash.
    """
    slices = 1.0
    if nfused:
        slices = max(estimate_nnz(stats, stats_factory, output_indices[:nfused]), 1.0)
    formats: list[LevelFormat] = []
    for level in range(nfused, len(output_indices)):
        idx = output_indices[level]
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
        dense_memory_footprint = prev_nnz / slices * dim_size

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
        ) or (
            0 < nfused == len(output_indices) - 1
            and needs_random_writes
            and dense_memory_footprint <= WORKSPACE_LIMIT
        ):
            fmt = LevelFormat.BYTEMAP
        elif needs_random_writes:
            fmt = LevelFormat.HASH
        else:
            fmt = LevelFormat.SPARSE_LIST
        formats.append(fmt)
    return tuple(formats)


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
        fill_type: AbstractFillFType,
        shape_type: tuple[FType, ...],
        stats: NumericStats,
        stats_factory: StatsFactory,
        loop_order: tuple[Field, ...],
        nfused: int = 0,
    ) -> FiberTensorFType:
        """
        The format of a tensor with stats, built under loop_order, which stores
        the fields of stats after the first nfused, those a workspace is fused
        over.
        """
        output_indices = stats.index_order
        if len(shape_type) != len(output_indices) - nfused:
            raise ValueError(
                f"Got {len(shape_type)} shape dimensions for "
                f"{len(output_indices) - nfused} stored stats dimensions."
            )

        formats = select_output_format(
            stats, stats_factory, loop_order, output_indices, nfused
        )
        logger.debug("Galley formats for %s: %s", output_indices[nfused:], formats)

        lvl = element(fill_type, fill_type.element_type)
        for level in reversed(range(len(formats))):
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
        shape_types: dict[lgc.Alias, tuple[FType | None, ...]] = {
            var: val.shape_type for var, val in bindings.items()
        }
        fill_types = prgm.infer_fill_type(
            {var: val.fill_type for var, val in bindings.items()}
        )

        # The loop order of a query is the fields of its enclosing Fuse nodes.
        def formatter(
            node: lgc.LogicStatement, loop_order: tuple[Field, ...] = ()
        ) -> lgc.LogicStatement:
            match node:
                case lgc.Plan(bodies):
                    return lgc.Plan(
                        tuple(formatter(body, loop_order) for body in bodies)
                    )
                case lgc.Fuse(idx, body):
                    return lgc.Fuse(idx, formatter(body, (*loop_order, idx)))
                case lgc.Query(lgc.Table(lgc.Alias() as lhs, _), _) | lgc.QueryInto(
                    lgc.Table(lgc.Alias() as lhs, _), _, _
                ):
                    node.infer_shape_type(shape_types)
                    # Evaluating the query gives the stats of the stored result,
                    # in the order of its left-hand table.
                    rhs_stats = stats_interpreter(node, stats_bindings)
                    if not isinstance(rhs_stats, NumericStats):
                        raise TypeError("GalleyFormatter requires NumericStats.")
                    stats_bindings[lhs] = rhs_stats  # ty: ignore[invalid-assignment]

                    alias, nfused = lhs.unfused, lhs.nfused
                    if alias not in bindings:
                        # A workspace's formats are chosen per slice, from the
                        # stats over every field, not the union of its slices.
                        format_stats = stats_bindings[lhs]
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftypes.intp
                            for dim in shape_types[alias][nfused:]
                        )
                        bindings[alias] = self.get_tensor_ftype(
                            fill_types[alias],
                            shape_type,
                            format_stats,
                            stats_factory,
                            loop_order,
                            nfused,
                        )

                    return node
                case lgc.Produces():
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        prgm = formatter(prgm)

        logger.debug(prgm)

        return self.ctx(prgm, bindings, stats_bindings, stats_factory)
