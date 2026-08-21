from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

import finch as ft
from finch.algebra import FinchOperator, ffuncs
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    StatsFactory,
    Table,
)

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


def build_block_expr(
    arr: Any,
    fields: tuple[Field, ...],
    starts: Mapping[Field, int],
    ends: Mapping[Field, int],
    coord: tuple[int, ...],
):

    data_table = Table(Literal(arr), fields)

    select_tbls = []
    for d, f in enumerate(fields):
        start, end = starts[f], ends[f]
        block_size = end - start
        dim_size = arr.shape[d]
        out_f = Field(f"out_{f.name}")
        select = np.zeros((block_size, dim_size), dtype=np.float64)
        select[np.arange(block_size), np.arange(start, end)] = 1
        select_tbls.append(Table(Literal(ft.asarray(select)), (out_f, f)))

    joined = MapJoin(Literal(ffuncs.mul), (data_table, *select_tbls))

    return Aggregate(Literal(ffuncs.add), Literal(np.float64(0.0)), joined, fields)


def get_blocks_subtensor(
    arr: Any,
    fields: tuple[Field, ...],
    blocks_per_dim: Mapping[Field, int],
) -> dict[tuple[int, ...], Any]:

    from finch.autoschedule.default_schedulers import NON_RECURSIVE_SCHEDULER

    fields = tuple(fields)
    grid_dim = [blocks_per_dim[f] for f in fields]
    base_block_sizes = {
        f: arr.shape[d] / blocks_per_dim[f] for d, f in enumerate(fields)
    }

    blocks: dict[tuple[int, ...], Any] = {}

    for coord in np.ndindex(*grid_dim):
        starts, ends = {}, {}
        for d, f in enumerate(fields):
            start = math.floor(coord[d] * base_block_sizes[f])
            end = int(
                arr.shape[d]
                if coord[d] == blocks_per_dim[f] - 1
                else math.floor((coord[d] + 1) * base_block_sizes[f])
            )
            starts[f], ends[f] = start, end

        expr = build_block_expr(arr, fields, starts, ends, coord)
        out = Alias(f"block_{'_'.join(str(c) for c in coord)}")
        prgm = Plan((Query(out, expr), Produces((out,))))
        (result,) = NON_RECURSIVE_SCHEDULER(prgm)
        blocks[coord] = result

    return blocks


class BlockedStatsFactory(
    BaseTensorStatsFactory["BlockedStats"], StatsFactory["BlockedStats"]
):
    def __init__(
        self,
        stats_factory: StatsFactory[NumericStats],
        block_count: int = 5,
        block_width: int = 5,
        blocks_per_dim: Mapping[Field, int] | None = None,
    ):
        super().__init__(BlockedStats)
        self.block_count = block_count
        self.block_width = block_width
        self.blocks_per_dim = (
            dict(blocks_per_dim) if blocks_per_dim is not None else None
        )
        self.inner_factory = stats_factory

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> BlockedStats:
        base = super().__call__(tensor, fields)
        blocks_per_dim = (
            self.blocks_per_dim
            if self.blocks_per_dim is not None
            else {
                f: max(1, min(self.block_count, n // self.block_width))
                for f, n in zip(fields, tensor.shape, strict=True)
            }
        )
        grid = BlockedStats.build_grid(
            base, blocks_per_dim, self.inner_factory, data=tensor
        )
        return BlockedStats(grid, dict(blocks_per_dim), base, self.inner_factory)

    def copy(self, stat: BlockedStats) -> BlockedStats:
        if not isinstance(stat, BlockedStats):
            raise TypeError("copy expected a BlockedStats instance")

        new_blocks = np.empty_like(stat.blocks)
        for i in range(stat.blocks.size):
            new_blocks.flat[i] = stat.stats_factory.copy(stat.blocks.flat[i])

        return BlockedStats(
            new_blocks,
            stat.blocks_per_dim.copy(),
            stat,
            stat.stats_factory,
        )

    def mapjoin(self, op: FinchOperator, *args: BlockedStats) -> BlockedStats:

        b_args: list[BlockedStats] = list(args)
        first_arg = b_args[0]
        base = BaseTensorStatsFactory._mapjoin_defs(op, *b_args)

        blocks_per_dim = {k: v for arg in b_args for k, v in arg.blocks_per_dim.items()}

        new_blocks = np.empty(
            tuple(blocks_per_dim[idx] for idx in base.index_order), dtype=object
        )

        inner_factory = first_arg.stats_factory

        for coord in np.ndindex(new_blocks.shape):
            local_blocks: list[NumericStats] = []
            global_coord = dict(zip(base.index_order, coord, strict=True))
            for arg in b_args:
                local_coord = tuple(global_coord[idx] for idx in arg.index_order)
                block: Any = arg.blocks[local_coord]
                if isinstance(block, NumericStats):
                    local_blocks.append(block)
            new_blocks[coord] = inner_factory.mapjoin(op, *local_blocks)

        return BlockedStats(new_blocks, first_arg.blocks_per_dim, base, inner_factory)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: BlockedStats,
    ) -> BlockedStats:
        if not isinstance(stats, BlockedStats):
            raise TypeError("BlockedStats expected for aggregate")

        base = BaseTensorStatsFactory.aggregate_def(op, init, reduce_indices, stats)
        grid_reduce_axes = []
        for i, idx in enumerate(stats.index_order):
            if idx in reduce_indices:
                grid_reduce_axes.append(i)

        new_grid_shape = list(stats.blocks.shape)
        for axis in grid_reduce_axes:
            new_grid_shape[axis] = 1

        new_blocks = np.empty(new_grid_shape, dtype=object)

        for out_coord in np.ndindex(*new_grid_shape):
            lane_slices: list[slice | int] = []
            for i, val in enumerate(out_coord):
                if i in grid_reduce_axes:
                    lane_slices.append(slice(None))
                else:
                    lane_slices.append(val)

            blocks_in_lane = stats.blocks[tuple(lane_slices)].flat

            lane_accumulator = None
            for b in blocks_in_lane:
                local_reduced = stats.stats_factory.aggregate(
                    op, init, reduce_indices, b
                )

                if lane_accumulator is None:
                    lane_accumulator = local_reduced
                else:
                    lane_accumulator = stats.stats_factory.mapjoin(
                        op, lane_accumulator, local_reduced
                    )

            new_blocks[out_coord] = lane_accumulator

        final_grid = np.squeeze(new_blocks, axis=tuple(grid_reduce_axes))
        new_blocks_per_dim = {
            k: v for k, v in stats.blocks_per_dim.items() if k not in reduce_indices
        }

        return BlockedStats(
            final_grid,
            new_blocks_per_dim,
            base,
            stats.stats_factory,
        )

    def relabel(
        self, stats: BlockedStats, relabel_indices: tuple[Field, ...]
    ) -> BlockedStats:
        base = BaseTensorStatsFactory.relabel_def(stats, relabel_indices)

        if not isinstance(stats, BlockedStats):
            raise TypeError("BlockedStats expected for relabel")

        name_map = dict(zip(stats.index_order, relabel_indices, strict=True))
        new_blocks_per_dim = {name_map[k]: v for k, v in stats.blocks_per_dim.items()}

        new_blocks = np.empty_like(stats.blocks)
        for coord in np.ndindex(stats.blocks.shape):
            block: Any = stats.blocks[coord]
            if isinstance(block, NumericStats):
                new_blocks[coord] = stats.stats_factory.relabel(block, relabel_indices)

        return BlockedStats(new_blocks, new_blocks_per_dim, base, stats.stats_factory)

    def reorder(
        self, stats: BlockedStats, reorder_indices: tuple[Field, ...]
    ) -> BlockedStats:
        if not isinstance(stats, BlockedStats):
            raise TypeError("BlockedStats expected for reorder")

        base = BaseTensorStatsFactory.reorder_def(stats, reorder_indices)

        old_order = stats.index_order
        dropped = [
            i for i, idx in enumerate(old_order) if idx not in set(reorder_indices)
        ]
        axes_mapping = [
            old_order.index(idx) for idx in reorder_indices if idx in old_order
        ] + dropped

        new_blocks = np.transpose(stats.blocks, axes=axes_mapping)

        expanded_shape = [stats.blocks_per_dim.get(idx, 1) for idx in reorder_indices]
        new_blocks = new_blocks.reshape(expanded_shape)

        final_blocks = np.empty_like(new_blocks)
        for coord in np.ndindex(new_blocks.shape):
            block: Any = new_blocks[coord]
            if isinstance(block, NumericStats):
                final_blocks[coord] = stats.stats_factory.reorder(
                    block, reorder_indices
                )

        new_blocks_per_dim = {
            idx: stats.blocks_per_dim.get(idx, 1) for idx in reorder_indices
        }

        return BlockedStats(
            final_blocks,
            new_blocks_per_dim,
            base,
            stats.stats_factory,
        )


class BlockedStats(NumericStats):
    def __init__(
        self,
        blocks: np.ndarray,
        blocks_per_dim: dict[Field, int],
        base: BaseTensorStats,
        stats_factory: StatsFactory[NumericStats],
    ):
        super().__init__(base)
        self.blocks = blocks
        self.blocks_per_dim = blocks_per_dim
        self.stats_factory = stats_factory

    @classmethod
    def build_grid(
        cls,
        d: BaseTensorStats,
        blocks_per_dim: Mapping[Field, int],
        stats_factory: StatsFactory[NumericStats],
        data: Any,
    ) -> np.ndarray:
        grid_dim = [blocks_per_dim[idx] for idx in d.index_order]
        blocks_grid = np.empty(grid_dim, dtype=object)

        blocks = get_blocks_subtensor(data, d.index_order, blocks_per_dim)

        for coord, block in blocks.items():
            blocks_grid[coord] = stats_factory(block, d.index_order)

        return blocks_grid

    def estimate_non_fill_values(self):
        return float(sum(b.estimate_non_fill_values() for b in self.blocks.flat))

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        total_elements = math.prod(self.dim_sizes.values())
        num_blocks = self.blocks.size
        block_volume = total_elements / num_blocks
        densities = [
            b.estimate_non_fill_values() / block_volume for b in self.blocks.flat
        ]
        density_array = np.array(densities)
        dense_part = np.log2(density_array + 1)
        size_part = np.log2(sizes)

        return np.concatenate([size_part, dense_part])

    def copy(self) -> BlockedStats:
        new_blocks = np.empty_like(self.blocks)
        for i in range(self.blocks.size):
            new_blocks.flat[i] = self.stats_factory.copy(self.blocks.flat[i])

        return BlockedStats(
            new_blocks,
            self.blocks_per_dim.copy(),
            self,
            self.stats_factory,
        )
