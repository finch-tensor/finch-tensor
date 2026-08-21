from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from finch.algebra.algebra import FinchOperator, is_annihilator, is_identity
from finch.finch_logic import Field, StatsFactory

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


def build_grid_uniform(
    d: BaseTensorStats, blocks_per_dim: Mapping[Field, int], data: np.ndarray
) -> np.ndarray:
    index_order = d.index_order
    base_block_size = {
        idx: d.dim_sizes[idx] / blocks_per_dim[idx] for idx in index_order
    }
    block_starts: dict[Field, list[int]] = {
        idx: [math.floor(k * base_block_size[idx]) for k in range(blocks_per_dim[idx])]
        for idx in index_order
    }

    block_sizes: dict[Field, np.ndarray] = {}
    for idx in index_order:
        starts = [*block_starts[idx], int(d.dim_sizes[idx])]
        block_sizes[idx] = np.diff(np.array(starts, dtype=float))

    arr = np.asarray(data)
    nnz_grid = (arr != d.fill_value).astype(np.int64)

    for axis in reversed(range(len(index_order))):
        idx = index_order[axis]
        nnz_grid = np.add.reduceat(nnz_grid, block_starts[idx], axis=axis)

    return nnz_grid.astype(float), block_sizes


def _block_volume_grid(
    index_order: tuple[Field, ...], block_sizes: dict[Field, np.ndarray]
) -> np.ndarray:
    grid_shape = tuple(block_sizes[idx].shape[0] for idx in index_order)
    vol = np.ones(grid_shape, dtype=float)
    for axis, idx in enumerate(index_order):
        shape = [1] * len(index_order)
        shape[axis] = -1
        vol = vol * block_sizes[idx].reshape(shape)
    return vol


def _merge_block_sizes(
    index_order: tuple[Field, ...], args: tuple[BlockedUniformStats, ...]
) -> dict[Field, np.ndarray]:
    block_sizes: dict[Field, np.ndarray] = {}
    for idx in index_order:
        for arg in args:
            if idx in arg.index_order:
                block_sizes[idx] = arg.block_sizes[idx]
                break
    return block_sizes


class BlockedUniformStatsFactory(
    BaseTensorStatsFactory["BlockedUniformStats"], StatsFactory["BlockedUniformStats"]
):
    def __init__(
        self,
        block_count: int = 5,
        block_width: int = 5,
        blocks_per_dim: Mapping[Field, int] | None = None,
    ):
        super().__init__(BlockedUniformStats)
        self.block_count = block_count
        self.block_width = block_width
        self.blocks_per_dim = (
            dict(blocks_per_dim) if blocks_per_dim is not None else None
        )

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> BlockedUniformStats:
        base = super().__call__(tensor, fields)
        blocks_per_dim = (
            self.blocks_per_dim
            if self.blocks_per_dim is not None
            else {
                f: max(1, min(self.block_count, n // self.block_width))
                for f, n in zip(fields, tensor.shape, strict=True)
            }
        )
        nnz_grid, block_sizes = build_grid_uniform(base, blocks_per_dim, tensor)
        return BlockedUniformStats(base, dict(blocks_per_dim), nnz_grid, block_sizes)

    def copy(self, stat: BlockedUniformStats) -> BlockedUniformStats:
        if not isinstance(stat, BlockedUniformStats):
            raise TypeError("copy expected a BlockedUniformStats instance")
        return BlockedUniformStats(
            stat,
            stat.blocks_per_dim.copy(),
            stat.nnz_grid.copy(),
            {k: v.copy() for k, v in stat.block_sizes.items()},
        )

    def _mapjoin_union(
        self, op: FinchOperator, *union_args: BlockedUniformStats
    ) -> BlockedUniformStats:
        base = self._mapjoin_defs(op, *union_args)
        blocks_per_dim: dict[Field, int] = {}
        for arg in union_args:
            blocks_per_dim.update(arg.blocks_per_dim)
        block_sizes = _merge_block_sizes(base.index_order, union_args)
        grid_shape = tuple(blocks_per_dim[idx] for idx in base.index_order)

        new_vol = base.get_dim_space_size(base.index_order)
        if new_vol == 0.0:
            return BlockedUniformStats(
                base, blocks_per_dim, np.zeros(grid_shape), block_sizes
            )

        inv_p = np.ones(grid_shape, dtype=float)
        for s in union_args:
            inv_p = inv_p * (1 - s._align_density(base.index_order))
        res_p = 1 - inv_p

        vol_grid = _block_volume_grid(base.index_order, block_sizes)
        return BlockedUniformStats(base, blocks_per_dim, res_p * vol_grid, block_sizes)

    def _mapjoin_join(
        self, op: FinchOperator, *join_args: BlockedUniformStats
    ) -> BlockedUniformStats:
        base = self._mapjoin_defs(op, *join_args)
        blocks_per_dim: dict[Field, int] = {}
        for arg in join_args:
            blocks_per_dim.update(arg.blocks_per_dim)
        block_sizes = _merge_block_sizes(base.index_order, join_args)
        grid_shape = tuple(blocks_per_dim[idx] for idx in base.index_order)

        new_vol = base.get_dim_space_size(base.index_order)
        if new_vol == 0.0:
            return BlockedUniformStats(
                base, blocks_per_dim, np.zeros(grid_shape), block_sizes
            )

        res_p = np.ones(grid_shape, dtype=float)
        for s in join_args:
            res_p = res_p * s._align_density(base.index_order)

        vol_grid = _block_volume_grid(base.index_order, block_sizes)
        return BlockedUniformStats(base, blocks_per_dim, res_p * vol_grid, block_sizes)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: BlockedUniformStats,
    ) -> BlockedUniformStats:
        if not isinstance(stats, BlockedUniformStats):
            raise TypeError("BlockedUniformStats arguments expected")
        base = self.aggregate_def(op, init, reduce_indices, stats)

        reduce_axes = tuple(
            i for i, idx in enumerate(stats.index_order) if idx in reduce_indices
        )
        new_blocks_per_dim = {
            k: v for k, v in stats.blocks_per_dim.items() if k not in reduce_indices
        }
        new_block_sizes = {
            k: v for k, v in stats.block_sizes.items() if k not in reduce_indices
        }

        if not reduce_axes:
            return BlockedUniformStats(
                base, new_blocks_per_dim, stats.nnz_grid.copy(), new_block_sizes
            )
        density = stats.density_grid()

        k = np.ones(density.shape, dtype=float)
        for axis in reduce_axes:
            idx = stats.index_order[axis]
            shape = [1] * len(stats.index_order)
            shape[axis] = -1
            k = k * stats.block_sizes[idx].reshape(shape)

        if is_annihilator(op, stats.fill_value):
            local_p = np.power(density, k)
        elif is_identity(op, stats.fill_value):
            local_p = 1 - np.power(1 - density, k)
        else:
            local_p = np.ones_like(density)

        if is_annihilator(op, base.fill_value):
            combined_p = np.prod(local_p, axis=reduce_axes)
        elif is_identity(op, base.fill_value):
            combined_p = 1 - np.prod(1 - local_p, axis=reduce_axes)
        else:
            combined_p = np.mean(local_p, axis=reduce_axes)

        new_vol_grid = _block_volume_grid(base.index_order, new_block_sizes)
        nnz_grid = combined_p.reshape(new_vol_grid.shape) * new_vol_grid

        return BlockedUniformStats(base, new_blocks_per_dim, nnz_grid, new_block_sizes)

    def relabel(
        self, stats: BlockedUniformStats, relabel_indices: tuple[Field, ...]
    ) -> BlockedUniformStats:
        if not isinstance(stats, BlockedUniformStats):
            raise TypeError("BlockedUniformStats args expected")
        base = self.relabel_def(stats, relabel_indices)
        name_map = dict(zip(stats.index_order, relabel_indices, strict=True))
        new_blocks_per_dim = {name_map[k]: v for k, v in stats.blocks_per_dim.items()}
        new_block_sizes = {name_map[k]: v for k, v in stats.block_sizes.items()}
        return BlockedUniformStats(
            base, new_blocks_per_dim, stats.nnz_grid.copy(), new_block_sizes
        )

    def reorder(
        self, stats: BlockedUniformStats, reorder_indices: tuple[Field, ...]
    ) -> BlockedUniformStats:
        if not isinstance(stats, BlockedUniformStats):
            raise TypeError("BlockedUniformStats args expected")
        base = self.reorder_def(stats, reorder_indices)
        old_order = stats.index_order
        dropped = [
            i for i, idx in enumerate(old_order) if idx not in set(reorder_indices)
        ]
        axes_mapping = [
            old_order.index(idx) for idx in reorder_indices if idx in old_order
        ] + dropped

        new_nnz_grid = np.transpose(stats.nnz_grid, axes=axes_mapping)
        expanded_shape = [stats.blocks_per_dim.get(idx, 1) for idx in reorder_indices]
        new_nnz_grid = new_nnz_grid.reshape(expanded_shape)

        new_block_sizes = {}
        for idx in reorder_indices:
            if idx in stats.block_sizes:
                new_block_sizes[idx] = stats.block_sizes[idx]
            else:
                new_block_sizes[idx] = np.array([base.dim_sizes[idx]], dtype=float)
        new_block_per_dim = {
            idx: stats.blocks_per_dim.get(idx, 1) for idx in reorder_indices
        }

        return BlockedUniformStats(
            base, new_block_per_dim, new_nnz_grid, new_block_sizes
        )


class BlockedUniformStats(NumericStats):
    def __init__(
        self,
        base: BaseTensorStats,
        blocks_per_dim: dict[Field, int],
        nnz_grid: np.ndarray,
        block_sizes: dict[Field, np.ndarray],
    ):
        super().__init__(base)
        self.blocks_per_dim = blocks_per_dim
        self.nnz_grid = nnz_grid
        self.block_sizes = block_sizes

    def _block_volume_grid(self) -> np.ndarray:
        return _block_volume_grid(self.index_order, self.block_sizes)

    def density_grid(self) -> np.ndarray:
        vol = self._block_volume_grid()
        return np.divide(
            self.nnz_grid, vol, out=np.zeros_like(self.nnz_grid), where=vol > 0
        )

    def _align_density(self, base_index_order: tuple[Field, ...]) -> np.ndarray:
        # accounts for broadcasting, putting 1 for dimensions that doesn't exist
        density = self.density_grid()
        present = [idx for idx in base_index_order if idx in self.index_order]
        perm = [self.index_order.index(idx) for idx in present]
        density = np.transpose(
            density, perm
        )  # making sure we have the index in correct order

        shape = []
        it = iter(density.shape)
        shape = [
            shape.append(next(it) if idx in self.index_order else 1)
            for idx in base_index_order
        ]
        return density.reshape(shape)

    def estimate_non_fill_values(self) -> float:
        return float(np.sum(self.nnz_grid))

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        density = self.density_grid().ravel()
        density_part = np.log2(density + 1)
        size_part = np.log2(sizes)
        return np.concatenate([size_part, density_part])

    def copy(self) -> BlockedUniformStats:
        return BlockedUniformStats(
            self,
            self.blocks_per_dim.copy(),
            self.nnz_grid.copy(),
            {k: v.copy() for k, v in self.block_sizes.items()},
        )
