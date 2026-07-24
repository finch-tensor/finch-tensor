from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from finch.finch_logic import Field

from .tensor_stats import BaseTensorStats
from .uniform_stats import UniformStats


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

    data = np.asarray(data)
    nnz_grid = (data != d.fill_value).astype(np.int64)

    for axis in reversed(range(len(index_order))):
        idx = index_order[axis]
        nnz_grid = np.add.reduceat(nnz_grid, block_starts[idx], axis=axis)

    grid_dim = [blocks_per_dim[idx] for idx in index_order]
    blocks_grid = np.empty(grid_dim, dtype=object)

    for coord in np.ndindex(*grid_dim):
        block_dim_sizes = {}
        for i, idx in enumerate(index_order):
            start = math.floor(coord[i] * base_block_size[idx])
            end = int(
                d.dim_sizes[idx]
                if coord[i] == (blocks_per_dim[idx] - 1)
                else math.floor((coord[i] + 1) * base_block_size[idx])
            )
            block_dim_sizes[idx] = float(end - start)
        block_def = BaseTensorStats(
            index_order=index_order, dim_sizes=block_dim_sizes, fill_value=d.fill_value
        )
        blocks_grid[coord] = UniformStats(block_def, nnz=float(nnz_grid[coord]))
    return blocks_grid
