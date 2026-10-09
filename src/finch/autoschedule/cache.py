import logging

import numpy as np
from numpy.linalg import vector_norm

from finch.algebra.tensor import TensorFType
from finch.autoschedule.tensor_stats.numeric_stats import NumericStats
from finch.finch_logic import (
    Alias,
    LogicLoader,
    LogicStatement,
    StatsFactory,
    TensorStats,
)
from finch.symbolic.stage import UnvalidatedForm
from finch.util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)
LOG_FLOOR = -64.0
LOG_CEIL = 64.0


class LogicCacheLRU_Embeddings_Norms_Matrix(UnvalidatedForm, LogicLoader):
    def __init__(
        self,
        ctx: LogicLoader,
        max_depth: int = 10,
        threshold: float = 1.0,
        norm_order: float = np.inf,
    ):
        self.ctx = ctx
        self.max_depth = max_depth
        self.cache: dict[tuple, dict] = {}
        self.threshold = threshold
        self.norm_order = norm_order

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        prgm_key = (prgm, tuple(bindings.items()), stats_factory)
        entry = self.cache.setdefault(prgm_key, {"cached_emb_matrix":None, "kernels":[]})

        current_vec = None
        parts = [
            s.get_embedding() for s in stats.values() if isinstance(s, NumericStats)
        ]
        if parts:
            embedding = np.concatenate(parts).astype(float)

            embedding = np.nan_to_num(
                embedding, nan=LOG_FLOOR, neginf=LOG_FLOOR, posinf=LOG_CEIL
            )
            factor = vector_norm(np.ones(len(embedding)), ord=self.norm_order)
            current_vec = embedding / factor

        kernels = entry["kernels"]
        idx = None
        if kernels and current_vec is None:
            idx = len(kernels) - 1
        elif kernels:
            distances = vector_norm(np.abs(entry["cached_emb_matrix"] - current_vec), 
                            ord=self.norm_order,
                            axis=1)
            
            chosen_idx = int(np.argmin(distances))
            if distances[chosen_idx] < self.threshold:
                idx = chosen_idx
        if idx is not None:
            logger.debug("CacheLRU_Embeddings_Norms HIT, reusing kernel")
            kernels.append(kernels.pop(idx))
            if entry["cached_emb_matrix"] is not None:
                m = entry["cached_emb_matrix"]
                entry["cached_emb_matrix"] = np.concatenate((m[:idx],m[idx+1:],m[idx:idx+1]))
            return kernels[-1]

        logger.debug(
            "CacheLRU_Embeddings_Norms MISS, compiling new kernel and embeddings"
        )
        result = self.ctx(prgm, bindings, stats, stats_factory)
        kernels.append(result)
        if current_vec is not None:
            row = current_vec[None, :]
            entry["cached_emb_matrix"] = (
                row if entry["cached_emb_matrix"] is None else np.vstack((entry["cached_emb_matrix"], row))
            )
        if len(kernels) > self.max_depth:
            kernels.pop(0)  # evict the least recently used
            if entry["cached_emb_matrix"] is not None:
                entry["cached_emb_matrix"] = entry["cached_emb_matrix"][1:]
        return result

