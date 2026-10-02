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

class LogicCacheLRU_Embeddings_Norms(UnvalidatedForm, LogicLoader):
    def __init__(
        self,
        ctx: LogicLoader,
        max_depth: int = 10,
        threshold: float = 1.0,
        norm_order: float = np.inf,
    ):
        self.ctx = ctx
        self.max_depth = max_depth
        self.cache: dict[tuple, list[tuple]] = {}
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
        entries = self.cache.setdefault(prgm_key,[])

        current_vec = None
        parts = [
            s.get_embedding()
            for s in stats.values()
            if isinstance(s, NumericStats)

        ]
        if parts : 
            embedding = np.concatenate(parts).astype(float)

            embedding = np.nan_to_num(embedding,nan=LOG_FLOOR,neginf=LOG_FLOOR,posinf=LOG_CEIL)
            factor = vector_norm(np.ones(len(embedding)), ord=self.norm_order)
            current_vec = embedding / factor

        idx = None
        if entries and current_vec is None:
            idx = len(entries)-1
        elif entries:
            distances = [
                vector_norm(np.abs(emb - current_vec), ord=self.norm_order)
                for emb, _ in entries
            ]
            chosen_idx = int(np.argmin(distances))
            if distances[chosen_idx] < self.threshold:
                idx = chosen_idx
        if idx is not None:
            logger.debug("CacheLRU_Embeddings_Norms HIT, reusing kernel")
            entries.append(entries.pop(idx))
            return entries[-1][1]

        logger.debug(
            "CacheLRU_Embeddings_Norms MISS, compiling new kernel and embeddings"
        )
        result = self.ctx(prgm, bindings, stats, stats_factory)

        entries.append((current_vec,result))
        if len(entries)> self.max_depth:
            entries.pop(0)
        return result
