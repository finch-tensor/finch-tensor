from abc import abstractmethod
from collections.abc import Iterable

import numpy as np

from finch.finch_logic import Field

from .tensor_stats import BaseTensorStats


class NumericStats(BaseTensorStats):
    @abstractmethod
    def estimate_non_fill_values(self, max: Iterable[Field] = ()) -> float:
        """
        Return an estimate on the number of non-fill values. If `max` lists
        fields, estimate the largest number of non-fill values in any slice
        which fixes those fields instead.
        """
        ...

    @abstractmethod
    def get_embedding(self) -> np.ndarray:
        """
        Returns vector embedding for the stat.
        """
        ...
