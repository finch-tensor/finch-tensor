from functools import wraps

import pytest

import numpy as np

from finch.autoschedule.cost_models import LoopBasis
from finch.autoschedule.cost_models.loop_cost_models import transposed
from finch.autoschedule.stages import LoopOrderedForm


def mlir_backend(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        _ = pytest.importorskip("mlir", reason="MLIR Python bindings not installed.")
        return func(*args, **kwargs)

    return pytest.mark.mlir_backend(func_wrapper)


def with_workspaces(program, *args):
    """The arguments of a kernel, followed by the workspaces which it takes
    after them, such as the diagonal masks of concordization."""
    workspaces = program.funcs[0].args[len(args) :]
    return (
        *args,
        *(arg.result_type.construct((0,) * arg.result_type.ndim) for arg in workspaces),
    )


class TransposeBasis(LoopBasis, LoopOrderedForm):
    """
    The dense and sparse entries of the tables each prefix transposes, a basis
    of two features for the tests of the bases which combine others.
    """

    @property
    def n_features(self) -> int:
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("dense_transposes", "sparse_transposes")

    def get_component_features(self, component, machine) -> np.ndarray:
        features = np.zeros((1, self.n_features))
        for _, stat in transposed(component, machine):
            nnz = stat.estimate_non_fill_values()
            space = stat.get_dim_space_size(stat.index_order)
            dense = 0 < space < float("inf") and nnz / space > 0.1
            features[0, 0 if dense else 1] += nnz
        return features
