from functools import wraps

import pytest

from finch import ffuncs
from finch.algebra import FinchOperator, SingletonOperatorFType


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


class _BumpFType(SingletonOperatorFType):
    @property
    def operator(self):
        return bump

    def return_type(self, *args):
        return ffuncs.add.ftype.return_type(*args)


class _Bump(FinchOperator):
    """adds its arguments, but its ftype doesn't say it's pure"""

    @property
    def ftype(self):
        return _BumpFType()

    def __call__(self, *args):
        return ffuncs.add(*args)


bump = _Bump()
