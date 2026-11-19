from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg.blas import get_blas_funcs

from finch import ffuncs
from finch.algebra import FType, ftypes
from finch.algebra.tensor import TensorFType
from finch.finch_logic import (
    Aggregate,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Query,
    Table,
)
from finch.tensor.bufferized_ndarray import BufferizedNDArrayFType


@dataclass(frozen=True)
class LibraryKernel:
    """
    A function of an external library which computes a logic query.

    Attributes:
        name: The name of the kernel.
        pattern: The query which the kernel computes, given the values of its
            parameters as keywords. The aliases of its tables name the operands.
        params: The values which each parameter may take.
        accepts: Whether the kernel takes operands of the given formats, by
            name, and computes elements of the given type.
        call: Computes the query, given its operands as NumPy arrays by name,
            a scalar `alpha` which multiplies the result, and the parameters,
            all as keywords. Returns the result as a NumPy array.
    """

    name: str
    pattern: Callable[..., Query]
    params: Mapping[str, tuple[Any, ...]]
    accepts: Callable[[dict[str, TensorFType], FType], bool]
    call: Callable[..., np.ndarray]


def dense_matrices_of(*types: FType) -> Callable[[dict[str, TensorFType], FType], bool]:
    """Accepts dense matrices of one of types, which is also the result type."""

    def accepts(operands: dict[str, TensorFType], out_type: FType) -> bool:
        if out_type not in types:
            return False
        for fmt in operands.values():
            match fmt:
                case BufferizedNDArrayFType() if (
                    fmt.ndim == 2 and fmt.element_type == out_type
                ):
                    pass
                case _:
                    return False
        return True

    return accepts


i, j, k = Field("i"), Field("j"), Field("k")
A, B, C = HardAlias("A"), HardAlias("B"), HardAlias("C")


def gemm_pattern(trans_a: bool, trans_b: bool) -> Query:
    """`C[i, j] = Σ_k op(A)[i, k] * op(B)[k, j]`, where op transposes if flagged."""
    a = Table(A, (k, i) if trans_a else (i, k))
    b = Table(B, (j, k) if trans_b else (k, j))
    product = MapJoin(Literal(ffuncs.mul), (a, b))
    return Query(
        Table(C, (i, j)), Aggregate(Literal(ffuncs.add), Literal(0), product, (k,))
    )


def gemm(
    A: np.ndarray, B: np.ndarray, *, alpha: Any, trans_a: bool, trans_b: bool
) -> np.ndarray:
    # BLAS stores matrices by column, so it reads a row-major array as its
    # transpose. It computes C^T = op(B)^T op(A)^T from the arrays without
    # copying them, and C^T stored by column is C stored by row.
    blas_gemm = get_blas_funcs("gemm", (A, B))
    return blas_gemm(alpha, B.T, A.T, trans_a=trans_b, trans_b=trans_a).T


GEMM = LibraryKernel(
    name="gemm",
    pattern=gemm_pattern,
    params={"trans_a": (False, True), "trans_b": (False, True)},
    accepts=dense_matrices_of(ftypes.float32, ftypes.float64),
    call=gemm,
)

BLAS_KERNELS = (GEMM,)
