import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from finch.algebra.tensor import TensorFType
from finch.finch_logic import (
    Alias,
    Fuse,
    Literal,
    LogicExpression,
    LogicLoader,
    LogicStatement,
    Plan,
    Produces,
    Query,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import UnvalidatedForm
from finch.tensor.bufferized_ndarray import BufferizedNDArray
from finch.util.logging import LOG_LOGIC_POST_OPT

from .canonicalize import canonicalize
from .kernels import BLAS_KERNELS, LibraryKernel

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


@dataclass(frozen=True)
class LibraryCall:
    """
    A call to a library kernel, loaded in place of a compiled program. Like the
    library of a compiled program, its main takes the tensors which the program
    is bound to, in the order of `args`, and returns the tensors it produces.
    """

    kernel: LibraryKernel
    params: dict[str, Any]
    operands: dict[str, Alias]
    scalars: tuple[LogicExpression, ...]
    args: tuple[Alias, ...]

    def main(self, *args):
        values = dict(zip(self.args, args, strict=True))
        alpha = 1
        for scalar in self.scalars:
            match scalar:
                case Literal(val):
                    alpha *= val
                case Table(Alias() as tns, ()):
                    alpha *= np.asarray(values[tns])[()]
        operands = {
            name: np.asarray(values[tns]) for name, tns in self.operands.items()
        }
        result = self.kernel.call(**operands, alpha=alpha, **self.params)
        return (BufferizedNDArray.from_numpy(result),)


class KernelTable:
    """Library kernels, by the canonical form of each query they compute."""

    def __init__(self, kernels: Iterable[LibraryKernel]):
        self.entries: dict[
            Query, list[tuple[LibraryKernel, dict[str, Any], tuple[str, ...]]]
        ] = defaultdict(list)
        for kernel in kernels:
            for values in product(*kernel.params.values()):
                params = dict(zip(kernel.params, values, strict=True))
                canon = canonicalize(kernel.pattern(**params))
                if canon is None or canon.scalars:
                    raise ValueError(
                        f"The pattern of {kernel.name} with {params} has no "
                        "canonical form without scalars."
                    )
                names = tuple(tns.name for tns in canon.operands)
                self.entries[canon.term].append((kernel, params, names))

    def match(
        self, query: Query, bindings: dict[Alias, TensorFType]
    ) -> LibraryCall | None:
        """A call to the first kernel which computes query from bindings."""
        canon = canonicalize(query)
        if canon is None or not all(tns in bindings for tns in canon.operands):
            return None
        for scalar in canon.scalars:
            match scalar:
                case Literal(_):
                    pass
                case Table(Alias() as tns, ()) if tns in bindings:
                    pass
                case _:
                    return None
        out_type = query.rhs.element_type(
            {tns: fmt.element_type for tns, fmt in bindings.items()}
        )
        for kernel, params, names in self.entries.get(canon.term, ()):
            # A kernel which reads an operand twice must read one tensor twice.
            operands: dict[str, Alias] = {}
            for name, tns in zip(names, canon.operands, strict=True):
                if operands.setdefault(name, tns) != tns:
                    break
            else:
                formats = {name: bindings[tns] for name, tns in operands.items()}
                if kernel.accepts(formats, out_type):
                    return LibraryCall(
                        kernel, params, operands, canon.scalars, tuple(bindings)
                    )
        return None


def single_query(prgm: LogicStatement) -> Query | None:
    """The query of a program which is one query and its result, without loops."""
    match prgm:
        case Plan((stmt, Produces((Alias() as out,)))):
            while True:
                match stmt:
                    case Fuse(_, body):
                        stmt = body
                    case Query(Table(lhs, _), _) if lhs == out:
                        return stmt
                    case _:
                        return None
    return None


class LogicKernelDispatcher(UnvalidatedForm, LogicLoader):
    """
    Load a program which is a single query that a library kernel computes as a
    call to that kernel, and pass any other program to ctx. A query matches a
    kernel up to the differences which canonicalize removes, if the kernel
    accepts the formats of the tensors it reads.
    """

    def __init__(
        self, ctx: LogicLoader, kernels: Iterable[LibraryKernel] = BLAS_KERNELS
    ):
        self.ctx = ctx
        self.table = KernelTable(kernels)

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        query = single_query(prgm)
        if query is not None:
            call = self.table.match(query, bindings)
            if call is not None:
                logger.debug(
                    "Calling %s with %s to compute %s",
                    call.kernel.name,
                    call.params,
                    query.lhs.tns,
                )
                return call, dict(bindings)
        return self.ctx(prgm, bindings, stats, stats_factory)
