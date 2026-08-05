"""
Shared functionality across TensorStats implementations.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from finch.algebra import Tensor, ffuncs
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finch.tensor import BufferizedNDArray


def _scalar(value) -> Table:
    """A 0-d table holding ``value``, so it broadcasts against any MapJoin arg."""
    return Table(Literal(BufferizedNDArray.from_numpy(np.asarray(value))), ())


def get_lp_norms(
    arr: Tensor, fields: Iterable[Field], norms: Iterable[float]
) -> dict[Field, list[float]]:
    """Scan ``arr`` and return the Lp norms of each axis' degree sequence.

    The degree sequence of axis ``i`` counts, for each index of that axis, how
    many non-fill entries carry it. Its norms are what :class:`DC` degree
    constraints are built from:

    * ``p == 0`` -- the number of indices with at least one non-fill entry,
      i.e. the size of the projection onto that axis.
    * ``p == inf`` -- the largest degree.
    * otherwise -- ``(sum(degree ** p)) ** (1 / p)``.

    Args:
        arr: The tensor to scan.
        fields: The axis names of ``arr`` (one per dimension), in order.
        norms: The Lp norms to compute.

    Returns:
        A dictionary mapping each field to its norms, ordered as ``norms``.
    """
    from finch.autoschedule.default_schedulers import NON_RECURSIVE_SCHEDULER

    fields = tuple(fields)
    norms = tuple(norms)
    if not fields or not norms:
        return {field: [] for field in fields}

    int_zero = Literal(np.int64(0))
    float_zero = Literal(np.float64(0.0))
    # Indicator of the non-fill structure. Left inlined in each reduction below
    # rather than bound to its own alias, so no full-size temporary is built.
    non_fill = MapJoin(
        Literal(ffuncs.ne),
        (Table(Literal(arr), fields), _scalar(arr.fill_value)),
    )

    bodies: list[Query] = []
    outputs: list[Alias] = []
    for dim, field in enumerate(fields):
        degrees = Alias(f"degrees_{dim}")
        bodies.append(
            Query(
                degrees,
                Aggregate(
                    Literal(ffuncs.add),
                    int_zero,
                    non_fill,
                    tuple(f for f in fields if f != field),
                ),
            )
        )
        degree_table = Table(degrees, (field,))

        for k, norm in enumerate(norms):
            out = Alias(f"norm_{dim}_{k}")
            rhs: Aggregate
            if norm == 0:
                rhs = Aggregate(
                    Literal(ffuncs.add),
                    int_zero,
                    MapJoin(Literal(ffuncs.ne), (degree_table, _scalar(np.int64(0)))),
                    (field,),
                )
            elif np.isinf(norm):
                # Degrees are non-negative, so 0 is a safe identity for max.
                rhs = Aggregate(Literal(ffuncs.max), int_zero, degree_table, (field,))
            else:
                # The outer ** (1 / norm) is applied to the scalar result below.
                rhs = Aggregate(
                    Literal(ffuncs.add),
                    float_zero,
                    MapJoin(
                        Literal(ffuncs.pow), (degree_table, _scalar(np.float64(norm)))
                    ),
                    (field,),
                )
            bodies.append(Query(out, rhs))
            outputs.append(out)

    prgm = Plan((*bodies, Produces(tuple(outputs))))
    results = NON_RECURSIVE_SCHEDULER(prgm)

    flat: list[float] = []
    all_norms = [norm for _ in fields for norm in norms]
    for result, norm in zip(results, all_norms, strict=True):
        value = float(np.asarray(result)[()])
        if norm != 0 and not np.isinf(norm):
            value = value ** (1.0 / norm)
        flat.append(value)

    return {
        field: flat[i * len(norms) : (i + 1) * len(norms)]
        for i, field in enumerate(fields)
    }
