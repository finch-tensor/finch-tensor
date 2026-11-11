from collections.abc import Mapping

from finch import ffuncs
from finch.algebra import TensorFType
from finch.algebra.algebra import is_identity
from finch.algebra.fill import is_static
from finch.autoschedule.cost_models import DenseLevelOption, level_options
from finch.autoschedule.stages import FormattedForm
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Fuse,
    Literal,
    LogicExpression,
    LogicLoader,
    LogicStatement,
    MapJoin,
    Plan,
    Query,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import PostOrderDFS
from finch.tensor.bufferized_ndarray import BufferizedNDArrayFType


def dropping_fill(arg: LogicExpression, value) -> LogicExpression:
    """arg, marked so that a query which writes it writes it only where it isn't
    value."""
    return MapJoin(Literal(ffuncs.drop_fill(value)), (arg,))


def drops(op, value, reduced: tuple[Field, ...], fmt: TensorFType) -> bool:
    """
    Whether a query which aggregates terms with op over reduced into an output
    of format fmt may skip writing the terms equal to value: those which leave
    the entry as it is, the identity of op, or, where nothing is reduced and op
    overwrites each entry once, the fill of fmt, as an entry left unwritten is.
    """
    if is_identity(op.ftype, value):
        return True
    fill = fmt.fill_type
    return (
        op == ffuncs.overwrite
        and not reduced
        and is_static(fill)
        and bool(ffuncs.same(value, fill.value))
    )


def stores_densely(fmt: TensorFType, depth: int) -> bool:
    """Whether fmt stores the field at a zero-based stored depth in a dense
    level."""
    match fmt:
        case BufferizedNDArrayFType():
            return True
    match (level_options(fmt) or [])[depth : depth + 1]:
        case [DenseLevelOption()]:
            return True
    return False


def stores_sparsely(fmt: TensorFType) -> bool:
    """Whether fmt stores a field in a sparse level, which stores only the
    entries written to it, so that a fill written to it costs room."""
    for option in level_options(fmt) or []:
        match option:
            case DenseLevelOption():
                pass
            case _:
                return True
    return False


def reads_densely(
    arg: LogicExpression, bindings: Mapping[Alias, TensorFType], idx: Field
) -> bool:
    """Whether a table which arg reads stores idx in a dense level, so that a
    loop over idx visits every index of it, whether or not arg is a fill
    there."""
    for node in PostOrderDFS(arg):
        match node:
            case Table(Alias() as tns, idxs) if (
                idx in idxs[tns.nfused :] and tns.unfused in bindings
            ):
                depth = idxs.index(idx) - tns.nfused
                if stores_densely(bindings[tns.unfused], depth):
                    return True
    return False


def drop_fills(prgm: Plan, bindings: Mapping[Alias, TensorFType]) -> Plan:
    """
    Mark the argument of each aggregate whose output stores a field in a
    sparse level, and which reads a table whose level of the field of the
    innermost loop is dense, so that it is written only where it isn't its
    static fill, as dropping_fill does, where drops lets the query skip those
    writes. The dense loop computes the fills of the argument, which the query
    then drops within the loop, rather than storing them in the output. A copy
    into such an output from a workspace fused over every field it stores is
    marked likewise.
    """
    fills = {alias: fmt.fill_type for alias, fmt in bindings.items()}

    def rewrite(stmt: LogicStatement, loop_order: tuple[Field, ...]) -> LogicStatement:
        match stmt:
            case Plan(bodies):
                return Plan(tuple(rewrite(body, loop_order) for body in bodies))
            case Fuse(idx, body):
                return Fuse(idx, rewrite(body, (*loop_order, idx)))
            case Query(
                Table(Alias() as tns, _) as lhs,
                Aggregate(Literal(op) as agg_op, init, arg, idxs),
            ) if (
                loop_order
                and tns.unfused in bindings
                and stores_sparsely(bindings[tns.unfused])
                and reads_densely(arg, bindings, loop_order[-1])
            ):
                fill = arg.fill_type(fills)
                if is_static(fill) and drops(
                    op, fill.value, idxs, bindings[tns.unfused]
                ):
                    arg = dropping_fill(arg, fill.value)
                    return Query(lhs, Aggregate(agg_op, init, arg, idxs))
            case Query(
                Table(Alias() as tns, _) as lhs,
                Table(Alias() as src, src_idxs) as arg,
            ) if (
                loop_order
                and len(src_idxs) == src.nfused
                and tns.unfused in bindings
                and src.unfused in bindings
                and stores_sparsely(bindings[tns.unfused])
            ):
                # A copy out of a workspace which stores nothing, an
                # accumulator, writes one entry in each fused iteration, which
                # is a fill where nothing was accumulated.
                fill = bindings[src.unfused].fill_type
                if is_static(fill) and drops(
                    ffuncs.overwrite, fill.value, (), bindings[tns.unfused]
                ):
                    return Query(lhs, dropping_fill(arg, fill.value))
        return stmt

    return Plan(tuple(rewrite(body, ()) for body in prgm.bodies))


class LogicFillDropper(FormattedForm, LogicLoader):
    """
    Pass a program in FormattedForm to ctx, after marking the fills its
    aggregates compute over dense loops to be dropped, as drop_fills does.
    """

    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[Alias, TensorFType]]:
        assert isinstance(prgm, Plan)
        return self.ctx(drop_fills(prgm, bindings), bindings, stats, stats_factory)
