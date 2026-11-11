from collections.abc import Callable, Iterator

from finch.algebra.tensor import TensorFType
from finch.autoschedule.stages import LoopOrderedForm
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import (
    Alias,
    Fuse,
    LogicLoader,
    LogicStatement,
    Plan,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import PostOrderDFS, PostWalk, Rewrite


def statements(body: LogicStatement) -> tuple[LogicStatement, ...]:
    match body:
        case Plan(bodies):
            return bodies
    return (body,)


def field_reads(fuse: Fuse) -> frozenset[tuple[Alias, int]]:
    """The tensors which the queries of fuse read with its field, each with
    the position of the field in their tables."""
    reads = set()
    for node in PostOrderDFS(fuse.body):
        match node:
            case Query(_, rhs) | QueryInto(_, _, rhs):
                for table in PostOrderDFS(rhs):
                    match table:
                        case Table(Alias() as tns, idxs) if fuse.idx in idxs:
                            reads.add((tns.unfused, idxs.index(fuse.idx)))
    return frozenset(reads)


def merge_fuses(first: Fuse, second: Fuse) -> Fuse | None:
    """
    A Fuse which runs the bodies of first and second, in order, for each value
    of the field of first, which renames the field of second, or None if
    their fields read different tensors or positions, or the field of first
    already occurs in second.
    """
    reads = field_reads(first)
    if not reads or field_reads(second) != reads:
        return None
    idx, other = first.idx, second.idx
    if idx != other and any(node == idx for node in PostOrderDFS(second.body)):
        return None
    body = Rewrite(PostWalk(lambda node: idx if node == other else None))(second.body)
    return Fuse(idx, Plan((*statements(first.body), *statements(body))))


def tensor_accesses(stmt: LogicStatement) -> tuple[set[Alias], set[Alias]]:
    """The tensors which the queries of stmt read and those which they write."""
    reads: set[Alias] = set()
    writes: set[Alias] = set()
    for node in PostOrderDFS(stmt):
        match node:
            case Query(Table(Alias() as lhs, _), rhs):
                writes.add(lhs.unfused)
            case QueryInto(Table(Alias() as lhs, _), _, rhs):
                writes.add(lhs.unfused)
                reads.add(lhs.unfused)
            case _:
                continue
        reads.update(var.unfused for var in PostOrderDFS(rhs) if isinstance(var, Alias))
    return reads, writes


def depends(first: LogicStatement, second: LogicStatement) -> bool:
    """Whether second must run after first, as it reads or writes a tensor which
    first writes, or writes a tensor which first reads."""
    reads, writes = tensor_accesses(first)
    later_reads, later_writes = tensor_accesses(second)
    return bool(writes & (later_reads | later_writes) or reads & later_writes)


def split_between(
    first: LogicStatement,
    between: tuple[LogicStatement, ...],
    second: LogicStatement,
) -> tuple[list[LogicStatement], list[LogicStatement]] | None:
    """
    The statements of between which may run before first, and those which
    must still run after it, in order, so that second may run next to first,
    or None if second must run after one which must run after first.
    """
    before: list[LogicStatement] = []
    after: list[LogicStatement] = []
    for stmt in between:
        if any(depends(prev, stmt) for prev in (first, *after)):
            after.append(stmt)
        else:
            before.append(stmt)
    if any(depends(stmt, second) for stmt in after):
        return None
    return before, after


def sibling_merges(node: LogicStatement) -> Iterator[LogicStatement]:
    """
    Each program which merges one pair of Fuse nodes of a Plan of node, as
    merge_fuses does, outermost pairs first. The statements between them which
    don't depend on the first move before it, and the rest after the merged
    Fuse, as long as the second depends on none of the rest.
    """
    match node:
        case Plan(bodies):
            for n, first in enumerate(bodies):
                for m in range(n + 1, len(bodies)):
                    match first, bodies[m]:
                        case Fuse(), Fuse() as second:
                            merged = merge_fuses(first, second)
                            if merged is None:
                                continue
                            split = split_between(first, bodies[n + 1 : m], second)
                            if split is None:
                                continue
                            before, after = split
                            yield Plan(
                                (
                                    *bodies[:n],
                                    *before,
                                    merged,
                                    *after,
                                    *bodies[m + 1 :],
                                )
                            )
            for n, body in enumerate(bodies):
                for merged in sibling_merges(body):
                    yield Plan((*bodies[:n], merged, *bodies[n + 1 :]))
        case Fuse(idx, body):
            for merged in sibling_merges(body):
                yield Fuse(idx, merged)


def merge_sibling_fuses(
    prgm: LogicStatement, legal: Callable[[LogicStatement], bool]
) -> LogicStatement:
    """
    Merge Fuse nodes of the same Plan of prgm, which have the same enclosing
    Fuse nodes and whose fields read the same tensors in the same positions,
    one pair at a time, as long as legal accepts the merged program, until no
    pair merges. A merge runs the bodies of both for each value of one field,
    so a single pass over the tensors they read serves both. The pair needn't
    be adjacent, as sibling_merges moves the statements between them.
    """
    while True:
        for merged in sibling_merges(prgm):
            if legal(merged):
                prgm = merged
                break
        else:
            return prgm


class LogicFuseMerger(LoopOrderedForm, LogicLoader):
    """
    Pass a program in LoopOrderedForm to ctx, after merging its sibling Fuse
    nodes, as merge_sibling_fuses does, wherever the merged program is still
    in LoopOrderedForm, whose Fuse nodes may only run statements together when
    running them for each value of the field in turn means the same.
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
        def legal(merged: LogicStatement) -> bool:
            assert isinstance(merged, Plan)
            try:
                LoopOrderedForm.validate_inputs(merged, bindings, stats, stats_factory)
            except ValueError:
                return False
            return True

        prgm = merge_sibling_fuses(prgm, legal)
        return self.ctx(prgm, bindings, stats, stats_factory)
