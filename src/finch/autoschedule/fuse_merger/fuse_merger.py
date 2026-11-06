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


def sibling_merges(node: LogicStatement) -> Iterator[LogicStatement]:
    """Each program which merges one pair of adjacent sibling Fuse nodes of
    node, as merge_fuses does, outermost pairs first."""
    match node:
        case Plan(bodies):
            for n, (first, second) in enumerate(zip(bodies, bodies[1:], strict=False)):
                match first, second:
                    case Fuse(), Fuse():
                        merged = merge_fuses(first, second)
                        if merged is not None:
                            yield Plan((*bodies[:n], merged, *bodies[n + 2 :]))
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
    Merge adjacent sibling Fuse nodes of prgm, which have the same enclosing
    Fuse nodes and whose fields read the same tensors in the same positions,
    one pair at a time, as long as legal accepts the merged program, until no
    pair merges. A merge runs the bodies of both for each value of one field,
    so a single pass over the tensors they read serves both.
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
