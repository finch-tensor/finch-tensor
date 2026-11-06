from collections import Counter

from finch.algebra.tensor import TensorFType
from finch.autoschedule.fuse_merger.fuse_merger import statements
from finch.autoschedule.stages import LoopOrderedForm
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import (
    Alias,
    Field,
    Fuse,
    FusedAlias,
    HardAlias,
    LogicLoader,
    LogicStatement,
    Plan,
    Query,
    QueryInto,
    StatsFactory,
    Table,
    TensorStats,
)
from finch.symbolic import Namespace, PostOrderDFS, PostWalk, Rewrite


def common_reads(
    stmts: tuple[LogicStatement, ...], loops: tuple[Field, ...]
) -> list[Table]:
    """The tables which the queries of stmts read more than once, whose fields
    are those of the enclosing Fuse nodes loops."""
    counts: Counter[Table] = Counter()
    for stmt in stmts:
        for query in PostOrderDFS(stmt):
            match query:
                case Query(_, rhs) | QueryInto(_, _, rhs):
                    for node in PostOrderDFS(rhs):
                        match node:
                            case Table(Alias(), idxs) if set(idxs) == set(loops):
                                counts[node] += 1
    return [table for table, n in counts.items() if n > 1]


def extract_common_reads(prgm: LogicStatement) -> LogicStatement:
    """
    Copy each table which the body of a Fuse of prgm reads more than once, over
    exactly the fields of its enclosing Fuse nodes, into a temporary at the
    start of the body, and read the temporary instead. The temporary is fused
    over every enclosing field, so it holds one scalar for each iteration, and
    the tensor is accessed once per iteration however many statements read it.
    """
    spc = Namespace(prgm)

    def extract(node: LogicStatement, loops: tuple[Field, ...]) -> LogicStatement:
        match node:
            case Plan(bodies):
                return Plan(tuple(extract(body, loops) for body in bodies))
            case Fuse(idx, body):
                loops = (*loops, idx)
                body = extract(body, loops)
                stmts = statements(body)
                temps: dict[Table, Table] = {}
                for table in common_reads(stmts, loops):
                    match table:
                        case Table(Alias(name=name), _):
                            temp = HardAlias(spc.freshen(name, "cse"))
                            temps[table] = Table(FusedAlias(temp, len(loops)), loops)
                if not temps:
                    return Fuse(idx, body)
                copies = tuple(Query(temp, table) for table, temp in temps.items())
                reads = Rewrite(PostWalk(temps.get))
                return Fuse(idx, Plan((*copies, *(reads(stmt) for stmt in stmts))))
        return node

    return extract(prgm, ())


class CSEExtractor(LoopOrderedForm, LogicLoader):
    """
    Pass a program in LoopOrderedForm to ctx, after reading the tables which
    the body of a Fuse reads more than once into temporaries, as
    extract_common_reads does, so that a Fuse merged by LogicFuseMerger
    accesses each tensor it shares once per iteration.
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
        return self.ctx(extract_common_reads(prgm), bindings, stats, stats_factory)
