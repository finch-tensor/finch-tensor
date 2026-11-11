from collections import Counter

from finch.algebra.tensor import TensorFType
from finch.autoschedule.fuse_merger.fuse_merger import statements
from finch.autoschedule.stages import LoopOrderedForm
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import (
    Aggregate,
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


def written(stmt: LogicStatement) -> set[Alias]:
    """The aliases which the queries of stmt write."""
    found: set[Alias] = set()
    for node in PostOrderDFS(stmt):
        match node:
            case Query(Table(Alias() as lhs, _), _) | QueryInto(
                Table(Alias() as lhs, _), _, _
            ):
                found.add(lhs.unfused)
    return found


def read(stmt: LogicStatement) -> set[Alias]:
    """The aliases which the queries of stmt read."""
    found: set[Alias] = set()
    for node in PostOrderDFS(stmt):
        match node:
            case Query(_, rhs) | QueryInto(_, _, rhs):
                for leaf in PostOrderDFS(rhs):
                    match leaf:
                        case Table(Alias() as tns, _):
                            found.add(tns.unfused)
    return found


def sequence(stmts: tuple[LogicStatement, ...]) -> LogicStatement:
    """stmts as one statement."""
    return stmts[0] if len(stmts) == 1 else Plan(stmts)


def accumulate_in_scalars(prgm: LogicStatement) -> LogicStatement:
    """
    Reduce each query of an inner Fuse of prgm whose output has exactly the
    fields of the enclosing Fuse nodes into a temporary fused over those
    fields, and copy the temporary into the output once the inner loop ends.
    The temporary holds one scalar for each iteration of the enclosing loops,
    so the reduction accumulates in a scalar rather than updating the output
    at every iteration of the inner loop.
    """
    spc = Namespace(prgm)

    def accumulators(
        inner: LogicStatement, loops: tuple[Field, ...]
    ) -> dict[Alias, Table]:
        """The temporaries of the outputs of the queries of inner which reduce
        into exactly loops, and which no other statement of inner touches."""
        stmts = statements(inner)
        temps: dict[Alias, Table] = {}
        for k, stmt in enumerate(stmts):
            match stmt:
                case Query(Table(Alias() as lhs, idxs), Aggregate()) if (
                    not isinstance(lhs, FusedAlias) and idxs == loops
                ):
                    others = (*stmts[:k], *stmts[k + 1 :])
                    if not any(
                        lhs in written(s) or lhs in read(s) for s in others
                    ) and lhs not in read(stmt):
                        temp = HardAlias(spc.freshen(lhs.name, "acc"))
                        temps[lhs] = Table(FusedAlias(temp, len(loops)), loops)
        return temps

    def retarget(stmt: LogicStatement, temps: dict[Alias, Table]) -> LogicStatement:
        match stmt:
            case Query(Table(lhs, _), rhs) if lhs in temps:
                return Query(temps[lhs], rhs)
        return stmt

    def accumulate(node: LogicStatement, loops: tuple[Field, ...]) -> LogicStatement:
        match node:
            case Plan(bodies):
                return Plan(tuple(accumulate(body, loops) for body in bodies))
            case Fuse(idx, body):
                loops = (*loops, idx)
                stmts: list[LogicStatement] = []
                for stmt in statements(accumulate(body, loops)):
                    match stmt:
                        case Fuse(inner_idx, inner) if temps := accumulators(
                            inner, loops
                        ):
                            inner = sequence(
                                tuple(retarget(s, temps) for s in statements(inner))
                            )
                            stmts.append(Fuse(inner_idx, inner))
                            stmts.extend(
                                Query(Table(out, loops), temp)
                                for out, temp in temps.items()
                            )
                        case _:
                            stmts.append(stmt)
                return Fuse(idx, sequence(tuple(stmts)))
        return node

    return accumulate(prgm, ())


class CSEExtractor(LoopOrderedForm, LogicLoader):
    """
    Pass a program in LoopOrderedForm to ctx, after reading the tables which
    the body of a Fuse reads more than once into temporaries, as
    extract_common_reads does, so that a Fuse merged by LogicFuseMerger
    accesses each tensor it shares once per iteration, and reducing into
    scalars the outputs which an inner loop reduces into, as
    accumulate_in_scalars does.
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
        prgm = accumulate_in_scalars(extract_common_reads(prgm))
        return self.ctx(prgm, bindings, stats, stats_factory)
