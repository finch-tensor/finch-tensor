"""
Check the cost models against the operations which example programs make
under the assembly interpreter, counted by loop and by buffer.
"""

import re
from collections import Counter

import numpy as np

import finch as fl
from finch import ffuncs
from finch import finch_assembly as asm
from finch.algebra import TensorFType
from finch.autoschedule import CompilerFormLowerer, LogicCapture, NotationGenerator
from finch.autoschedule.cost_models import (
    DenseLevelOption,
    LevelOption,
    PigeonBasis,
    ProtocolBasis,
    ProtocolChoice,
    SparseListLevelOption,
    WithCopies,
)
from finch.autoschedule.cost_models.loop_cost_models import transpose_copy
from finch.autoschedule.formatter.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer.loop_ordering import AbstractLoopOrderer
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finch.autoschedule.tensor_stats.stats_interpreter import StatsMachine
from finch.autoschedule.util import fuse_loop_order
from finch.compile import NotationCompiler
from finch.finch_assembly.interpreter import HaltState
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Follow,
    Gallop,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    QueryInto,
    Table,
    TensorStats,
    Walk,
)
from finch.symbolic import PostOrderDFS

from .test_sparse_update import run_copy, tensors
from .utils import with_workspaces


class CountingInterpreter(asm.AssemblyInterpreter):
    """
    Interprets assembly, counting the iterations of each loop by the name of
    its variable, and the loads and stores of each buffer by its name, across
    every scope, in counts.
    """

    counts: Counter = Counter()

    def scope(self, *args, **kwargs):
        ctx = super().scope(*args, **kwargs)
        ctx.__class__ = type(self)
        return ctx

    @staticmethod
    def name(node) -> str:
        match node:
            case asm.Variable(name, _) | asm.Slot(name, _):
                # Names are made unique with numbers, which are dropped.
                name = re.sub(r"#\d+", "", name)
                return re.sub(r"_\d+$", "", name).strip("#")
            case asm.GetAttr(obj, attr):
                return f"{CountingInterpreter.name(obj)}.{attr}"
        return type(node).__name__

    def _dispatch(self, prgm):
        counts = type(self).counts
        match prgm:
            case asm.Load(buf, _):
                counts[f"load {self.name(buf)}"] += 1
            case asm.Store(buf, _, _):
                counts[f"store {self.name(buf)}"] += 1
            case asm.ForLoop(asm.Variable() as var, start, end, body):
                start_e, end_e = self(start), self(end)
                ctx_2 = self.scope(loop_state=HaltState())
                var_e = start_e
                while var_e < end_e:
                    if ctx_2.should_halt():
                        break
                    counts[f"for {self.name(var)}"] += 1
                    ctx_3 = self.scope()
                    ctx_3(asm.Block((asm.Assign(var, asm.Literal(var_e)), body)))
                    var_e = type(var_e)(var_e + 1)
                return None
            case asm.WhileLoop(cond, body):
                ctx_2 = self.scope(loop_state=HaltState())
                while self(cond):
                    ctx_3 = ctx_2.scope()
                    if ctx_3.should_halt():
                        break
                    counts["while"] += 1
                    ctx_3(body)
                return None
        return super()._dispatch(prgm)


class FixedLoopOrderer(AbstractLoopOrderer):
    """Fuses every query under one loop order, and then transposes the inputs
    it reads out of order, as the loop orderers do."""

    def __init__(self, ctx, order):
        super().__init__(ctx)
        self.order = order

    def set_loop_orders(self, prgm, stats, stats_factory):
        return Plan(
            tuple(
                fuse_loop_order(body, self.order)
                if isinstance(body, Query | QueryInto)
                else body
                for body in prgm.bodies
            )
        )


def stored(fmt, data):
    return run_copy(asm.AssemblyInterpreter(), *tensors(fmt, data))


def run(plan, inputs, outputs, stage=DefaultLogicFormatter):
    """
    Lower plan through stage, a loader factory taking the final loader, run it
    on inputs, and return the counts of its operations and the program as
    lowered. outputs maps each output alias to its shape.
    """
    capture = LogicCapture()
    stage(CompilerFormLowerer(capture))(
        plan, {var: tns.ftype for var, tns in inputs.items()}, {}, None
    )
    program = NotationGenerator()(capture.last_prgm, capture.last_bindings, {}, None)
    bindings = dict(inputs)
    for var, shape in outputs.items():
        bindings[var] = capture.last_bindings[var].construct(shape)
    CountingInterpreter.counts = Counter()
    lib = NotationCompiler(CountingInterpreter())(program)
    lib.main(*with_workspaces(program, *bindings.values()))
    return dict(CountingInterpreter.counts), capture.last_prgm


def loops(counts) -> dict[str, int]:
    return {
        k.removeprefix("for "): v for k, v in counts.items() if k.startswith("for ")
    }


def loads(counts, alias, buffer) -> int:
    """The loads of the buffers named buffer of the tensor of alias."""
    pattern = rf"^load _{alias}_.*_{buffer}(_|$)"
    return sum(v for k, v in counts.items() if re.search(pattern, k))


def test_inputs_are_transposed_rather_than_read_out_of_order():
    # C[k, i] = X[i, k] under loops (k, i) reads X out of order.
    i, k = Field("i"), Field("k")
    X, C = HardAlias("X"), HardAlias("C")
    rng = np.random.default_rng(0)
    data = (rng.random((16, 16)) < 0.2).astype(np.int64)
    x = stored("DS", data)
    sf = DCStatsFactory()
    stats: dict[Alias, TensorStats] = {X: sf(fl.asarray(data), (i, k))}
    query = Query(
        Table(C, (k, i)),
        Aggregate(Literal(ffuncs.add), Literal(0), Table(X, (i, k)), ()),
    )
    plan = Plan((query, Produces((C,))))
    nnz = int(data.sum())

    # The loop orderer copies X into loop order, and the query reads the copy
    # in order. The loop model prices that copy at the out-of-order prefix, as
    # it prices any query.
    counts, lowered = run(
        plan,
        {X: x},
        {C: (16, 16)},
        lambda ctx: FixedLoopOrderer(DefaultLogicFormatter(ctx), (k, i)),
    )
    reads = []
    for node in PostOrderDFS(lowered):
        match node:
            case Query(Table(tns, _), Aggregate(_, _, arg, _)) if tns == C:
                reads.append(arg)
    assert len(reads) == 1
    assert isinstance(reads[0], Table) and reads[0].tns != X
    assert reads[0].idxs == (k, i)

    def features(basis, order):
        plan = Plan((fuse_loop_order(query, order), Produces((C,))))
        return basis.get_features(plan, {}, stats, sf).sum(axis=0)

    copy = transpose_copy(X, stats[X], (k,))
    copy_rows = PigeonBasis().get_features(copy, {}, stats, sf).sum(axis=0)
    # The copy iterates over every entry of X.
    assert copy_rows[0] >= nnz
    assert np.array_equal(
        features(WithCopies(PigeonBasis()), (k, i)),
        features(PigeonBasis(), (k, i)) + copy_rows,
    )
    assert np.array_equal(
        features(WithCopies(PigeonBasis()), (i, k)), features(PigeonBasis(), (i, k))
    )

    # Fusing the query directly instead selects the out-of-order index with a
    # mask in an extra loop, which the loop basis doesn't see, where a copy in
    # order loops over its fields alone.
    direct, _ = run(
        Plan((fuse_loop_order(query, (k, i)), Produces((C,)))), {X: x}, {C: (16, 16)}
    )
    copy = Query(
        Table(C, (i, k)),
        Aggregate(Literal(ffuncs.add), Literal(0), Table(X, (i, k)), ()),
    )
    concordant, _ = run(
        Plan((fuse_loop_order(copy, (i, k)), Produces((C,)))), {X: x}, {C: (16, 16)}
    )
    assert set(loops(concordant)) == {"i", "k"}
    assert set(loops(direct)) == {"i", "k", "s"}
    assert sum(loops(direct).values()) > sum(loops(concordant).values())


def test_dense_levels_are_looked_up_at_every_visit():
    # c = sum_i x[i] * y[i], with 4 entries of 1000 in x, and y full.
    n = 1000
    i = Field("i")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    xd = np.zeros(n, np.int64)
    xd[[10, 200, 300, 999]] = 1
    yd = np.ones(n, np.int64)
    sf = ExactStatsFactory()
    stats = {x: sf(fl.asarray(xd), (i,)), y: sf(fl.asarray(yd), (i,))}
    basis = ProtocolBasis()
    names = basis.feature_names

    def plan(protocol=None):
        protocols = () if protocol is None else (protocol,)
        product = MapJoin(
            Literal(ffuncs.mul), (Table(x, (i,), protocols), Table(y, (i,)))
        )
        query = Query(
            Table(c, ()), Aggregate(Literal(ffuncs.add), Literal(0), product, (i,))
        )
        return Plan((fuse_loop_order(query, (i,)), Produces((c,))))

    def reads(option, protocol=None):
        """The visits of x, stored as a level of option, under its protocol."""
        formats: dict[Alias, list[LevelOption]] = {
            x: [option],
            y: [SparseListLevelOption()],
        }
        machine = StatsMachine(sf, dict(stats))
        for component in basis.components(plan(protocol), formats, machine):
            match component:
                case ProtocolChoice(table=Table(tns)) if tns == x:
                    row = basis.get_component_features(component, machine)[0]
                    name = type(component.protocol).__name__.lower()
        return row[names.index(f"{option.kind}.{name}.visits")]

    # A list of x is stepped through at its 4 entries, while a dense x is
    # looked up at every entry of y.
    assert reads(SparseListLevelOption()) == 4
    assert reads(DenseLevelOption(), Follow()) == n
    counts, _ = run(plan(), {x: stored("S", xd), y: stored("S", yd)}, {c: ()})
    assert loads(counts, "x", "val") == 4
    counts, _ = run(plan(), {x: stored("D", xd), y: stored("S", yd)}, {c: ()})
    assert loads(counts, "x", "val") == n


def test_gallops_jump_within_the_smallest_support():
    # c = sum_i x[i] * y[i], with 5 entries of 1000 in x and 100 in y.
    n = 1000
    i = Field("i")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    xd = np.zeros(n, np.int64)
    xd[[3, 400, 500, 600, 998]] = 1
    yd = np.zeros(n, np.int64)
    yd[np.arange(0, n, 10)] = 1
    sf = ExactStatsFactory()
    stats = {x: sf(fl.asarray(xd), (i,)), y: sf(fl.asarray(yd), (i,))}
    inputs = {x: stored("S", xd), y: stored("S", yd)}
    formats: dict[Alias, TensorFType] = {var: tns.ftype for var, tns in inputs.items()}

    def program(px, py):
        product = MapJoin(
            Literal(ffuncs.mul), (Table(x, (i,), (px,)), Table(y, (i,), (py,)))
        )
        query = Query(
            Table(c, ()), Aggregate(Literal(ffuncs.add), Literal(0), product, (i,))
        )
        return Plan((fuse_loop_order(query, (i,)), Produces((c,))))

    def jumps(px, py):
        machine = StatsMachine(sf, dict(stats))
        basis = ProtocolBasis()
        # The reads of x and y at the loop over i are the components.
        rows = [
            basis.get_component_features(component, machine)[0]
            for component in basis.components(program(px, py), formats, machine)
        ]
        return sum(rows)[basis.feature_names.index("sparse_list.gallop.visits")]

    # Two gallops each jump at most as often as the smaller list has entries.
    # A walk doesn't end the phases of a gallop beside it, so the gallop jumps
    # through every entry of its own.
    assert jumps(Gallop(), Gallop()) == 10
    assert jumps(Walk(), Gallop()) == 100
    assert jumps(Gallop(), Walk()) == 5
    assert jumps(Walk(), Follow()) == 0
    walks, _ = run(program(Walk(), Walk()), inputs, {c: ()})
    gallops, _ = run(program(Gallop(), Gallop()), inputs, {c: ()})
    walk_gallop, _ = run(program(Walk(), Gallop()), inputs, {c: ()})
    assert walk_gallop["while"] >= 100
    assert gallops["while"] < walks["while"]
    assert loads(gallops, "y", "idx") < loads(walks, "y", "idx")
