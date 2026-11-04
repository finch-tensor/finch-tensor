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
    ElementLevelOption,
    LevelBasis,
    LevelChoice,
    ProtocolBasis,
    SparseListLevelOption,
    TransposeBasis,
    level_executions,
)
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
    # in order, which the transpose basis charges for the out-of-order prefix.
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
    transposed = TransposeBasis().get_features(
        Plan((fuse_loop_order(query, (k, i)), Produces((C,)))), {}, stats, sf
    )
    assert transposed.sum() == nnz
    in_order = TransposeBasis().get_features(
        Plan((fuse_loop_order(query, (i, k)), Produces((C,)))), {}, stats, sf
    )
    assert in_order.sum() == 0

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
    product = MapJoin(Literal(ffuncs.mul), (Table(x, (i,)), Table(y, (i,))))
    query = Query(
        Table(c, ()), Aggregate(Literal(ffuncs.add), Literal(0), product, (i,))
    )
    plan = Plan((fuse_loop_order(query, (i,)), Produces((c,))))
    machine = StatsMachine(sf, dict(stats))
    choice = LevelChoice(plan, x, 0, (1.0,), SparseListLevelOption(), 8)
    row = LevelBasis(type(choice.option)).get_component_features(choice, machine)
    features = dict(
        zip(LevelBasis(type(choice.option)).feature_names, row.sum(axis=0), strict=True)
    )
    assert features["visits"] == 4
    assert features["dense_visits"] == n

    # A list of x is stepped through at its 4 entries, while a dense x is
    # looked up at every entry of y, which the dense term counts.
    counts, _ = run(plan, {x: stored("S", xd), y: stored("S", yd)}, {c: ()})
    assert loads(counts, "x", "val") == 4
    counts, _ = run(plan, {x: stored("D", xd), y: stored("S", yd)}, {c: ()})
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
        _, prefix = basis.components(program(px, py), formats, machine)
        row = basis.get_component_features(prefix, machine)[0]
        return row[basis.feature_names.index("sparse_list.gallop_visits")]

    # Two gallops each jump at most as often as the smaller list has entries,
    # and a gallop driven by a walk jumps once for each step of the walk.
    assert jumps(Gallop(), Gallop()) == 10
    assert jumps(Walk(), Gallop()) == 5
    assert jumps(Gallop(), Walk()) == 5
    assert jumps(Walk(), Follow()) == 0
    walks, _ = run(program(Walk(), Walk()), inputs, {c: ()})
    gallops, _ = run(program(Gallop(), Gallop()), inputs, {c: ()})
    assert gallops["while"] < walks["while"]
    assert loads(gallops, "y", "idx") < loads(walks, "y", "idx")


def test_levels_are_entered_as_often_as_the_level_above_is_visited():
    # c[i] = sum_j x[i, j] * y[i, j], with 2 entries in each of the first 4
    # of 16 rows of x, and y full.
    n = 16
    i, j = Field("i"), Field("j")
    x, y, c = HardAlias("x"), HardAlias("y"), HardAlias("c")
    xd = np.zeros((n, n), np.int64)
    xd[:4, :2] = 1
    yd = np.ones((n, n), np.int64)
    sf = ExactStatsFactory()
    stats = {x: sf(fl.asarray(xd), (i, j)), y: sf(fl.asarray(yd), (i, j))}
    product = MapJoin(Literal(ffuncs.mul), (Table(x, (i, j)), Table(y, (i, j))))
    query = Query(
        Table(c, (i,)), Aggregate(Literal(ffuncs.add), Literal(0), product, (j,))
    )
    plan = Plan((fuse_loop_order(query, (i, j)), Produces((c,))))
    machine = StatsMachine(sf, dict(stats))
    dense, lists = DenseLevelOption(), SparseListLevelOption()

    def choice(depth, option, executions=None):
        return LevelChoice(plan, x, depth, (1.0,), option, 8, executions)

    def accesses(depth, option, executions=None):
        basis = LevelBasis(type(option))
        rows = basis.get_component_features(choice(depth, option, executions), machine)
        # The allocation row comes first.
        return dict(zip(basis.feature_names, rows[1:].sum(axis=0), strict=True))

    # A list at the first level is visited at its 4 rows, where a dense level
    # is visited at the 16 rows of y, and each passes those fibers on.
    assert accesses(0, lists)["visits"] == 4
    assert accesses(0, lists)["dense_visits"] == n
    assert level_executions(choice(0, lists), machine) == (4.0,)
    assert level_executions(choice(0, dense), machine) == (16.0,)
    # The second level walks its 8 entries under either, but under the dense
    # level the loop over j runs in 12 more rows, where x has no entries and
    # y drives it over all 16 columns, which a dense second level is looked up
    # at, and its element level with it.
    under_list = accesses(1, lists, (4.0,))
    under_dense = accesses(1, lists, (16.0,))
    assert under_list["visits"] == under_dense["visits"] == 8
    assert under_list["entries"] == under_dense["entries"] == 8
    assert under_list["dense_visits"] == 4 * n
    assert under_dense["dense_visits"] == n * n
    assert level_executions(choice(1, lists, (16.0,)), machine) == (8.0,)
    assert level_executions(choice(1, dense, (16.0,)), machine) == (256.0,)
    assert accesses(2, ElementLevelOption(), (8.0,))["lookups"] == 8
    assert accesses(2, ElementLevelOption(), (256.0,))["lookups"] == 256
    # The interpreter looks x up as often as the model says.
    counts, _ = run(plan, {x: stored("SS", xd), y: stored("DD", yd)}, {c: (n,)})
    assert loads(counts, "x", "val") == 8
    counts, _ = run(plan, {x: stored("DD", xd), y: stored("DD", yd)}, {c: (n,)})
    assert loads(counts, "x", "val") == n * n
