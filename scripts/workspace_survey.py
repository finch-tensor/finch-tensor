"""
Compare WorkspaceLoopOrderer with BFSLoopOrderer, the loop orderer it extends,
on random einsums. Each problem contracts --tensors random sparse tensors,
which Galley factorizes, either loop orderer orders, WorkspaceCostFormatter
formats, and SmartProtocolizer reads with the cheapest protocols, before Numba
compiles it. Every input is CSF, with the same number of non-zeros, and their
coordinates are correlated: each index draws its values from one Zipf law,
shared by every tensor which uses it, so the same hot coordinates recur across
the inputs. The inputs are built from their coordinates, so their dimensions
can be far larger than would fit densely.

For each problem, WorkspaceLoopOrderer gives every query which scatters into
its output a workspace, so none should be left without one. The compiled kernels of
each loop orderer are timed once they are cached, without the time spent on
statistics, optimization, or compilation. Problems run in --jobs processes at
once, each of which is killed after --timeout seconds, and each problem draws
its inputs from the seed and its number, so the problems don't depend on how
many run at once.

    pixi run -e test python -m scripts.workspace_survey \
        [--problems N] [--tensors N] [--nnz N] [--skew A] [--min-dim N] \
        [--max-dim N] [--seed S] [--jobs N] [--timeout SECONDS]
"""

import argparse
import multiprocessing
import queue
import time
from collections import Counter

import numpy as np

import finch as ft
from finch.algebra import ftype
from finch.autoschedule import (
    CompilerFormLowerer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.formatter.smart_formatter import WorkspaceCostFormatter
from finch.autoschedule.loop_orderer import BFSLoopOrderer, WorkspaceLoopOrderer
from finch.autoschedule.loop_orderer.loop_order_workspace import (
    workspace_depth,
)
from finch.autoschedule.protocolizer import SmartProtocolizer
from finch.autoschedule.tensor_stats import DCStatsFactory
from finch.codegen import NumbaCompiler, NumpyBuffer
from finch.compile import NotationCompiler
from finch.finch_assembly import AssemblySimplify, LowerPackedStructSlots
from finch.finch_logic import (
    FusedAlias,
    HardAlias,
    LogicSimplify,
    Query,
    Table,
)
from finch.symbolic import PostOrderDFS
from finch.tensor import (
    DenseLevel,
    ElementLevel,
    ElementLevelFType,
    FiberTensor,
    SparseListLevel,
    element,
)

FIELDS = "ijklm"


def numba_compiler():
    return NotationCompiler(
        NumbaCompiler(), ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify())
    )


class Recorder(LogicCapture):
    """Record every program the formatter passes on, with its formats."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.programs: list[tuple] = []

    def lower(self, prgm, bindings, stats, stats_factory):
        self.programs.append((prgm, dict(bindings)))
        return super().lower(prgm, bindings, stats, stats_factory)


class _TimedModule:
    def __init__(self, mod, executor):
        self.mod = mod
        self.executor = executor

    def main(self, *args):
        start = time.perf_counter()
        result = self.mod.main(*args)
        self.executor.kernel_time += time.perf_counter() - start
        return result


class TimedExecutor(LogicExecutor):
    """
    Cache the kernels of each program, and add the time each call of one takes
    to kernel_time.
    """

    def __init__(self, ctx, stats_factory):
        super().__init__(ctx, stats_factory, cache=True)
        self.kernel_time = 0.0

    def _load_cached(self, stmt, binding_ftypes, bindings):
        mod, ftypes = super()._load_cached(stmt, binding_ftypes, bindings)
        return _TimedModule(mod, self), ftypes


def scheduler(orderer_cls):
    recorder = Recorder(CompilerFormLowerer(LogicCompiler(numba_compiler())))
    executor = TimedExecutor(
        GalleyLogicFactorizer(
            LogicSimplify(
                orderer_cls(WorkspaceCostFormatter(SmartProtocolizer(recorder)))
            )
        ),
        DCStatsFactory(),
    )
    return LogicNormalizer(executor), recorder, executor


def csf(coords: np.ndarray, vals: np.ndarray, shape: tuple[int, ...]) -> FiberTensor:
    """
    A CSF tensor of the non-zeros vals at the sorted, distinct rows of coords:
    a dense root over sorted lists, or a sorted list for a vector.
    """
    fmt = element(np.float64(0), ftype(np.float64), ftype(np.intp))
    lvl = ElementLevel(fmt, NumpyBuffer(vals))
    dense_root = len(shape) >= 2
    for depth in reversed(range(int(dense_root), len(shape))):
        # The entries of a level are the distinct prefixes of the coordinates,
        # grouped by the prefix of their parent.
        children = np.unique(coords[:, : depth + 1], axis=0)
        if depth == 0:
            n_parents, parents = 1, np.zeros(len(children), dtype=np.intp)
        elif dense_root and depth == 1:
            n_parents, parents = shape[0], children[:, 0]
        else:
            uniq, parents = np.unique(children[:, :depth], axis=0, return_inverse=True)
            n_parents = len(uniq)
        ptr = np.zeros(n_parents + 1, dtype=np.intp)
        np.add.at(ptr, parents + 1, 1)
        lvl = SparseListLevel(
            lvl,
            np.intp(shape[depth]),
            NumpyBuffer(np.cumsum(ptr)),
            NumpyBuffer(children[:, depth].astype(np.intp)),
        )
    if dense_root:
        lvl = DenseLevel(lvl, np.intp(shape[0]))
    return FiberTensor(lvl)


def correlated_tensor(rng, fields, dims, weights, nnz) -> FiberTensor:
    """
    A CSF tensor of nnz distinct non-zeros (or every entry, if fewer), whose
    coordinates draw each field from its shared weights.
    """
    shape = tuple(dims[f] for f in fields)
    nnz = min(nnz, int(np.prod(shape)))
    coords: set[tuple[int, ...]] = set()
    for _ in range(50):
        if len(coords) >= nnz:
            break
        batch = 2 * (nnz - len(coords))
        drawn = zip(
            *(rng.choice(dims[f], size=batch, p=weights[f]) for f in fields),
            strict=True,
        )
        for coord in drawn:
            coords.add(coord)
            if len(coords) == nnz:
                break
    # Very skewed laws can't fill the space; spread the rest uniformly.
    while len(coords) < nnz:
        coords.add(tuple(int(rng.integers(d)) for d in shape))
    return csf(np.array(sorted(coords)), rng.uniform(0.5, 1.5, size=nnz), shape)


def random_problem(rng, args):
    """
    An einsum of --tensors tensors, each of 1 to 3 fields, into 1 to 3 fields,
    and its inputs.
    """
    while True:
        inputs = [
            tuple(rng.choice(list(FIELDS), size=rng.integers(1, 4), replace=False))
            for _ in range(args.tensors)
        ]
        used = sorted(set().union(*inputs), key=FIELDS.index)
        if len(used) >= 2:
            break
    # The eager interface can't index a scalar FiberTensor, so every output
    # keeps a field.
    n_out = int(rng.integers(1, min(3, len(used)) + 1))
    output = tuple(rng.permutation(used)[:n_out])
    dims = {f: int(rng.integers(args.min_dim, args.max_dim + 1)) for f in used}
    weights = {}
    for f in used:
        w = 1.0 / np.arange(1, dims[f] + 1) ** args.skew
        weights[f] = (w / w.sum())[rng.permutation(dims[f])]
    tensors = [
        correlated_tensor(rng, fields, dims, weights, args.nnz) for fields in inputs
    ]
    subscripts = ",".join("".join(t) for t in inputs) + "->" + "".join(output)
    return subscripts, tensors, dims


def workspaces(programs) -> tuple[int, int, Counter]:
    """
    Count the queries which were given workspaces, the queries which still
    scatter into their outputs, and the formats of the workspaces.
    """
    used, declined, formats = 0, 0, Counter()
    for prgm, bindings in programs:
        copied = set()
        for node in PostOrderDFS(prgm):
            match node:
                case Query(Table(HardAlias() as out, _), Table(FusedAlias(ws, _), _)):
                    used += 1
                    copied.add(out)
                    formats[short(bindings[ws])] += 1
        for node in PostOrderDFS(prgm):
            match node:
                case Query(Table(HardAlias() as out, _), _) if (
                    out not in copied and workspace_depth(prgm, out) is not None
                ):
                    declined += 1
    return used, declined, formats


def short(fmt) -> str:
    """The levels of fmt above its elements, as Dense(SparseList)."""
    names = []
    lvl = fmt.lvl_t
    while True:
        match lvl:
            case ElementLevelFType():
                return "(".join(names) + ")" * (len(names) - 1)
        names.append(type(lvl).__name__.removesuffix("LevelFType"))
        lvl = lvl.lvl_t


def kernel_time(orderer_cls, subscripts, inputs, repeat=3):
    """
    The fastest of repeat runs of the cached kernels of the einsum under
    orderer_cls, and the programs they were compiled from.
    """
    sched, recorder, executor = scheduler(orderer_cls)
    ft.set_default_scheduler(ctx=sched)
    ft.einsum(subscripts, *inputs)
    times = []
    for _ in range(repeat):
        executor.kernel_time = 0.0
        ft.einsum(subscripts, *inputs)
        times.append(executor.kernel_time)
    return min(times), recorder.programs


def run_problem(n: int, args) -> tuple[str, Counter]:
    """Run problem n, whose inputs depend only on the seed and n."""
    rng = np.random.default_rng([args.seed, n])
    subscripts, inputs, dims = random_problem(rng, args)
    with_ws, programs = kernel_time(WorkspaceLoopOrderer, subscripts, inputs)
    without_ws, _ = kernel_time(BFSLoopOrderer, subscripts, inputs)
    used, declined, formats = workspaces(programs)
    counts = Counter(
        problems=1, with_workspace=used > 0, workspaces=used, declined=declined
    )
    sizes = " ".join(f"{f}={d}" for f, d in dims.items())
    line = (
        f"{n:3d} {subscripts:20s} {sizes:34s} workspaces={used} declined={declined}"
        f" with={with_ws * 1e3:9.2f}ms without={without_ws * 1e3:9.2f}ms"
        f" speedup={without_ws / with_ws:5.2f}x"
    )
    if formats:
        line += " " + ", ".join(f"{k}x{v}" for k, v in formats.items())
    return line, counts


def worker(n: int, args, results) -> None:
    try:
        results.put((n, *run_problem(n, args)))
    except Exception as e:  # noqa: BLE001
        results.put((n, f"{n:3d} failed: {type(e).__name__}: {e}", Counter()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problems", type=int, default=6)
    parser.add_argument("--tensors", type=int, default=3)
    parser.add_argument("--nnz", type=int, default=10000)
    parser.add_argument("--skew", type=float, default=1.0)
    parser.add_argument("--min-dim", type=int, default=1000)
    parser.add_argument("--max-dim", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    # Each problem runs in a process of its own, so that one which runs too
    # long can be killed, even inside a compiled kernel.
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    pending = list(range(args.problems))
    running: dict[int, tuple] = {}
    totals: Counter = Counter()
    while pending or running:
        while pending and len(running) < args.jobs:
            n = pending.pop(0)
            proc = context.Process(target=worker, args=(n, args, results))
            proc.start()
            running[n] = (proc, time.monotonic())
        try:
            n, line, counts = results.get(timeout=1)
            running.pop(n)[0].join()
            totals.update(counts)
            print(line, flush=True)
        except queue.Empty:
            pass
        for n, (proc, start) in list(running.items()):
            if time.monotonic() - start > args.timeout:
                proc.kill()
                proc.join()
                del running[n]
                totals.update(timeouts=1)
                print(f"{n:3d} timed out after {args.timeout:.0f}s", flush=True)
            elif not proc.is_alive() and proc.exitcode:
                del running[n]
                print(f"{n:3d} crashed with exit code {proc.exitcode}", flush=True)

    print(
        f"\n{totals['with_workspace']} of {totals['problems']} finished problems "
        f"used {totals['workspaces']} workspaces, {totals['declined']} scattering "
        f"queries had none, and {totals['timeouts']} problems timed out."
    )


if __name__ == "__main__":
    main()
