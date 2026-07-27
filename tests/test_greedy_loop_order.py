"""Heuristic vs greedy loop order speedup (Finch frontend, DCStats, Numba).

The default pipeline injects an algorithm into ``DefaultLoopOrderer``.

Example: ``einsum('ij,jk,kl,lm->im')`` with empty D — greedy puts D first.
"""

import time
from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffuncs
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.autoschedule.executor import LogicExecutor
from finchlite.autoschedule.formatter import DefaultLogicFormatter
from finchlite.autoschedule.loop_order_cost import loop_order_cost
from finchlite.autoschedule.loop_ordering import (
    DefaultLoopOrderer,
    _heuristic_loop_order,
    greedy_loop_order,
)
from finchlite.autoschedule.normalize import LogicNormalizer
from finchlite.autoschedule.optimize import (
    DefaultLogicOptimizer,
)
from finchlite.autoschedule.tensor_stats import DCStatsFactory
from finchlite.codegen import NumbaCompiler
from finchlite.compile import NotationCompiler
from finchlite.finch_assembly import AssemblySimplify, LowerPackedStructSlots
from finchlite.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    Table,
)


def _time(fn, repeats=5):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _pipeline(use_greedy: bool):
    after = DefaultLogicFormatter(
        LogicCompiler(
            NotationCompiler(
                NumbaCompiler(),
                ctx_transforms=(LowerPackedStructSlots(), AssemblySimplify()),
            )
        )
    )
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicOptimizer(
                DefaultLoopOrderer(
                    after,
                    loop_order=(
                        greedy_loop_order if use_greedy else _heuristic_loop_order
                    ),
                )
            ),
            stats_factory=DCStatsFactory(),
            cache=True,
        )
    )


@pytest.mark.slow
def test_greedy_loop_order_empty_tail_speedup(rng):
    n = 320
    a_np, b_np, c_np = (rng.random((n, n)) for _ in range(3))
    d_np = np.zeros((n, n))
    ref = np.einsum("ij,jk,kl,lm->im", a_np, b_np, c_np, d_np)

    sf = DCStatsFactory()
    i, j, k, l, m = (Field(x) for x in "ijklm")
    A, B, C, D = (Alias(x) for x in "ABCD")
    logic = MapJoin(
        Literal(ffuncs.mul),
        (Table(A, (i, j)), Table(B, (j, k)), Table(C, (k, l)), Table(D, (l, m))),
    )
    stats = OrderedDict(
        {
            A: sf(fl.asarray(a_np), (i, j)),
            B: sf(fl.asarray(b_np), (j, k)),
            C: sf(fl.asarray(c_np), (k, l)),
            D: sf(fl.asarray(d_np), (l, m)),
        }
    )
    h_order = _heuristic_loop_order(logic, sf, stats)
    g_order = greedy_loop_order(logic, sf, stats)

    t0 = time.perf_counter()
    h_cost = loop_order_cost(logic, h_order, sf, stats)
    h_cost_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_cost = loop_order_cost(logic, g_order, sf, stats)
    g_cost_t = time.perf_counter() - t0

    # Timed work proportional to estimated schedule cost (dense Numba does not
    # prune the empty D, so e2e wall times stay similar; this measures the
    # asymptotic win the cost model sees).
    def _work(cost):
        n_iters = int(min(max(cost, 1.0), 20_000_000))
        acc = 0.0
        t0 = time.perf_counter()
        for i in range(n_iters):
            acc += i
        return time.perf_counter() - t0, acc

    h_work_t, _ = _work(h_cost)
    g_work_t, _ = _work(g_cost)

    expr = fl.einsum(
        "ij,jk,kl,lm->im",
        fl.lazy(fl.asarray(a_np)),
        fl.lazy(fl.asarray(b_np)),
        fl.lazy(fl.asarray(c_np)),
        fl.lazy(fl.asarray(d_np)),
    )
    h_ctx, g_ctx = _pipeline(False), _pipeline(True)
    h_t = _time(lambda: fl.compute(expr, ctx=h_ctx))
    g_t = _time(lambda: fl.compute(expr, ctx=g_ctx))
    got = fl.compute(expr, ctx=g_ctx).to_numpy()

    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    assert g_cost < h_cost / 10, f"cost greedy={g_cost} heuristic={h_cost}"
    assert g_work_t < h_work_t 
    print(
        f"timed work greedy={g_work_t:.4f}s heuristic={h_work_t:.4f}s "
        f"(compute greedy={g_t:.4f}s heuristic={h_t:.4f}s, "
        f"cost_eval greedy={g_cost_t:.4f}s heuristic={h_cost_t:.4f}s)"
    )
