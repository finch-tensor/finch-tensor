"""
Time matmul on the Julia backend for dense and sparse inputs.

Separates the two costs that behave differently:

* **cold** -- the first ``compute`` of a kernel, which includes Julia codegen
  and compilation. Paid once per kernel shape, so each case is measured in a
  fresh subprocess; measuring it in-process after another case would report a
  warm cache instead.
* **warm** -- steady-state execution, reported as the minimum over repetitions.
  The minimum, not the mean: it is the sample least polluted by unrelated
  system activity.

Also records the loop order the scheduler picked, since a change in generated
code is the only way a scheduling fix can move these numbers -- an identical
loop order means any delta is noise.

Usage::

    pixi run -e test-julia python scripts/time_matmul_julia.py
    pixi run -e test-julia python scripts/time_matmul_julia.py --reps 7 --json out.json

    # one case in-process (used internally for honest cold timings)
    pixi run -e test-julia python scripts/time_matmul_julia.py --case sparse
"""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler
from finch.compile_jl.julia import julia_available

CASES = ("dense", "sparse", "sparse_dense", "transpose", "sparse_dense_pre_t")


def build(case: str, n: int, density: float, seed: int):
    """Return (a, b, reference) for one case, plus a short description."""
    rng = np.random.default_rng(seed)
    if case == "dense":
        a = rng.random((n, n))
        b = rng.random((n, n))
        return a, b, f"dense {n}x{n} @ dense {n}x{n}"
    if case == "sparse":
        a = sps.random(n, n, density=density, format="csr", random_state=rng)
        b = sps.random(n, n, density=density, format="csr", random_state=rng)
        return a, b, f"csr {n}x{n} (nnz={a.nnz}) @ csr (nnz={b.nnz})"
    if case == "sparse_dense":
        a = sps.random(n, n, density=density, format="csr", random_state=rng)
        b = rng.random((n, n))
        return a, b, f"csr {n}x{n} (nnz={a.nnz}) @ dense {n}x{n}"
    if case == "transpose":
        # The transpose alone, on the same sparse operand the `sparse_dense`
        # schedule is forced to swizzle. Isolates the rebuild cost from the
        # multiply that consumes it.
        a = sps.random(n, n, density=density, format="csr", random_state=rng)
        return a, None, f"transpose of csr {n}x{n} (nnz={a.nnz})"
    if case == "sparse_dense_pre_t":
        # Same math and same loop order as `sparse_dense`, but A's data is
        # already stored transposed as CSR. The read order then matches storage,
        # so no swizzle fires and A stays a sorted SparseList instead of being
        # rebuilt as SparseHash -- isolating loop order from format.
        a = sps.random(n, n, density=density, format="csr", random_state=rng)
        b = rng.random((n, n))
        return a, b, f"csr-transposed {n}x{n} (nnz={a.nnz}) @ dense {n}x{n}"
    raise ValueError(f"unknown case {case!r}")


def record_loop_orders():
    """
    Capture every loop order the heuristic picks.

    Wrapping the module-level name is deliberate: the schedule is an internal
    decision with no public accessor, and it is the one signal that explains
    whether a scheduling change could have moved the timings at all.
    """
    from finch.autoschedule import loop_ordering

    seen: list[tuple[str, ...]] = []
    original = loop_ordering._heuristic_loop_order

    def wrapper(root):
        order = original(root)
        seen.append(tuple(f.name for f in order))
        return order

    loop_ordering._heuristic_loop_order = wrapper
    return seen


def densify(tensor) -> np.ndarray:
    """
    Materialize a result as a dense array whatever its level structure.

    Neither conversion on `FiberTensor` covers every format: `to_numpy` reshapes
    the stored-value buffer, so it is only right for all-dense levels, and
    `to_scipy` handles SparseList/SparseCOO but not the SparseHash levels a
    sparse matmul actually produces. So route anything else through Finch.jl's
    own reformat into CSR first, the way `test_julia_backend._to_csr` does.
    """
    from finch.tensor.level import DenseLevelFType, ElementLevelFType

    lvl_t = tensor.ftype.lvl_t
    while isinstance(lvl_t, DenseLevelFType):
        lvl_t = lvl_t.lvl_t
    if isinstance(lvl_t, ElementLevelFType):
        return np.asarray(tensor.to_numpy())

    from finch.compile_jl.interop import jl_tensor_to_python, tensor_to_jl
    from finch.compile_jl.julia import jl

    csr_level = jl.Dense(jl.SparseList(jl.Element(tensor.fill_value)))
    csr = jl_tensor_to_python(jl.Tensor(csr_level, tensor_to_jl(tensor)))
    return csr.to_scipy().toarray()


def run_case(case: str, n: int, density: float, reps: int, seed: int) -> dict:
    a, b, desc = build(case, n, density, seed)
    orders = record_loop_orders()

    if case == "transpose":
        want = a.T.toarray() if sps.issparse(a) else np.asarray(a).T
    else:
        reference = a @ b
        want = (
            reference.toarray() if sps.issparse(reference) else np.asarray(reference)
        )

    # Everything stays inside the scheduler context, the conversion of the
    # result included: converting a tensor can itself run a kernel, and outside
    # the block that kernel would go to the default interpreter instead.
    with with_default_scheduler(COMPILE_JULIA):
        if case == "transpose":
            expr = ft.permute_dims(ft.defer(ft.asarray(a)), (1, 0))
        elif case == "sparse_dense_pre_t":
            # Store A transposed, then present it back as A logically.
            a_t = ft.asarray(sps.csr_matrix(a.T))
            expr = ft.matmul(
                ft.permute_dims(ft.defer(a_t), (1, 0)), ft.defer(ft.asarray(b))
            )
        else:
            a_f, b_f = ft.asarray(a), ft.asarray(b)
            expr = ft.matmul(ft.defer(a_f), ft.defer(b_f))

        start = time.perf_counter()
        result = ft.compute(expr)
        cold = time.perf_counter() - start

        warm = []
        for _ in range(reps):
            start = time.perf_counter()
            ft.compute(expr)
            warm.append(time.perf_counter() - start)

        # Correctness gate: a timing number from a wrong kernel is worthless.
        got = densify(result)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)

    return {
        "case": case,
        "desc": desc,
        "n": n,
        "density": density,
        "cold_s": cold,
        "warm_min_s": min(warm),
        "warm_med_s": float(np.median(warm)),
        "reps": reps,
        "loop_orders": sorted(set(orders)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, help="run one case in-process")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--sparse-n", type=int, default=2000)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write results as JSON to this path")
    parser.add_argument("--label", default="", help="tag for the run, e.g. a branch")
    args = parser.parse_args()

    if not julia_available():
        print("Julia backend unavailable (juliacall/juliapkg missing)", file=sys.stderr)
        return 1

    if args.case:
        n = args.sparse_n if args.case != "dense" else args.n
        print(json.dumps(run_case(args.case, n, args.density, args.reps, args.seed)))
        return 0

    # Each case gets a fresh process so its cold timing is not served by a
    # kernel cache another case already filled.
    results = []
    for case in CASES:
        proc = subprocess.run(
            [
                sys.executable, __file__, "--case", case,
                "--n", str(args.n), "--sparse-n", str(args.sparse_n),
                "--density", str(args.density), "--reps", str(args.reps),
                "--seed", str(args.seed),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"{case}: FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    label = f" [{args.label}]" if args.label else ""
    print(f"\nmatmul on the Julia backend{label}")
    print(f"{'case':14} {'cold (s)':>10} {'warm min (s)':>13} {'loop order'}")
    print("-" * 74)
    for r in results:
        orders = "; ".join(",".join(o) for o in r["loop_orders"]) or "-"
        print(
            f"{r['case']:14} {r['cold_s']:10.3f} {r['warm_min_s']:13.4f} {orders}"
        )
    for r in results:
        print(f"\n{r['case']}: {r['desc']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"label": args.label, "results": results}, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
