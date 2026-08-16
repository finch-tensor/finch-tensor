import argparse
from time import perf_counter

import numpy as np
import scipy.sparse as sps

import finch
from finch.tensor import FiberTensor


def make_inputs(size, density):
    rng = np.random.default_rng(0)
    matrices = []

    for _ in range(3):
        csr = sps.random(
            size,
            size,
            density=density,
            format="csr",
            random_state=rng,
        )
        matrices.append(
            sps.csr_array(
                (
                    csr.data,
                    csr.indices.astype(np.intp),
                    csr.indptr.astype(np.intp),
                ),
                shape=csr.shape,
            )
        )

    a_csr, b_csr, c_csr = matrices
    a = FiberTensor.from_scipy_csr(a_csr)
    b = FiberTensor.from_scipy_csr(b_csr)
    c = FiberTensor.from_scipy_csr(c_csr)
    return a, b, c


def spgemm2(a, b, c):
    return finch.compute(
        finch.matmul(
            finch.matmul(finch.lazy(a), finch.lazy(b)),
            finch.lazy(c),
        )
    )


def scipy_spgemm2(a, b, c):
    return (a @ b) @ c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--backend", choices=("mlir", "numba", "scipy"), required=True)
    args = parser.parse_args()

    inputs = make_inputs(args.size, args.density)

    if args.backend == "scipy":
        operation = scipy_spgemm2
        a, b, c = inputs
        inputs = (a.to_scipy(), b.to_scipy(), c.to_scipy())
    else:
        scheduler = {"mlir": finch.COMPILE_MLIR, "numba": finch.COMPILE_NUMBA}[
            args.backend
        ]
        finch.set_default_scheduler(ctx=scheduler)
        operation = spgemm2

    operation(*inputs)

    start = perf_counter()

    for _ in range(args.iterations):
        operation(*inputs)

    elapsed = perf_counter() - start

    print(f"backend:   {args.backend}")
    print(f"size:      {args.size}")
    print(f"density:   {args.density}")
    print(f"iterations:{args.iterations}")
    print(f"total:     {elapsed:.6f} seconds")
    print(f"average:   {elapsed / args.iterations:.6f} seconds")


if __name__ == "__main__":
    main()
