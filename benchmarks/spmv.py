import argparse
from time import perf_counter

import numpy as np
import scipy.sparse as sps

import finch
from finch.tensor import FiberTensor


def make_inputs(size, density):
    rng = np.random.default_rng(0)

    matrix = sps.random(
        size,
        size,
        density=density,
        format="csr",
        random_state=rng,
    )
    vector = sps.random(
        size,
        1,
        density=density,
        format="csr",
        random_state=rng,
    )

    a_csr = sps.csr_array(
        (
            matrix.data,
            matrix.indices.astype(np.intp),
            matrix.indptr.astype(np.intp),
        ),
        shape=matrix.shape,
    )
    b_csr = sps.csr_array(
        (
            vector.data,
            vector.indices.astype(np.intp),
            vector.indptr.astype(np.intp),
        ),
        shape=vector.shape,
    )

    a = FiberTensor.from_scipy_csr(a_csr)
    b = FiberTensor.from_scipy_csr(b_csr)
    return a, b


def spmv(a, b):
    return finch.compute(finch.matmul(finch.lazy(a), finch.lazy(b)))


def scipy_spmv(a, b):
    return a @ b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--backend", choices=("mlir", "numba", "scipy"), required=True)
    args = parser.parse_args()

    inputs = make_inputs(args.size, args.density)

    if args.backend == "scipy":
        operation = scipy_spmv
        a, b = inputs
        inputs = (a.to_scipy(), b.to_scipy())
    else:
        scheduler = {
            "mlir": finch.COMPILE_MLIR,
            "numba": finch.COMPILE_NUMBA,
        }[args.backend]
        finch.set_default_scheduler(ctx=scheduler)
        operation = spmv

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
