import argparse
from time import perf_counter

import numpy as np
import scipy.sparse as sps

import finch
from finch.tensor import FiberTensor


def make_inputs(size, density):

    upper = sps.triu(
        sps.random(
            size,
            size,
            density=density,
            format="csr",
            random_state=np.random.default_rng(0),
        ),
        k=1,
        format="csr",
    )
    adj = upper + upper.T
    adj.data[:] = 1.0

    a_csr = sps.csr_array(
        (
            adj.data,
            adj.indices.astype(np.intp),
            adj.indptr.astype(np.intp),
        ),
        shape=adj.shape,
    )

    return FiberTensor.from_scipy_csr(a_csr)


def triangle(a):
    return finch.compute(
        finch.sum(
            finch.multiply(
                finch.matmul(
                    finch.lazy(a),
                    finch.lazy(a),
                ),
                finch.lazy(a),
            )
        )
        / finch.asarray(6)
    )


def scipy_triangle(a):
    return (a @ a * a).sum() / 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--backend", choices=("mlir", "numba", "scipy"), required=True)
    args = parser.parse_args()

    inputs = make_inputs(args.size, args.density)

    if args.backend == "scipy":
        operation = scipy_triangle
        a = inputs
        inputs = a.to_scipy()
    else:
        scheduler = {
            "mlir": finch.COMPILE_MLIR,
            "numba": finch.COMPILE_NUMBA,
        }[args.backend]
        finch.set_default_scheduler(ctx=scheduler)
        operation = triangle

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
