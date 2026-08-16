import argparse
from time import perf_counter

import numpy as np
import scipy.sparse as sps

import finch
from finch.tensor import FiberTensor


def make_inputs(size, density):
    rng = np.random.default_rng(0)
    matrices = []

    for _ in range(2):
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

    a_csr, b_csr = matrices
    a = FiberTensor.from_scipy_csr(a_csr)
    b = FiberTensor.from_scipy_csr(b_csr)

    return a, b


def hadamard(a, b):
    return finch.compute(finch.multiply(finch.lazy(a), finch.lazy(b)))


def scipy_hadamard(a, b):
    return a.multiply(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--backend", choices=("mlir", "numba", "scipy"), required=True)
    args = parser.parse_args()

    inputs = make_inputs(args.size, args.density)

    if args.backend == "scipy":
        operation = scipy_hadamard
        inputs = tuple(tensor.to_scipy() for tensor in inputs)
    else:
        scheduler = {"mlir": finch.COMPILE_MLIR, "numba": finch.COMPILE_NUMBA}[
            args.backend
        ]
        finch.set_default_scheduler(ctx=scheduler)
        operation = hadamard

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
