import argparse
from time import perf_counter

import numpy as np

import finch


def make_inputs(size):
    rng = np.random.default_rng(0)
    a = finch.asarray(rng.random((size, size)))
    b = finch.asarray(rng.random((size, size)))
    return a, b


def dense_hadamard(a, b):
    return finch.compute(finch.multiply(finch.lazy(a), finch.lazy(b)))


def scipy_hadamard(a, b):
    return a * b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--backend", choices=("mlir", "numba", "scipy"), required=True)
    args = parser.parse_args()

    inputs = make_inputs(args.size)

    if args.backend == "scipy":
        operation = scipy_hadamard
        a, b = inputs
        inputs = (a.to_numpy(), b.to_numpy())
    else:
        scheduler = {"mlir": finch.COMPILE_MLIR, "numba": finch.COMPILE_NUMBA}[
            args.backend
        ]
        finch.set_default_scheduler(ctx=scheduler)
        operation = dense_hadamard

    operation(*inputs)

    start = perf_counter()
    for _ in range(args.iterations):
        operation(*inputs)
    elapsed = perf_counter() - start

    print(f"backend:   {args.backend}")
    print(f"size:      {args.size}")
    print(f"iterations:{args.iterations}")
    print(f"total:     {elapsed:.6f} seconds")
    print(f"average:   {elapsed / args.iterations:.6f} seconds")


if __name__ == "__main__":
    main()
