import argparse
from time import perf_counter

import numpy as np
import scipy.sparse as sps

import finch
from finch.tensor import FiberTensor


def make_inputs(size, density):
    rng = np.random.default_rng(0)

    a = finch.asarray(rng.random((size, size)))
    b = finch.asarray(rng.random((size, size)))

    mask_csr = sps.random(
        size,
        size,
        density=density,
        format="csr",
        random_state=rng,
    )

    mask_csr = sps.csr_array(
        (
            mask_csr.data,
            mask_csr.indices.astype(np.intp),
            mask_csr.indptr.astype(np.intp),
        ),
        shape=mask_csr.shape,
    )

    mask = FiberTensor.from_scipy_csr(mask_csr)
    return mask, a, b


def sddmm(mask, a, b):
    return finch.compute(
        finch.multiply(
            finch.lazy(mask),
            finch.matmul(finch.lazy(a), finch.lazy(b)),
        )
    )


def scipy_sddmm(mask, a, b):
    return mask.multiply(a @ b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--backend", choices=("mlir", "numba", "scipy"), required=True)
    args = parser.parse_args()

    inputs = make_inputs(args.size, args.density)

    if args.backend == "scipy":
        operation = scipy_sddmm
        mask, a, b = inputs
        inputs = (mask.to_scipy(), a.to_numpy(), b.to_numpy())
    else:
        scheduler = {"mlir": finch.COMPILE_MLIR, "numba": finch.COMPILE_NUMBA}[
            args.backend
        ]
        finch.set_default_scheduler(ctx=scheduler)
        operation = sddmm

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
