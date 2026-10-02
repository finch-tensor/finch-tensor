def opt_fn(A, B):
    A, B = maybedefer((A, B))
    C = matmul(A, B)
    C, = maybedefer((C,))
    C, = compute((C,))
    return C
