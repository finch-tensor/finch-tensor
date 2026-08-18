def opt_fn(A, n):
    A, n = maybedefer((A, n))
    B = A
    A, B, n = compute((A, B, n))
    for _i in range(n):
        A, B, n = maybedefer((A, B, n))
        B = add(B, A)
        A, B, n = compute((A, B, n))
    B, = defer((B,))
    B, = compute((B,))
    return B
