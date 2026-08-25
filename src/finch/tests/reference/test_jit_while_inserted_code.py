def opt_fn(A, B, n):
    A, B, n = maybedefer((A, B, n))
    C = A
    B, C, n = compute((B, C, n))
    while n > 0:
        B, C, n = maybedefer((B, C, n))
        C = add(C, B)
        n = n - 1
        B, C, n = compute((B, C, n))
    C, = defer((C,))
    C, = compute((C,))
    return C
