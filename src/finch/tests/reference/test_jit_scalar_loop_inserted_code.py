def opt_fn(A, n):
    A, n = maybedefer((A, n))
    B = A
    A, B, n = compute((A, B, n))
    for _i in fused_call(range, n):
        A, B, n = maybedefer((A, B, n))
        B = add(B, A)
        A, B, n = compute((A, B, n))
    B, = maybedefer((B,))
    B, = compute((B,))
    return B
