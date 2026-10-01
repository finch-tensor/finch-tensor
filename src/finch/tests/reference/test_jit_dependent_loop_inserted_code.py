def opt_fn(A, n):
    A, n = maybedefer((A, n))
    B = A
    A, B = compute((A, B))
    for _i in fused_call(range, fused_call(getattr(sum(B), 'item'))):
        A, B = maybedefer((A, B))
        B = add(B, A)
        A, B = compute((A, B))
    B, = maybedefer((B,))
    B, = compute((B,))
    return B
