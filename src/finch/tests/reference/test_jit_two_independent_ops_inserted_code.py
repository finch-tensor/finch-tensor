def opt_fn(A, B, C):
    A, B, C = maybedefer((A, B, C))
    D = matmul(A, B)
    E = add(A, C)
    F = add(D, E)
    F, = defer((F,))
    F, = compute((F,))
    return F
