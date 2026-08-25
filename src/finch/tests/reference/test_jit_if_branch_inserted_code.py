def opt_fn(A, B, use_matmul):
    if use_matmul:
        A, B = maybedefer((A, B))
        result = matmul(A, B)
        result, = compute((result,))
    else:
        A, B = maybedefer((A, B))
        result = add(A, B)
        result, = compute((result,))
    result, = defer((result,))
    result, = compute((result,))
    return result
