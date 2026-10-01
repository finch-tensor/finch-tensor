def opt_fn(A, B, use_matmul):
    if concrete(use_matmul):
        A, B = maybedefer((A, B))
        result = matmul(A, B)
        result, = compute((result,))
    else:
        A, B = maybedefer((A, B))
        result = add(A, B)
        result, = compute((result,))
    result, = maybedefer((result,))
    result, = compute((result,))
    return result
