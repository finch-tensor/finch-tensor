import functools

from finch.autoschedule.default_schedulers import get_default_scheduler

from .calls import positional_args, wrap_calls
from .dataflow import insert_lazy_and_compute
from .parser import fused_function_to_python_function, parse_fused_function


def jit(f, /, ctx=None):
    """
    A decorator that marks a function for just-in-time compilation. This allows the
    function to be compiled and optimized for performance when called.

    Parameters:
    - f: The function to be marked for JIT compilation. This function can use
        basic python control flow and operations (e.g. while, for, if). However,
        it shouldn't use more complex features like generators or classes.
        Calls to other Python functions are traced into when their source can be
        parsed, so tensors stay lazy and are fused across the call boundary.
        Other calls receive computed tensors.
    - ctx: The scheduler to use for computation. Defaults to the result of
        `get_default_scheduler()`.

    Returns:
    - A transformed function that inserts defer and compute statements to do tracing
       and optimization.


    Example usage:
    @jit
    def my_function(A, B, C):
        D = A @ B
        while some_condition(D):
            D = D + C
        return D

    In this example, `my_function` will be transformed to include defer and compute
    statements, allowing it to be optimized and executed efficiently when called.
    def opt_my_function(A, B, C):
        A, B = defer(A), defer(B)
        D = A @ B
        D = compute(D)
        while some_condition(D):
            C = defer(C)
            D = D + C
            D = compute(D)
        D = compute(D)
        return D
    """
    if ctx is None:
        ctx = get_default_scheduler()
    fused_fn = wrap_calls(parse_fused_function(f))
    transformed_fn = insert_lazy_and_compute(fused_fn)
    opt_fn = fused_function_to_python_function(transformed_fn)
    if f.__code__.co_kwonlyargcount or f.__defaults__:
        compiled_fn = opt_fn

        @functools.wraps(f)
        def opt_fn(*args, **kwargs):
            return compiled_fn(*positional_args(f, args, kwargs))

    # Lets jit-compiled callers trace into this function instead of calling it.
    opt_fn.__finch_jit_wrapped__ = f  # ty: ignore[unresolved-attribute]
    return opt_fn
