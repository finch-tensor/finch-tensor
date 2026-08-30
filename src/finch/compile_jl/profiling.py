from collections.abc import Iterator
from contextlib import contextmanager

from .julia import jl


@contextmanager
def profile_julia_calls(
    *, n: int = 10**7, delay: float = 0.0005, mincount: int = 5
) -> Iterator[None]:
    """
    Profiles Julia-side execution during the wrapped block, including calls
    made into Julia from Python (e.g. `FinchJLKernel.__call__`). Prints a
    flat, count-sorted profile on exit.

    Example:
        with profile_julia_calls():
            ft.compute(expr)
    """
    jl.seval("using Profile")
    jl.seval("Profile.clear()")
    jl.seval(f"Profile.init(n={n}, delay={delay})")
    jl.seval("ccall(:jl_profile_start_timer, Cint, ())")
    try:
        yield
    finally:
        jl.seval("ccall(:jl_profile_stop_timer, Cvoid, ())")
        jl.seval("Profile.print(format=:tree)")
