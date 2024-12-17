from abc import abstractmethod
from functools import wraps

from .julia import jl
from .typing import JuliaObj
from typing import Any, Iterable, Iterator, Callable
from .tensor import Tensor


def _recurse(x: Iterable | Any, /, *, f: Callable[[Any], Any]) -> Iterable | Any:
    if isinstance(x, tuple | list):
        return type(x)(_recurse(xi, f=f) for xi in x)
    if isinstance(x, dict):
        return {k: _recurse(v, f=f) for k, v in x}
    return f(x)


def _recurse_iter(x: Iterable | Any, /) -> Iterator[Any]:
    if isinstance(x, tuple | list):
        for xi in x:
            yield from _recurse_iter(xi)
    if isinstance(x, dict):
        for xi in x.values():
            yield from _recurse_iter(xi)
    yield x


def _to_lazy_tensor(x: Tensor | Any, /) -> Tensor | Any:
    return x if not isinstance(x, Tensor) else lazy(x)


def compiled(opt=None, *, force_materialization=False, tag: int | None = None):
    def inner(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            args = tuple(args)
            kwargs = dict(kwargs)
            compute_at_end = force_materialization or all(
                t.is_computed()
                for t in _recurse_iter((args, kwargs))
                if isinstance(t, Tensor)
            )
            args = _recurse(args, f=_to_lazy_tensor)
            kwargs = _recurse(kwargs, f=_to_lazy_tensor)
            result = func(*args, **kwargs)
            if not compute_at_end:
                return result
            compute_kwargs = (
                {"ctx": opt.get_julia_scheduler()} if opt is not None else {}
            )
            if tag is not None:
                compute_kwargs["tag"] = tag
            return Tensor(jl.Finch.compute(result._obj, **compute_kwargs))

        return wrapper_func

    return inner


class AbstractScheduler:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def get_julia_scheduler(self) -> JuliaObj:
        pass


class GalleyScheduler(AbstractScheduler):
    def get_julia_scheduler(self) -> JuliaObj:
        return jl.Finch.galley_scheduler(verbose=self.verbose)


class DefaultScheduler(AbstractScheduler):
    def get_julia_scheduler(self) -> JuliaObj:
        return jl.Finch.default_scheduler(verbose=self.verbose)


def set_optimizer(opt: AbstractScheduler) -> None:
    jl.Finch.set_scheduler_b(opt.get_julia_scheduler())


def lazy(tensor: Tensor) -> Tensor:
    if tensor.is_computed():
        return Tensor(jl.Finch.LazyTensor(tensor._obj))
    return tensor


def compute(
    tensor: Tensor, *, opt: AbstractScheduler | None = None, tag: int = -1
) -> Tensor:
    if not tensor.is_computed():
        if opt is None:
            return Tensor(jl.Finch.compute(tensor._obj, tag=tag))
        else:
            return Tensor(
                jl.Finch.compute(
                    tensor._obj,
                    verbose=opt.verbose,
                    ctx=opt.get_julia_scheduler(),
                    tag=tag,
                )
            )
    return tensor
