from abc import abstractmethod
from functools import wraps

from .julia import jl
from .typing import JuliaObj


def _recurse(x, /, *, f):
    if isinstance(x, tuple | list):
        return type(x)(_recurse(xi, f=f) for xi in x)
    if isinstance(x, dict):
        return {k: _recurse(v, f=f) for k, v in x}
    return f(x)


def _recurse_iter(x, /):
    if isinstance(x, tuple | list):
        yield from (_recurse_iter(xi) for xi in x)
    if isinstance(x, dict):
        yield from (_recurse_iter(xi) for xi in x.values())
    yield x


def _to_lazy_tensor(x, /):
    from .tensor import Tensor

    if isinstance(x, Tensor) and not jl.isa(x._obj, jl.Finch.LazyTensor):
        return Tensor(jl.Finch.LazyTensor(x._obj))

    return x


def compiled(opt=None, *, force_materialization=False, tag: int | None = None):
    def inner(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            from .tensor import Tensor

            args = tuple(args)
            kwargs = dict(kwargs)
            compute_at_end = True
            if not force_materialization and any(
                isinstance(arg, Tensor) and jl.isa(arg._obj, jl.Finch.LazyTensor)
                for arg in _recurse_iter((args, kwargs))
            ):
                compute_at_end = False

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


def lazy(tensor):
    from .tensor import Tensor

    if tensor.is_computed():
        return Tensor(jl.Finch.LazyTensor(tensor._obj))
    return tensor


def compute(tensor, *, opt: AbstractScheduler | None = None, tag: int = -1):
    from .tensor import Tensor

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
