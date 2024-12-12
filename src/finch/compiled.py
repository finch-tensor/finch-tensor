from abc import abstractmethod
from functools import wraps

from .julia import jl
from .tensor import Tensor
from .typing import JuliaObj


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


def compiled(opt: AbstractScheduler | None = None, tag: int | None = None):
    def inner(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, Tensor) and not jl.isa(
                    arg._obj, jl.Finch.LazyTensor
                ):
                    new_args.append(Tensor(jl.Finch.LazyTensor(arg._obj)))
                else:
                    new_args.append(arg)
            result = func(*new_args, **kwargs)
            kwargs = (
                {
                    "ctx": opt.get_julia_scheduler(),
                    "verbose": opt.verbose,
                }
                if opt is not None
                else {}
            )
            if tag is not None:
                kwargs["tag"] = tag
            result_tensor = Tensor(jl.Finch.compute(result._obj, **kwargs))
            return result_tensor

        return wrapper_func

    return inner


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
