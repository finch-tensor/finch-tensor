from functools import wraps

from .julia import jl
from .tensor import Tensor


def compiled(func, opt="galley"):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, Tensor) and not jl.isa(arg._obj, jl.Finch.LazyTensor):
                new_args.append(Tensor(jl.Finch.LazyTensor(arg._obj)))
            else:
                new_args.append(arg)

        result = func(*new_args, **kwargs)
        result_tensor = Tensor(jl.Finch.compute(result._obj, opt=opt))

        return result_tensor

    return wrapper_func


def lazy(tensor: Tensor):
    if tensor.is_computed():
        return Tensor(jl.Finch.LazyTensor(tensor._obj))
    return tensor

def set_optimizer(opt="default"):
    if opt == "default":
        jl.Finch.set_scheduler_b(jl.Finch.default_scheduler())
    elif opt == "galley":
        jl.Finch.set_scheduler_b(jl.Finch.galley_scheduler())
    return

def clear_optimizer_cache():
    jl.empty_b(jl.Finch.codes)
    return

def compute(tensor: Tensor, *, verbose: bool = False, opt=""):
    if not tensor.is_computed():
        if opt == "":
            return Tensor(jl.Finch.compute(tensor._obj, verbose=verbose))
        elif opt == "default":
            return Tensor(jl.Finch.compute(tensor._obj, verbose=verbose, ctx=jl.Finch.default_scheduler()))
        elif opt == "galley":
            return Tensor(jl.Finch.compute(tensor._obj, verbose=verbose, ctx=jl.Finch.galley_scheduler(verbose=verbose)))
    return tensor
