from functools import wraps

from .julia import jl
from .tensor import Tensor


def compiled(opt=None, tag = -1):
    def inner(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, Tensor) and not jl.isa(arg._obj, jl.Finch.LazyTensor):
                    new_args.append(Tensor(jl.Finch.LazyTensor(arg._obj)))
                else:  
                    new_args.append(arg)
            result = func(*new_args, **kwargs)
            kwargs = {"ctx": opt.get_julia_scheduler(), "tag": tag} if opt is not None else {}
            result_tensor = Tensor(jl.Finch.compute(result._obj, **kwargs))
            return result_tensor
        return wrapper_func
    return inner

def lazy(tensor: Tensor):
    if tensor.is_computed():
        return Tensor(jl.Finch.LazyTensor(tensor._obj))
    return tensor

class AbstractScheduler():
    pass

class GalleyScheduler(AbstractScheduler):
    def __init__(self, verbose=False):
        self.verbose=verbose

    def get_julia_scheduler(self):
        return jl.Finch.galley_scheduler(verbose=self.verbose)
    
class DefaultScheduler(AbstractScheduler):
    def __init__(self, verbose=False):
        self.verbose=verbose

    def get_julia_scheduler(self):
        return jl.Finch.default_scheduler(verbose=self.verbose)

def set_optimizer(opt):
    jl.Finch.set_scheduler_b(opt.get_julia_scheduler())
    return

def compute(tensor: Tensor, *,  opt=None, tag=-1):
    if not tensor.is_computed():
        if opt == None:
            return Tensor(jl.Finch.compute(tensor._obj, verbose=opt.verbose))
        else:            
            return Tensor(jl.Finch.compute(tensor._obj, verbose=opt.verbose, ctx=opt.get_julia_scheduler()))
    return tensor
