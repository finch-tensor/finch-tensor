from functools import wraps

from .julia import jl
from .tensor import Tensor


def get_scheduler(name, verbose=False):
    if name == "default":
        return jl.Finch.default_scheduler()
    elif name == "galley":
        return jl.Finch.galley_scheduler(verbose=verbose)

def compiled(opt=""):
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
            kwargs = {"ctx": get_scheduler(name=opt)} if opt != "" else {}  
            result_tensor = Tensor(jl.Finch.compute(result._obj, **kwargs))  
            return result_tensor  
        return wrapper_func  

    return inner  

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

def compute(tensor: Tensor, *, verbose: bool = False, opt="", tag=-1):
    if not tensor.is_computed():
        if opt == "":
            return Tensor(jl.Finch.compute(tensor._obj, verbose=verbose, tag=tag))
        else:            
            return Tensor(jl.Finch.compute(tensor._obj, verbose=verbose, tag=tag, ctx=get_scheduler(opt, verbose=verbose)))
    return tensor
