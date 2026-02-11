from finchlite.compile import NotationCompiler
from finchlite.finch_assembly import AssemblyKernel, AssemblyLibrary
import finchlite.finch_notation.nodes as ntn
from typing import Any

from juliacall import Main as jl


class FinchJLKernel(AssemblyKernel):
    def __init__(self, func_name, jl_code):
        self.jl_code = jl_code
        self.func_name = func_name
        jl.seval(jl_code)

    # TODO: Switch back to (self, *args: tuple[FinchJLTensor, ...]) -> tuple[FinchJLTensor, ...]
    def __call__(self, *args: tuple[Any, ...]):
        argList = []
        for arg in args:
            argList.append(f"arg{len(argList)}")
            setattr(jl, argList[-1], arg)

        jl.seval(f"{self.func_name}({argList.join(',')})")


class FinchJLLibrary(AssemblyLibrary):
    def __init__(self, kernel_dict):
        self.kernel_dict = kernel_dict

    def __getattr__(self, name: str) -> FinchJLKernel:
        return self.kernels[name]


class FinchJLGenerator:
    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        match prgm:
            case ntn.Literal(val):
                ...
            # What this?
            case ntn.Value(ex, type_):
                ...
            case ntn.Variable(name, type_):
                ...
            case ntn.Call(op, args):
                ...
            case ntn.Dimension(tns, r):
                ...
            case ntn.Access(tns, mode, idxs):
                ...
            case ntn.Read():
                ...
            case ntn.AccessMode():
                ...
            case ntn.Update(op):
                ...
            case ntn.Increment(lhs,rhs):
                ...
            case ntn.Unwrap(arg):
                ...
            case ntn.Cached(arg,ref):
                ...
            case ntn.Loop(idx, ext, body):
                ...
            case ntn.If(cond, body):
                ...
            case ntn.IfElse(cond, then_body, else_body):
                ...
            case ntn.Assign(lhs,rhs):
                ...
            case ntn.Stack(obj,type):
                ...
            case ntn.Slot(name, type):
                ...
            case ntn.Unpack(lhs, rhs):
                ...
            case ntn.Repack(val, obj):
                ...
            case ntn.Declare(tns, init, op, shape):
                ...
            case ntn.Freeze(tns, op):
                ...
            case ntn.Thaw(tns, op):
                ...
            case ntn.Block(bodies):
                ...
            case ntn.Function(name,args,body):
                ...
            case ntn.Return(val):
                ...


class FinchJLCompiler(NotationCompiler):
    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for func in prgm.children:
            kernel_dict[func.name.name] = FinchJLKernel(func.name.name, generator(func))

        return FinchJLLibrary(kernel_dict)
