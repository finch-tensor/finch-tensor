from .tensor import FinchJLTensor
from finchlite import NotationCompiler, AssemblyKernel, AssemblyLibrary
import finchlite.finch_notation.nodes as ntn

from juliacall import Main as jl 

CompiledModuleName = "compiled_module"
FunctionName = "compiled_function"

class FinchJLKernel(AssemblyKernel):
    def __init__(self, func_name, jl_code): 
        self.func_name = func_name
        jl.seval(jl_code)
    def __call__(
        self, *args: tuple[FinchJLTensor, ...]
    ) -> tuple[FinchJLTensor, ...]:
        argList = []       
        for arg in args:
            argList.append(f"arg{len(argList)}")
            setattr(jl, argList[-1], arg)

        jl.seval(f"{self.func_name}({argList.join(',')})")

class FinchJLLibrary(AssemblyLibrary):
    def __init__(self, kernel_name, kernel):
        self.kernel_dict = {kernel_name: kernel}

    def __getattr__(self, name: str) -> FinchJLKernel:
        return self.kernels[name]


class FinchJLGenerator:
    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        match prgm:
            case ntn.Module(statements=stmts):
                ...

class FinchJLCompiler(NotationCompiler):
    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()
        jl_code = generator(prgm)
        return FinchJLLibrary(CompiledModuleName, FinchJLKernel(FunctionName, jl_code))