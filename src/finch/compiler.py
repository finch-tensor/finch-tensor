import finchlite

from finchlite import (
    Loop,
    Variable,
    Index,
    NotationStatement,
    NotationModule,
)

class FinchJLKernel(finchlite.AssemblyKernel):
    def __call__(self, *args: FinchJLTensor...) -> tuple[FinchJLTensor...]:
        ...
    
class FinchJLLibrary(finchlite.AssemblyLibrary):
    kernels: dict[str, FinchJLKernel]
    def getattr(self, name: str) -> FinchJLKernel:
        return self.kernels[name]

class FinchJLGenerator:

    def __call__(self, prgm: NotationModule) -> FinchJLLibrary:
        match prgm:
            case NotationModule(statements=stmts):
                ...

class FinchJLCompiler(finchlite.NotationCompiler):
    def __call__(self, prgm:NotationModule) -> finchlite.FinchJLLibrary:
        generator = FinchJLGenerator()
        jl_code = generator(prgm)
        return eval(jl_code)