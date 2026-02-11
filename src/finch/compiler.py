from finchlite.compile import NotationCompiler
from finchlite.finch_assembly import AssemblyKernel, AssemblyLibrary
import finchlite.finch_notation.nodes as ntn
from finchlite.compile import dimension
from typing import Any

import operator

from juliacall import Main as jl

ops_map = {operator.add: "+", operator.mul: "*"}


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
    def __init__(self):
        self.pack_dict = {}

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        self.pack_dict.clear()
        return self.generate_julia(prgm)

    def generate_julia(self, prgm, nestingLvl=0):
        match prgm:
            case ntn.Function(name, args, body):
                body_str = self.generate_julia(body, nestingLvl + 1)
                return f"function {name}\n{body_str}\nend"

            case ntn.Block(bodies):
                body_str = ""
                tab_str = {"    " * nestingLvl}
                for body in bodies:
                    body_str += f"{tab_str}{self.generate_julia(body, nestingLvl)}\n"

            case ntn.Assign(lhs, rhs):
                # TODO: Can we make this better?
                # Special condition to ignore all assigns associated with
                # finding loop bounds
                if isinstance(rhs, ntn.Call) and rhs.op.val == dimension:
                    return ""
                return f"{self.generate_julia(lhs, nestingLvl)} = {self.generate_julia(body, nestingLvl)}"

            case ntn.Declare(tns, init, op, shape):
                # TODO: what is the purpose of op here
                return f"@finch {self.generate_julia(tns, nestingLvl)} .= {self.generate_julia(init, nestingLvl)}"

            case ntn.Return(val):
                return f"return {self.generate_julia(val, nestingLvl)}"

            case ntn.Loop(idx, _, body):
                tab_str = "    " * nestingLvl
                loop_body = self.generate_julia(body, nestingLvl + 1)
                return f"for {idx.name} = _\n{loop_body}\n{tab_str}end"

            case ntn.Access(tns, _, idxs):
                tns_str = self.generate_julia(tns, nestingLvl)
                idx_str = [self.generate_julia(idx, nestingLvl) for idx in idxs].join(
                    ","
                )
                return f"{tns_str}[{idx_str}]"

            case ntn.Call(op, args):
                arg_str = [self.generate_julia(arg, nestingLvl) for arg in args].join(
                    ","
                )
                return f"{ops_map[op.val]}({arg_str})"

            case ntn.If(cond, body):
                tab_str = "    " * nestingLvl
                cond_str = self.generate_julia(cond, nestingLvl)
                body_str = self.generate_julia(body, nestingLvl + 1)
                return f"if {cond_str}\n{body_str}\n{tab_str}end"

            case ntn.IfElse(cond, then_body, else_body):
                cond_str = self.generate_julia(cond, nestingLvl)
                then_body_str = self.generate_julia(then_body, nestingLvl + 1)
                else_body_str = self.generate_julia(else_body, nestingLvl + 1)
                return f"if {cond_str}\n{then_body_str}\n{tab_str}else\n{else_body_str}\n{tab_str}end"

            case ntn.Increment(lhs, rhs):
                lhs_str = self.generate_julia(lhs, nestingLvl)
                rhs_str = self.generate_julia(rhs, nestingLvl)

                # TODO: Is this the correct assumption to make
                if not (
                    isinstance(lhs, ntn.Access) and isinstance(lhs.mode, ntn.Update)
                ):
                    raise Exception("Increment expects the lhs to be an access")

                return f"{lhs_str} {ops_map[lhs.mode.op.val]}= {rhs_str}"

            case ntn.Unwrap(arg):
                return self.generate_julia(arg, nestingLvl)

            case ntn.Unpack(lhs, rhs):
                # TODO: Is this the right assumption to make
                if not isinstance(rhs, ntn.Variable):
                    raise Exception("The unpack was not called with variable as RHS.")
                self.pack_dict[lhs.name] = rhs.name
                return ""

            case ntn.Repack(val, _):
                self.pack_dict.pop(val.name)
                return ""

            case ntn.Freeze(_, _):
                return ""

            case ntn.Slot(name):
                if name not in self.pack_dict:
                    raise Exception(f"{name} Slot does not exist in registry.")
                return self.pack_dict[name]

            case ntn.Literal(val):
                return str(val)

            case ntn.Variable(name, _):
                return name

            # TODO: Cached, Dimension, Thaw, Stack, Value are unimplemented.
            case _:
                raise Exception(f"Unhandled node type: {type(prgm)}")


class FinchJLCompiler(NotationCompiler):
    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for func in prgm.children:
            kernel_dict[func.name.name] = FinchJLKernel(func.name.name, generator(func))

        return FinchJLLibrary(kernel_dict)
