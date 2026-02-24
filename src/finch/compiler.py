import operator

import finchlite.finch_notation.nodes as ntn
from finchlite.algebra import make_tuple
from finchlite.compile import NotationCompiler, dimension
from finchlite.finch_assembly import AssemblyKernel, AssemblyLibrary

from .julia import jl
from .tensor import FinchJLTensor

ops_map = {operator.add: "+", operator.mul: "*"}
ops_to_ignore = [make_tuple]


class FinchJLKernel(AssemblyKernel):
    def __init__(self, func_name, jl_code):
        # We store this code so that we can verify it in pytest
        self.jl_code = jl_code
        self.func_name = func_name
        jl.seval(self.jl_code)

    def __call__(self, *args: tuple[FinchJLTensor, ...]) -> tuple[FinchJLTensor, ...]:
        finch_fn = getattr(jl, self.func_name)
        result = finch_fn(*[arg._obj for arg in args])

        # The finch function returns tuples when multiple values are returned
        # or a non-tuple when a single value is returned.
        if not isinstance(result, tuple):
            result = (result,)
        return tuple(FinchJLTensor(res) for res in result)


class FinchJLLibrary(AssemblyLibrary):
    def __init__(self, kernel_dict):
        self.kernel_dict = kernel_dict

    def __getattr__(self, name: str) -> FinchJLKernel:
        return self.kernel_dict[name]


# Test with
# https://github.com/finch-tensor/finch-tensor-lite/blob/main/tests/test_notation_interpreter.py
class FinchJLGenerator:
    def __init__(self):
        self.pack_dict = {}
        self.in_finch_block = False

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        self.pack_dict.clear()
        self.in_finch_block = False
        return self.generate_julia(prgm)

    def generate_julia(self, prgm, nestingLvl=0):
        match prgm:
            case ntn.Function(name, args, body):
                body_str = self.generate_julia(body, nestingLvl + 1)
                arg_str = ",".join(
                    [self.generate_julia(arg, nestingLvl) for arg in args]
                )
                return f"function {name}({arg_str})\n{body_str}end"

            case ntn.Block(bodies):
                body_str = ""
                for body in bodies:
                    curr_body_str = self.generate_julia(body, nestingLvl)
                    if curr_body_str != "":
                        body_str += f"{curr_body_str}\n"
                return body_str

            case ntn.Assign(lhs, rhs):
                # TODO: Can we make this better?
                # Special condition to ignore all assigns associated with
                # finding loop bounds
                if isinstance(rhs, ntn.Dimension) or (
                    isinstance(rhs, ntn.Call) and rhs.op.val == dimension
                ):
                    return ""

                tab_str = "    " * nestingLvl
                return (
                    f"{tab_str}{self.generate_julia(lhs, nestingLvl)} = "
                    f"{self.generate_julia(rhs, nestingLvl)}"
                )

            case ntn.Declare(tns, init, op, _):
                # TODO: what is the purpose of op here
                tab_str = "    " * nestingLvl
                return (
                    f"{tab_str}@finch {self.generate_julia(tns, nestingLvl)} .= "
                    f"{self.generate_julia(init, nestingLvl)}"
                )

            case ntn.Return(val):
                tab_str = "    " * nestingLvl
                return f"{tab_str}return {self.generate_julia(val, nestingLvl)}"

            case ntn.Loop(idx, _, body):
                tab_str = "    " * nestingLvl
                tab_str_1 = "    " * (nestingLvl + 1)

                is_outermost_loop = False
                if self.in_finch_block is False:
                    is_outermost_loop = True
                    self.in_finch_block = True
                    loop_body = self.generate_julia(body, nestingLvl + 2)
                else:
                    loop_body = self.generate_julia(body, nestingLvl + 1)

                if not is_outermost_loop:
                    return f"{tab_str}for {idx.name} = _\n{loop_body}{tab_str}end\n"
                self.in_finch_block = False
                return (
                    f"{tab_str}@finch begin\n{tab_str_1}for {idx.name} = "
                    f"_\n{loop_body}{tab_str_1}end\n{tab_str}end"
                )

            case ntn.Access(tns, _, idxs):
                tns_str = self.generate_julia(tns, nestingLvl)
                idx_str = ",".join(
                    [self.generate_julia(idx, nestingLvl) for idx in idxs]
                )
                return f"{tns_str}[{idx_str}]"

            case ntn.Call(op, args):
                arg_str = ",".join(
                    [self.generate_julia(arg, nestingLvl) for arg in args]
                )
                if op.val in ops_to_ignore:
                    return f"{arg_str}"
                return f"{ops_map[op.val]}({arg_str})"

            case ntn.If(cond, body):
                tab_str = "    " * nestingLvl
                cond_str = self.generate_julia(cond, nestingLvl)
                body_str = self.generate_julia(body, nestingLvl + 1)
                return f"{tab_str}if {cond_str}\n{body_str}\n{tab_str}end"

            case ntn.IfElse(cond, then_body, else_body):
                cond_str = self.generate_julia(cond, nestingLvl)
                then_body_str = self.generate_julia(then_body, nestingLvl + 1)
                else_body_str = self.generate_julia(else_body, nestingLvl + 1)
                return (
                    f"{tab_str}if {cond_str}\n{then_body_str}\n"
                    f"{tab_str}else\n{else_body_str}\n{tab_str}end"
                )

            case ntn.Increment(lhs, rhs):
                tab_str = "    " * nestingLvl
                lhs_str = self.generate_julia(lhs, nestingLvl)
                rhs_str = self.generate_julia(rhs, nestingLvl)

                # TODO: Is this the correct assumption to make
                if not (
                    isinstance(lhs, ntn.Access) and isinstance(lhs.mode, ntn.Update)
                ):
                    raise Exception("Increment expects the lhs to be an access")

                return f"{tab_str}{lhs_str} {ops_map[lhs.mode.op.val]}= {rhs_str}"

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
                # finch tensor lite uses character(#) in the naming of variables
                # that however is not valid julia syntax
                return name.replace("#", "_")

            # TODO: Cached, Dimension, Thaw, Stack, Value are unimplemented.
            case _:
                raise Exception(f"Unhandled node type: {type(prgm)}")


class FinchJLCompiler(NotationCompiler):
    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for func in prgm.children:
            generated_prgm = generator(func)
            kernel_dict[func.name.name] = FinchJLKernel(func.name.name, generated_prgm)

        return FinchJLLibrary(kernel_dict)
