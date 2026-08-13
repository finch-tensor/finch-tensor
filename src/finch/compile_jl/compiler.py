import uuid
from typing import ClassVar

import numpy as np

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra.algebra import FinchOperator
from finch.algebra.ffuncs import make_tuple, overwrite
from finch.compile import NotationCompiler, dimension
from finch.finch_assembly import AssemblyKernel, AssemblyLibrary

from .interop import jl_tensor_to_python, tensor_to_jl
from .julia import jl
from .types import ftype_to_jl_constructor_str, ftype_to_jl_type_str

# Single source of truth for Python ffunc name → Julia operator/function name.
# max/min use Finch's <<op>> semiring syntax; they appear only as Aggregate
# (reduction) nodes, never as plain element-wise Calls, so this is safe.
_JULIA_NAMES = {
    # arithmetic
    "add": "+",
    "mul": "*",
    "sub": "-",
    "truediv": "/",
    "divide": "/",
    "floordiv": "div",
    "mod": "mod",
    "remainder": "mod",
    "pow": "^",
    "neg": "-",
    "pos": "+",
    # comparisons
    "eq": "==",
    "equal": "==",
    "ne": "!=",
    "not_equal": "!=",
    "lt": "<",
    "less": "<",
    "le": "<=",
    "less_equal": "<=",
    "gt": ">",
    "greater": ">",
    "ge": ">=",
    "greater_equal": ">=",
    # bitwise / logical
    "and_": "&",
    "or_": "|",
    "not_": "!",
    "invert": "~",
    "lshift": "<<",
    "rshift": ">>",
    "logical_and": "&",
    "logical_or": "|",
    "logical_not": "!",
    "logical_xor": "xor",
    # reductions (Finch <<op>>= semiring syntax)
    "max": "<<max>>",
    "min": "<<min>>",
    # misc
    "divmod": "divrem",
    "square": "abs2",
    "reciprocal": "inv",
    "atan2": "atan",
    "conjugate": "conj",
    "where": "ifelse",
    "clip": "clamp",
    "truth": "Bool",
}

# Names of ffuncs that are valid as reduction operators.
_REDUCTION_OPS = {
    "add",
    "mul",
    "max",
    "min",
    "and_",
    "or_",
    "logical_and",
    "logical_or",
}


def _ops_for(names=None) -> dict:
    """Build a FinchOperator → Julia-name map from _JULIA_NAMES."""
    return {
        obj: _JULIA_NAMES.get(n, n)
        for n in (names if names is not None else dir(ffuncs))
        if isinstance(obj := getattr(ffuncs, n, None), FinchOperator)
    }


ops_map = _ops_for()
red_ops_map = _ops_for(_REDUCTION_OPS)
ops_to_ignore = [make_tuple]


class CompiledJLKernel:
    """Pure-data compiled-but-not-evaluated kernel: self-contained Julia
    source text, with no Python-side values left to inject."""

    def __init__(self, func_name: str, jl_code: str):
        self.func_name = func_name
        self.jl_code = jl_code

    def evaluate(self) -> "FinchJLKernel":
        """Defines the kernel function in the running Julia session,
        returning the now-callable kernel."""
        jl.seval(self.jl_code)
        return FinchJLKernel(self.func_name, self.jl_code)


class FinchJLKernel(AssemblyKernel):
    """A kernel already defined (evaluated) in the running Julia session."""

    def __init__(self, func_name, jl_code):
        # We store this code so that we can verify it in pytest
        self.jl_code = jl_code
        self.func_name = func_name

    def __call__(self, *args):
        finch_fn = getattr(jl, self.func_name)
        raw_args = [tensor_to_jl(arg) for arg in args]
        result = finch_fn(*raw_args)

        # @finch_kernel-generated functions return a NamedTuple keyed by the
        # returned variable name(s), unlike @finch's bare Tensor/tuple.
        if jl.isa(result, jl.NamedTuple):
            result = jl.values(result)

        # The finch function returns tuples when multiple values are returned
        # or a non-tuple when a single value is returned.
        if jl.isa(result, jl.Finch.Tensor):
            return (jl_tensor_to_python(result),)
        return tuple(jl_tensor_to_python(res) for res in result)


class FinchJLLibrary(AssemblyLibrary):
    def __init__(self, kernel_dict):
        self.kernel_dict = kernel_dict

    def __getattr__(self, name: str) -> FinchJLKernel:
        return self.kernel_dict[name]


class FinchJLGenerator:
    def __init__(self):
        self.pack_dict = {}
        self.names: dict[str, str] = {}

    def __call__(self, prgm: ntn.Module) -> str:
        self.pack_dict.clear()
        self.names.clear()
        return self.generate_julia(prgm)

    def emit_name(self, sym: str) -> str:
        return self.names.setdefault(sym, f"v{len(self.names)}")

    def generate_julia(self, prgm, nestingLvl=0):
        match prgm:
            case ntn.Function(name, args, body):
                body_str = self.generate_julia(body, nestingLvl + 2)
                arg_strs = []
                proto_lines = []
                for arg in args:
                    match arg:
                        case ntn.Variable(sym, type_):
                            arg_name = self.emit_name(sym)
                            proto_lines.append(
                                f"        {arg_name} = "
                                f"{ftype_to_jl_constructor_str(type_)}"
                            )
                            arg_strs.append(arg_name)
                        case _:
                            raise NotImplementedError
                arg_str = ",".join(arg_strs)
                proto_str = "\n".join(proto_lines)
                return (
                    "eval(let\n"
                    f"{proto_str}\n"
                    f"    Finch.@finch_kernel function {name}({arg_str})\n"
                    f"{body_str}\n    end\n"
                    "end)"
                )

            case ntn.Block(bodies):
                body_str = ""
                body_strs = [self.generate_julia(body, nestingLvl) for body in bodies]
                body_strs = [body_str for body_str in body_strs if body_str != ""]
                return "\n".join(body_strs)

            case ntn.Assign(lhs, rhs):
                # Ignore assigns used only to find loop bounds.
                if isinstance(rhs, ntn.Dimension) or (
                    isinstance(rhs, ntn.Call) and rhs.op.val == dimension
                ):
                    return ""

                tab_str = "    " * nestingLvl
                stmt = (
                    f"{self.generate_julia(lhs, nestingLvl)} = "
                    f"{self.generate_julia(rhs, nestingLvl)}"
                )
                return f"{tab_str}{stmt}"

            case ntn.Declare(tns, init, _, _):
                tab_str = "    " * nestingLvl
                return (
                    f"{tab_str}{self.generate_julia(tns, nestingLvl)} .= "
                    f"{self.generate_julia(init, nestingLvl)}"
                )

            case ntn.Return(val):
                tab_str = "    " * nestingLvl
                return f"{tab_str}return {self.generate_julia(val, nestingLvl)}"

            case ntn.Loop(idx, _, body):
                tab_str = "    " * nestingLvl
                idx_str = self.generate_julia(idx, nestingLvl)
                loop_body = self.generate_julia(body, nestingLvl + 1)
                return f"{tab_str}for {idx_str} = _\n{loop_body}\n{tab_str}end"

            case ntn.Access(tns, _, idxs):
                tns_str = self.generate_julia(tns, nestingLvl)
                idx_str = ",".join(
                    [self.generate_julia(idx, nestingLvl) for idx in reversed(idxs)]
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
                tab_str = "    " * nestingLvl
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
                if lhs.mode.op.val == overwrite:
                    stmt = f"{lhs_str} = {rhs_str}"
                else:
                    stmt = f"{lhs_str} {red_ops_map[lhs.mode.op.val]}= {rhs_str}"
                return f"{tab_str}{stmt}"

            case ntn.Unwrap(arg):
                return self.generate_julia(arg, nestingLvl)

            case ntn.Unpack(lhs, rhs):
                if not isinstance(rhs, ntn.Variable):
                    raise Exception("The unpack was not called with variable as RHS.")
                self.pack_dict[lhs.name] = self.generate_julia(rhs, nestingLvl)
                return ""

            case ntn.Repack(val, _):
                self.pack_dict.pop(val.name)
                return ""

            case ntn.Freeze(_, _):
                return ""

            case ntn.Thaw(_, _):
                return ""

            case ntn.Cached(_, _):
                return ""

            case ntn.Slot(name):
                if name not in self.pack_dict:
                    raise Exception(f"{name} Slot does not exist in registry.")
                return self.pack_dict[name]

            case ntn.Literal(val):
                # Julia booleans are lowercase; numpy.bool_ is not a bool subclass.
                if isinstance(val, bool | np.bool_):
                    return "true" if val else "false"
                if isinstance(val, float | np.floating) and np.isinf(val):
                    return "Inf" if val > 0 else "-Inf"
                return str(val)

            case ntn.Variable(name, _):
                # finch uses '#' in generated names; not valid Julia syntax.
                return self.emit_name(name)

            case _:
                # Dimension, Stack, Value are deliberately unimplemented.
                raise Exception(f"Unhandled node type: {type(prgm)}")


class FinchJLCompiler(NotationCompiler):
    # Keyed by (generated source, per-arg Julia type strings): the generated
    # source alone isn't self-describing here -- argument types are inferred
    # by @finch_kernel from prototype *values*, not written into the source
    # text, so two calls with identical bodies but different argument types
    # would otherwise collide on the same cache entry.
    _kernels: ClassVar[dict[tuple[str, tuple[str, ...]], FinchJLKernel]] = {}

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for func in prgm.children:
            generated_prgm = generator(func)
            arg_type_strs = tuple(ftype_to_jl_type_str(arg.type_) for arg in func.args)
            key = (generated_prgm, arg_type_strs)
            kernel = self._kernels.get(key)
            if kernel is None:
                jl_name = f"kernel_{uuid.uuid4().hex}"
                compiled = CompiledJLKernel(
                    jl_name,
                    generated_prgm.replace(func.name.name, jl_name, 1),
                )
                kernel = compiled.evaluate()
                self._kernels[key] = kernel
            kernel_dict[func.name.name] = kernel

        return FinchJLLibrary(kernel_dict)
