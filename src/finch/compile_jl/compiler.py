import hashlib
from typing import ClassVar

import numpy as np

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra.algebra import FinchOperator
from finch.algebra.ffuncs import make_tuple, overwrite
from finch.algebra.ftypes import FType
from finch.algebra.tensor import TensorFType
from finch.compile import NotationCompiler, dimension
from finch.finch_assembly import AssemblyKernel, AssemblyLibrary

from . import dtypes as jl_dtypes
from .interop import jl_tensor_to_python, tensor_to_jl
from .julia import jl

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


def _representative_jl_arg(ftype: FType):
    """
    A small representative Julia value for `ftype`, suitable for
    `@finch_kernel`'s `args`, which needs an instance (not just a type) to
    infer each argument's Julia type from.
    """
    if isinstance(ftype, TensorFType):
        shape = tuple(0 for _ in range(ftype.ndim))
        return tensor_to_jl(ftype.construct(shape))
    return jl_dtypes.to_jl_value(ftype, 0)


class FinchJLKernel(AssemblyKernel):
    def __init__(self, func_name: str, body_str: str, args: list[tuple[str, FType]]):
        self.func_name = func_name
        for name, ftype in args:
            setattr(jl.Main, name, _representative_jl_arg(ftype))
        arg_str = ",".join(name for name, _ in args)
        # We store this code so that we can verify it in pytest
        self.jl_code = (
            f"eval(@finch_kernel {func_name}({arg_str}) = begin\n{body_str}\nend)"
        )
        jl.seval(self.jl_code)

    def __call__(self, *args):
        finch_fn = getattr(jl, self.func_name)
        raw_args = [tensor_to_jl(arg) for arg in args]
        result = finch_fn(*raw_args)

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
        # (jl arg name, FType) for the top-level ntn.Function's arguments, in order.
        self.args: list[tuple[str, FType]] = []

    def __call__(self, prgm: ntn.Module) -> str:
        self.pack_dict.clear()
        self.names.clear()
        self.args = []
        return self.generate_julia(prgm)

    def emit_name(self, sym: str) -> str:
        """
        Canonical Julia identifier for each notation symbol. Avoids
        issues with gensym making duplicate programs appear distinct.
        """
        return self.names.setdefault(sym, f"_v{len(self.names)}")

    def generate_julia(self, prgm, nestingLvl=0):
        match prgm:
            case ntn.Function(name, args, body):
                body_str = self.generate_julia(body, nestingLvl)
                self.args = []
                for arg in args:
                    match arg:
                        case ntn.Variable(sym, type_):
                            self.args.append((self.emit_name(sym), type_))
                        case _:
                            raise NotImplementedError
                return body_str

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
    # Keyed on (body text, arg ftypes): a `@finch_kernel`-generated function is
    # monomorphic, so two calls with the same program shape but different
    # argument dtypes/formats need distinct cached Julia functions.
    _kernels: ClassVar[dict[tuple[str, tuple[FType, ...]], FinchJLKernel]] = {}

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for func in prgm.children:
            body_str = generator(func)
            arg_ftypes = tuple(ftype for _, ftype in generator.args)
            cache_key = (body_str, arg_ftypes)
            kernel = self._kernels.get(cache_key)
            if kernel is None:
                # Give every distinct program its own Julia symbol to avoid
                # conflicts in the julia runtime.
                digest = hashlib.sha256(
                    (body_str + repr(arg_ftypes)).encode()
                ).hexdigest()
                jl_name = f"{func.name.name}_{digest}"
                kernel = FinchJLKernel(jl_name, body_str, generator.args)
                self._kernels[cache_key] = kernel
            kernel_dict[func.name.name] = kernel

        return FinchJLLibrary(kernel_dict)
