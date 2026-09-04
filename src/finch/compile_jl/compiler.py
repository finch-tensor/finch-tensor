import uuid
from typing import ClassVar

import numpy as np

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple, overwrite
from finch.algebra.fill import (
    AbstractFill,
    DynamicFill,
    DynamicFillError,
    StaticFill,
    is_dynamic,
)
from finch.algebra.ftypes import ftype
from finch.compile import NotationCompiler, dimension
from finch.finch_assembly import AssemblyKernel, AssemblyLibrary
from finch.symbolic import PostWalk, Rewrite

from .interop import jl_tensor_to_python, tensor_to_jl
from .julia import jl
from .types import ftype_to_jl_constructor_str, ftype_to_jl_type_str

_JULIA_OPS = {
    # arithmetic
    ffuncs.add: "+",
    ffuncs.mul: "*",
    ffuncs.sub: "-",
    ffuncs.truediv: "/",
    ffuncs.floordiv: "div",
    ffuncs.mod: "mod",
    ffuncs.pow: "^",
    ffuncs.neg: "-",
    ffuncs.pos: "+",
    # comparisons
    ffuncs.eq: "==",
    ffuncs.ne: "!=",
    ffuncs.lt: "<",
    ffuncs.le: "<=",
    ffuncs.gt: ">",
    ffuncs.ge: ">=",
    # bitwise / logical
    ffuncs.and_: "&",
    ffuncs.or_: "|",
    ffuncs.not_: "!",
    ffuncs.invert: "~",
    ffuncs.lshift: "<<",
    ffuncs.rshift: ">>",
    ffuncs.logical_and: "Finch.and",
    ffuncs.logical_or: "Finch.or",
    ffuncs.logical_not: "!",
    ffuncs.logical_xor: "xor",
    # misc
    ffuncs.divmod: "divrem",
    ffuncs.square: "abs2",
    ffuncs.reciprocal: "inv",
    ffuncs.atan2: "atan",
    ffuncs.conjugate: "conj",
    ffuncs.where: "ifelse",
    ffuncs.clip: "clamp",
    ffuncs.truth: "Bool",
}

_JULIA_REDUCTION_OPS = {
    ffuncs.add: "+",
    ffuncs.mul: "*",
    ffuncs.max: "<<max>>",
    ffuncs.min: "<<min>>",
    ffuncs.and_: "&",
    ffuncs.or_: "|",
    ffuncs.logical_and: "&",
    ffuncs.logical_or: "|",
}
_INFIX_OPS = {
    "+",
    "*",
    "-",
    "/",
    "^",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "&",
    "|",
    "<<",
    ">>",
}


class CompiledJLKernel:
    """Pure-data compiled-but-not-evaluated kernel: self-contained Julia
    source text, with no Python-side values left to inject."""

    def __init__(
        self, func_name: str, jl_code: str, dynamic_args: tuple[int, ...] = ()
    ):
        self.func_name = func_name
        self.jl_code = jl_code
        self.dynamic_args = dynamic_args

    def evaluate(self) -> "FinchJLKernel":
        """Defines the kernel function in the running Julia session,
        returning the now-callable kernel."""
        jl.seval(self.jl_code)
        return FinchJLKernel(self.func_name, self.jl_code, self.dynamic_args)


class FinchJLKernel(AssemblyKernel):
    """A kernel already defined (evaluated) in the running Julia session."""

    def __init__(self, func_name, jl_code, dynamic_args: tuple[int, ...] = ()):
        # We store this code so that we can verify it in pytest
        self.jl_code = jl_code
        self.func_name = func_name
        # Argument positions with dynamic fill values that are
        # arbitrarily set to zero. Other arguments keep their
        # Known fills.
        self.dynamic_args = dynamic_args
        jl.seval(self.jl_code)

    def __call__(self, *args):
        finch_fn = getattr(jl, self.func_name)
        raw_args = [
            tensor_to_jl(arg, pin_fill=i in self.dynamic_args)
            for i, arg in enumerate(args)
        ]
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

    def __call__(self, prgm: ntn.Module | ntn.Function) -> str:
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
                arg_strs = [self.generate_julia(arg, nestingLvl) for arg in args]
                if op.val == make_tuple:
                    return ",".join(arg_strs)
                julia_op = _JULIA_OPS.get(op.val, repr(op.val))
                if len(arg_strs) > 1 and julia_op in _INFIX_OPS:
                    return "(" + f" {julia_op} ".join(arg_strs) + ")"
                return f"{julia_op}(" + ",".join(arg_strs) + ")"

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
                    op = _JULIA_REDUCTION_OPS[lhs.mode.op.val]
                    stmt = f"{lhs_str} {op}= {rhs_str}"
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
                if isinstance(val, AbstractFill):
                    # str() would silently emit broken source.
                    raise DynamicFillError(
                        "cannot emit a wrapped fill as a Julia literal"
                    )
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


def handle_fills(func: ntn.Function) -> tuple[ntn.Function, tuple[int, ...]]:
    """Rewrite every Dynamic fill in `func` to a zero of its dtype, and report
    which argument positions carried a Dynamic fill. This is a necessary but
    potentially unsound rewrite which should be removed eventually.
    """
    dynamic_args = tuple(
        i
        for i, arg in enumerate(func.args)
        if is_dynamic(getattr(arg.type_, "fill_value", None))
    )

    def rule(node):
        match node:
            case ntn.Literal(DynamicFill() as fill):
                return ntn.Literal(ftype(fill.value)(0))
            case ntn.Literal(StaticFill() as fill):
                return ntn.Literal(fill.value)
        return None

    return Rewrite(PostWalk(rule))(func), dynamic_args


class FinchJLCompiler(NotationCompiler):
    # Keyed by (generated source, per-arg Julia type strings): the generated
    # source alone isn't self-describing here -- argument types are inferred
    # by @finch_kernel from prototype *values*, not written into the source
    # text, so two calls with identical bodies but different argument types
    # would otherwise collide on the same cache entry.
    _kernels: ClassVar[
        dict[tuple[str, tuple[str, ...], tuple[int, ...]], FinchJLKernel]
    ] = {}

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for orig_func in prgm.children:
            func, dynamic_args = handle_fills(orig_func)
            generated_prgm = generator(func)
            arg_type_strs = tuple(
                ftype_to_jl_type_str(arg.type_)
                for arg in func.args
                if arg.type_ is not None
            )
            # Flat key: source, argument types, and which fills were pinned. All
            # three vary independently, so none may be folded into another.
            key = (generated_prgm, arg_type_strs, dynamic_args)
            kernel = self._kernels.get(key)
            if kernel is None:
                jl_name = f"kernel_{uuid.uuid4().hex}"
                compiled = CompiledJLKernel(
                    jl_name,
                    generated_prgm.replace(func.name.name, jl_name, 1),
                    dynamic_args=dynamic_args,
                )
                kernel = compiled.evaluate()
                self._kernels[key] = kernel
            kernel_dict[func.name.name] = kernel

        return FinchJLLibrary(kernel_dict)
