import uuid

import numpy as np

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple
from finch.algebra.fill import (
    AbstractFill,
    DynamicFill,
    DynamicFillError,
    StaticFill,
    is_dynamic,
)
from finch.algebra.ftypes import ftype
from finch.algebra.tensor import TensorFType
from finch.compile import NotationCompiler, dimension
from finch.finch_assembly import AssemblyKernel, AssemblyLibrary
from finch.symbolic import PostWalk, Rewrite
from finch.tensor.patterns import PatternTensorFType

from .julia import jl
from .runtime import DefaultFinchJLRuntime, FinchJLRuntime
from .types import (
    _leaf_type_str,
    ftype_to_jl_constructor_str,
    ftype_to_jl_type_str,
)

_JULIA_OPS = {
    # arithmetic
    ffuncs.add.ftype: "+",
    ffuncs.mul.ftype: "*",
    ffuncs.sub.ftype: "-",
    ffuncs.truediv.ftype: "/",
    ffuncs.floordiv.ftype: "div",
    ffuncs.mod.ftype: "mod",
    ffuncs.pow.ftype: "^",
    ffuncs.neg.ftype: "-",
    ffuncs.pos.ftype: "+",
    ffuncs.divide.ftype: "/",
    ffuncs.remainder.ftype: "mod",
    # comparisons
    ffuncs.eq.ftype: "==",
    ffuncs.equal.ftype: "==",
    ffuncs.ne.ftype: "!=",
    ffuncs.not_equal.ftype: "!=",
    ffuncs.lt.ftype: "<",
    ffuncs.less.ftype: "<",
    ffuncs.le.ftype: "<=",
    ffuncs.less_equal.ftype: "<=",
    ffuncs.gt.ftype: ">",
    ffuncs.greater.ftype: ">",
    ffuncs.ge.ftype: ">=",
    ffuncs.greater_equal.ftype: ">=",
    # bitwise / logical
    ffuncs.and_.ftype: "&",
    ffuncs.or_.ftype: "|",
    ffuncs.not_.ftype: "!",
    ffuncs.invert.ftype: "~",
    ffuncs.lshift.ftype: "<<",
    ffuncs.rshift.ftype: ">>",
    ffuncs.logical_and.ftype: "Finch.and",
    ffuncs.logical_or.ftype: "Finch.or",
    ffuncs.logical_not.ftype: "!",
    ffuncs.logical_xor.ftype: "xor",
    # math / elementwise
    ffuncs.max.ftype: "max",
    ffuncs.min.ftype: "min",
    # misc
    ffuncs.divmod.ftype: "divrem",
    ffuncs.square.ftype: "abs2",
    ffuncs.reciprocal.ftype: "inv",
    ffuncs.atan2.ftype: "atan",
    ffuncs.conjugate.ftype: "conj",
    ffuncs.where.ftype: "ifelse",
    ffuncs.clip.ftype: "clamp",
    ffuncs.truth.ftype: "Bool",
    ffuncs.first_arg.ftype: "first_arg",
}

_JULIA_REDUCTION_OPS = {
    ffuncs.add.ftype: "+",
    ffuncs.mul.ftype: "*",
    ffuncs.max.ftype: "<<max>>",
    ffuncs.min.ftype: "<<min>>",
    ffuncs.and_.ftype: "&",
    ffuncs.or_.ftype: "|",
    ffuncs.logical_and.ftype: "&",
    ffuncs.logical_or.ftype: "|",
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


class FinchJLKernel(AssemblyKernel):
    """A callable Julia kernel."""

    def __init__(
        self,
        jl_code,
        type_,
        finch_program: ntn.Function,
        runtime: FinchJLRuntime,
        func_name: str,
        dynamic_args: tuple[int, ...] = (),
    ):
        super().__init__(type_)
        # We store this code so that we can verify it in pytest
        self.jl_code = jl_code
        self.finch_program = finch_program
        self.runtime = runtime
        self.func_name = func_name
        self.dynamic_args = dynamic_args

    def __call__(self, *args):
        return self.runtime.kernel_call(self.func_name, self, args)


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
                        case ntn.Variable(sym, TensorFType() as type_):
                            arg_name = self.emit_name(sym)
                            fill = type_.fill_value
                            if is_dynamic(fill):
                                raise DynamicFillError(
                                    "Julia backend does not support dynamic fills"
                                )
                            constructor = ftype_to_jl_constructor_str(type_)
                            proto_lines.append(f"        {arg_name} = {constructor}")
                            arg_strs.append(arg_name)
                        case ntn.Variable(_, type_):
                            raise TypeError(
                                f"Julia kernel argument must be a tensor, got {type_}"
                            )
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
                    isinstance(rhs, ntn.Call) and rhs.op.result_type == dimension.ftype
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
                access = f"{tns_str}[{idx_str}]"
                match tns.result_type:
                    case PatternTensorFType() as tensor_type:
                        # Julia masks yield Bool, but Python patterns can carry
                        # numeric dtypes with different arithmetic semantics.
                        elem_t = _leaf_type_str(tensor_type.element_type)
                        return f"{elem_t}({access})"
                return access

            case ntn.Call(op, args):
                arg_strs = [self.generate_julia(arg, nestingLvl) for arg in args]
                if op.result_type == make_tuple.ftype:
                    return ",".join(arg_strs)
                julia_op = _JULIA_OPS.get(op.result_type) or self.generate_julia(
                    op, nestingLvl
                )
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
                match lhs.mode.op.result_type:
                    case ffuncs._InitWriteFType():
                        op = self.generate_julia(lhs.mode.op, nestingLvl)
                        stmt = f"{lhs_str} <<{op}>>= {rhs_str}"
                    case ffuncs._OverwriteFType():
                        stmt = f"{lhs_str} := {rhs_str}"
                    case _:
                        op = _JULIA_REDUCTION_OPS[lhs.mode.op.result_type]
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

            case ntn.Literal(ffuncs._InitWrite(fill=fill)):
                if is_dynamic(fill):
                    raise DynamicFillError(
                        "Julia backend does not support dynamic fills"
                    )
                value = self.generate_julia(ntn.Literal(fill.value), nestingLvl)
                return f"Finch.initwrite({value})"

            case ntn.Literal(val):
                if isinstance(val, AbstractFill) and is_dynamic(val):
                    raise DynamicFillError(
                        "Julia backend does not support dynamic fills"
                    )
                # Julia booleans are lowercase; numpy.bool_ is not a bool subclass.
                if isinstance(val, bool | np.bool_):
                    return "true" if val else "false"
                if isinstance(val, float | np.floating):
                    if np.isinf(val):
                        return "Inf" if val > 0 else "-Inf"
                    if np.isnan(val):
                        return "NaN"
                return str(val)

            case ntn.Variable(name, _):
                # finch uses '#' in generated names; not valid Julia syntax.
                return self.emit_name(name)

            case _:
                # Dimension, Stack, Value are deliberately unimplemented.
                raise Exception(f"Unhandled node type: {type(prgm)}")


def handle_fills(func: ntn.Function) -> tuple[ntn.Function, tuple[int, ...]]:
    """Pin dynamic fills to zero before Julia source generation."""

    dynamic_args = tuple(
        position
        for position, arg in enumerate(func.args)
        if is_dynamic(getattr(arg.type_, "fill_value", None))
    )

    def rule(node):
        match node:
            case ntn.Variable(name, TensorFType() as type_) if is_dynamic(
                type_.fill_value
            ):
                return ntn.Variable(
                    name, type_.with_fill(ftype(type_.fill_value.value)(0))
                )
            case ntn.Literal(DynamicFill() as fill):
                return ntn.Literal(ftype(fill.value)(0))
            case ntn.Literal(StaticFill() as fill):
                return ntn.Literal(fill.value)
        return None

    return Rewrite(PostWalk(rule))(func), dynamic_args


def _argument_type_str(arg: ntn.Variable) -> str:
    match arg.type_:
        case TensorFType() as type_:
            return ftype_to_jl_type_str(type_)
        case type_:
            raise TypeError(f"Julia kernel argument must be a tensor, got {type_}")


class FinchJLCompiler(NotationCompiler):
    def __init__(self, runtime: FinchJLRuntime | None = None):
        self.runtime = DefaultFinchJLRuntime() if runtime is None else runtime

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for orig_func in prgm.children:
            func, dynamic_args = handle_fills(orig_func)
            generated_prgm = generator(func)
            arg_type_strs = tuple(_argument_type_str(arg) for arg in func.args)
            key = (generated_prgm, arg_type_strs, dynamic_args)
            kernel = self.runtime.get_cached_kernel(key)
            if kernel is None:
                func_name = f"kernel_{uuid.uuid4().hex}"
                jl.seval(generated_prgm.replace(func.name.name, func_name, 1))
                kernel = FinchJLKernel(
                    generated_prgm.replace(func.name.name, func_name, 1),
                    func.name.result_type,
                    func,
                    self.runtime,
                    func_name,
                    dynamic_args,
                )
                self.runtime.cache_kernel(key, kernel)
            elif kernel.ftype != func.name.result_type:
                kernel = FinchJLKernel(
                    kernel.jl_code,
                    func.name.result_type,
                    func,
                    self.runtime,
                    kernel.func_name,
                    dynamic_args,
                )
                self.runtime.cache_kernel(key, kernel)
            kernel_dict[func.name.name] = kernel

        return FinchJLLibrary(kernel_dict)
