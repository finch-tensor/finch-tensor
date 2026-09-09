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

from .interop import JuliaBufferContext
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
    ffuncs.divide: "/",
    ffuncs.remainder: "mod",
    # comparisons
    ffuncs.eq: "==",
    ffuncs.equal: "==",
    ffuncs.ne: "!=",
    ffuncs.not_equal: "!=",
    ffuncs.lt: "<",
    ffuncs.less: "<",
    ffuncs.le: "<=",
    ffuncs.less_equal: "<=",
    ffuncs.gt: ">",
    ffuncs.greater: ">",
    ffuncs.ge: ">=",
    ffuncs.greater_equal: ">=",
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
    # math / elementwise
    ffuncs.max: "max",
    ffuncs.min: "min",
    # misc
    ffuncs.divmod: "divrem",
    ffuncs.square: "abs2",
    ffuncs.reciprocal: "inv",
    ffuncs.atan2: "atan",
    ffuncs.conjugate: "conj",
    ffuncs.where: "ifelse",
    ffuncs.clip: "clamp",
    ffuncs.truth: "Bool",
    ffuncs.first_arg: "first_arg",
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

    def evaluate(self, buffer_context: JuliaBufferContext) -> "FinchJLKernel":
        """Defines the kernel function in the running Julia session,
        returning the now-callable kernel."""
        jl.seval(self.jl_code)
        return FinchJLKernel(
            self.func_name,
            self.jl_code,
            self.dynamic_args,
            buffer_context=buffer_context,
        )


class FinchJLKernel(AssemblyKernel):
    """A kernel already defined (evaluated) in the running Julia session."""

    _ARG_CACHE_SIZE = 32
    _OUTPUT_POOL_SIZE = 2

    def __init__(
        self,
        func_name,
        jl_code,
        dynamic_args: tuple[int, ...] = (),
        *,
        buffer_context: JuliaBufferContext,
    ):
        # We store this code so that we can verify it in pytest
        self.jl_code = jl_code
        self.func_name = func_name
        # Argument positions with dynamic fill values that are
        # arbitrarily set to zero. Other arguments keep their
        # Known fills.
        self.dynamic_args = dynamic_args
        self.buffer_context = buffer_context
        jl.seval(self.jl_code)
        # JuliaCall attribute lookup is a boundary crossing of its own. Kernel
        # methods are immutable once generated, so retain the callable proxy.
        self._finch_fn = getattr(jl, self.func_name)
        # Keep exact Python-object references, rather than ids, so an object-id
        # reuse can never select an unrelated Julia tensor. The small LRU is
        # enough for invariant inputs plus the active ping-pong state buffers.
        self._arg_cache: list[tuple[object, bool, object]] = []
        # Filled after the first invocation by matching returned Julia tensors
        # to positional arguments. Each position keeps two Julia-backed Python
        # wrappers, allowing an output buffer to be reused without aliasing the
        # current input state.
        self._result_arg_positions: tuple[int, ...] | None = None
        self._output_pools: dict[int, list[object]] = {}

    def _tensor_to_jl(self, arg, *, pin_fill: bool):
        for i, (cached_arg, cached_pin_fill, cached_jl) in enumerate(
            self._arg_cache
        ):
            if cached_arg is arg and cached_pin_fill == pin_fill:
                # Promote the hit so alternating ping-pong buffers stay hot.
                self._arg_cache.append(self._arg_cache.pop(i))
                return cached_jl
        jl_arg = self.buffer_context.tensor_to_jl(arg, pin_fill=pin_fill)
        self._arg_cache.append((arg, pin_fill, jl_arg))
        if len(self._arg_cache) > self._ARG_CACHE_SIZE:
            self._arg_cache.pop(0)
        return jl_arg

    def _recycled_output_args(self, args):
        """Use a non-aliasing Julia-backed output buffer when one is known."""

        if self._result_arg_positions is None:
            return list(args)
        recycled = list(args)
        for arg_pos, pool in self._output_pools.items():
            for candidate in pool:
                if all(
                    candidate is not arg
                    for i, arg in enumerate(args)
                    if i != arg_pos
                ):
                    recycled[arg_pos] = candidate
                    break
        return recycled

    def _learn_result_layout(self, result, raw_args):
        """Recover first-call outputs and learn which arguments they alias."""

        if jl.isa(result, jl.NamedTuple):
            result = jl.values(result)
        result_items = (result,) if jl.isa(result, jl.Finch.Tensor) else tuple(result)
        raw_arg_positions = {
            int(jl.objectid(raw_arg)): i for i, raw_arg in enumerate(raw_args)
        }
        result_arg_positions: list[int] = []
        recovered = []
        for result_item in result_items:
            arg_pos = raw_arg_positions.get(int(jl.objectid(result_item)))
            if arg_pos is None:
                # This kernel's result is not a supplied output buffer, so
                # retain Finch's ordinary recovery behavior.
                return tuple(
                    self.buffer_context.tensor_to_python(item) for item in result_items
                )
            result_arg_positions.append(arg_pos)
            recovered_item = self.buffer_context.tensor_to_python(result_item)
            recovered.append(recovered_item)
            pool = self._output_pools.setdefault(arg_pos, [])
            if all(existing is not recovered_item for existing in pool):
                pool.append(recovered_item)
                if len(pool) > self._OUTPUT_POOL_SIZE:
                    pool.pop(0)
        self._result_arg_positions = tuple(result_arg_positions)
        return tuple(recovered)

    def __call__(self, *args):
        call_args = self._recycled_output_args(args)
        raw_args = [
            self._tensor_to_jl(arg, pin_fill=i in self.dynamic_args)
            for i, arg in enumerate(call_args)
        ]
        result = self._finch_fn(*raw_args)

        if self._result_arg_positions is not None:
            # The generated Finch kernels return their supplied output tensors.
            # Once that layout is learned, the Julia return value need not cross
            # back through tensor recovery at all.
            return tuple(call_args[pos] for pos in self._result_arg_positions)
        return self._learn_result_layout(result, raw_args)

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
    # Kernels are cached process-wide, so their Julia tensor wrappers must be
    # cached process-wide as well. This lets a result produced by one compiled
    # kernel be passed to another without rebuilding its Julia wrapper.
    _buffer_context: ClassVar[JuliaBufferContext] = JuliaBufferContext()

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
                kernel = compiled.evaluate(buffer_context=self._buffer_context)
                self._kernels[key] = kernel
            kernel_dict[func.name.name] = kernel

        return FinchJLLibrary(kernel_dict)
