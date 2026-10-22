import uuid

import finch.algebra.ffuncs as ffuncs
import finch.finch_notation.nodes as ntn
from finch.algebra.ffuncs import make_tuple
from finch.algebra.fill import (
    AbstractFill,
    AbstractFillFType,
    DynamicFill,
    DynamicFillError,
    DynamicFillFType,
    StaticFill,
    StaticFillFType,
    is_dynamic,
)
from finch.compile import NotationCompiler, dimension
from finch.compile.lower import make_extent
from finch.finch_assembly import AssemblyKernel, AssemblyLibrary
from finch.symbolic import PostWalk, Rewrite
from finch.tensor.patterns import FillTensorFType, PatternTensorFType

from .julia import jl
from .runtime import DefaultFinchJLRuntime, FinchJLRuntime
from .types import (
    _julia_literal,
    _leaf_type_str,
    _zero_value,
    ftype_to_jl_constructor_str,
    ftype_to_jl_type_str,
)

_JULIA_OPS = {
    # arithmetic
    ffuncs.add.ftype: "+",
    ffuncs.mul.ftype: "*",
    ffuncs.sub.ftype: "-",
    ffuncs.truediv.ftype: "/",
    # fld floors like numpy; div truncates toward zero
    ffuncs.floordiv.ftype: "fld",
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
        extents: tuple[tuple[int, int, tuple[ffuncs._Cast, ...]], ...] = (),
    ):
        super().__init__(type_)
        # We store this code so that we can verify it in pytest
        self.jl_code = jl_code
        self.finch_program = finch_program
        self.runtime = runtime
        self.func_name = func_name
        self.dynamic_args = dynamic_args
        # The input dimension and casts used to compute each trailing extent.
        self.extents = extents

    def __call__(self, *args):
        return self.runtime.kernel_call(self.func_name, self, args)


class FinchJLLibrary(AssemblyLibrary):
    def __init__(self, kernel_dict):
        self.kernel_dict = kernel_dict

    def __getattr__(self, name: str) -> FinchJLKernel:
        return self.kernel_dict[name]


def register_julia_op(op, julia_name: str, definition: str | None = None) -> None:
    # generated kernels call op as julia_name; definition (Julia source) runs once
    if definition is not None and _JULIA_DEFINITIONS.get(julia_name) != definition:
        jl.seval(definition)
        _JULIA_DEFINITIONS[julia_name] = definition
    _JULIA_OPS[op.ftype] = julia_name


_JULIA_DEFINITIONS: dict[str, str] = {}


class FinchJLGenerator:
    def __init__(self):
        self.pack_dict = {}
        self.names: dict[str, str] = {}
        self.arg_positions: dict[str, int] = {}
        self.slot_args: dict[str, int] = {}
        # Loop extents are passed to the kernel as extra integer arguments, so
        # that loops don't have to infer them from the tensors they read. Each
        # records its Julia name, input position/axis, and call-time casts.
        self.extents: list[tuple[str, int, int, tuple[ffuncs._Cast, ...]]] = []
        self.used_extents: set[str] = set()
        self.constants: dict[str, ntn.Literal] = {}

    def __call__(self, prgm: ntn.Module | ntn.Function) -> str:
        self.pack_dict.clear()
        self.names.clear()
        self.arg_positions.clear()
        self.slot_args.clear()
        self.extents.clear()
        self.used_extents.clear()
        self.constants.clear()
        return self.generate_julia(prgm)

    def emit_name(self, sym: str) -> str:
        return self.names.setdefault(sym, f"v{len(self.names)}")

    def resolve_constant(self, expr):
        match expr:
            case ntn.Literal():
                return expr
            case ntn.Variable(name, _):
                return self.constants.get(name)
            case ntn.Call(ntn.Literal(ffuncs._Cast() as cast), (arg,)):
                match self.resolve_constant(arg):
                    case ntn.Literal(value):
                        return ntn.Literal(cast(value))
        return None

    def shape_extent(self, expr):
        match expr:
            case ntn.Variable(name, _):
                return next(
                    (ext[1:] for ext in self.extents if ext[0] == self.names.get(name)),
                    None,
                )
            case ntn.Call(ntn.Literal(ffuncs._Cast() as cast), (arg,)):
                extent = self.shape_extent(arg)
                if extent is not None:
                    pos, axis, casts = extent
                    return pos, axis, (*casts, cast)
        return None

    def generate_julia(self, prgm, nestingLvl=0):
        match prgm:
            case ntn.Function(name, args, body):
                for pos, arg in enumerate(args):
                    if isinstance(arg, ntn.Variable):
                        self.arg_positions[arg.name] = pos
                        self.emit_name(arg.name)
                body_str = self.generate_julia(body, nestingLvl + 2)
                self.extents = [
                    ext for ext in self.extents if ext[0] in self.used_extents
                ]
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
                for extent, _, _, _ in self.extents:
                    proto_lines.append(f"        {extent} = 1")
                    arg_strs.append(extent)
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
                body_strs = [self.generate_julia(body, nestingLvl) for body in bodies]
                return "\n".join(body for body in body_strs if body)

            case ntn.Assign(
                ntn.Variable(name, _),
                ntn.Dimension(tns, ntn.Literal(axis))
                | ntn.Call(ntn.Literal(), (tns, ntn.Literal(axis))) as rhs,
            ) if isinstance(rhs, ntn.Dimension) or (
                rhs.op.result_type == dimension.ftype
            ):
                # Dimensions are measured in Python and passed as arguments.
                match tns:
                    case ntn.Slot(tns_name, _):
                        pos = self.slot_args[tns_name]
                    case ntn.Variable(tns_name, _):
                        pos = self.arg_positions[tns_name]
                self.extents.append((self.emit_name(name), pos, int(axis), ()))
                return ""

            case ntn.Assign(lhs, rhs):
                constant = self.resolve_constant(rhs)
                if constant is not None:
                    self.constants[lhs.name] = constant
                    return ""
                extent = self.shape_extent(rhs)
                if extent is not None:
                    # Resolve shape aliases and casts before the kernel call,
                    # without opening scopes around tensor declarations.
                    self.extents.append((self.emit_name(lhs.name), *extent))
                    return ""
                raise NotImplementedError(
                    "Julia lowering only supports shape variable bindings"
                )

            case ntn.Declare(tns, init, _, _):
                tab_str = "    " * nestingLvl
                return (
                    f"{tab_str}{self.generate_julia(tns, nestingLvl)} .= "
                    f"{self.generate_julia(init, nestingLvl)}"
                )

            case ntn.Return(ntn.Call(op, args)) if op.result_type == make_tuple.ftype:
                tab_str = "    " * nestingLvl
                arg_strs = [self.generate_julia(arg, nestingLvl) for arg in args]
                return f"{tab_str}return {','.join(arg_strs)}"

            case ntn.Return(val):
                tab_str = "    " * nestingLvl
                return f"{tab_str}return {self.generate_julia(val, nestingLvl)}"

            case ntn.Loop(idx, ext, body):
                tab_str = "    " * nestingLvl
                idx_str = self.generate_julia(idx, nestingLvl)
                match ext:
                    case ntn.Call(ntn.Literal(op), (ntn.Literal(start), stop)) if (
                        op is make_extent
                    ):
                        # Python extents are zero-based and half-open.
                        stop_str = self.generate_julia(stop, nestingLvl)
                        ext_str = f"{int(start) + 1}:{stop_str}"
                    case _:
                        ext_str = "_"
                loop_body = self.generate_julia(body, nestingLvl + 1)
                return f"{tab_str}for {idx_str} = {ext_str}\n{loop_body}\n{tab_str}end"

            case ntn.Access(tns, _, _) if isinstance(tns.result_type, FillTensorFType):
                # A fill tensor is passed as a scalar holding its fill, since
                # loop extents no longer come from the tensors read.
                return f"{self.generate_julia(tns, nestingLvl)}[]"

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
                    return f"({', '.join(arg_strs)}{',' if len(arg_strs) == 1 else ''})"
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
                    case ffuncs._InitWriteFType() | ffuncs._ChooseFType():
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
                self.slot_args[lhs.name] = self.arg_positions[rhs.name]
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

            case ntn.Literal(ffuncs._Cast(dtype=dtype)):
                return _leaf_type_str(dtype)

            case ntn.Literal(ffuncs._InitWrite(fill=fill)):
                if is_dynamic(fill):
                    raise DynamicFillError("Julia init_write requires a static fill")
                value = self.generate_julia(ntn.Literal(fill.value), nestingLvl)
                return f"Finch.initwrite({value})"

            case ntn.Literal(ffuncs._Choose(fill=fill)):
                if is_dynamic(fill):
                    raise DynamicFillError("Julia choose requires a static fill")
                value = self.generate_julia(ntn.Literal(fill.value), nestingLvl)
                return f"Finch.choose({value})"

            case ntn.Literal(val):
                if isinstance(val, AbstractFill | AbstractFillFType):
                    # str() would silently emit broken source.
                    raise DynamicFillError(
                        "cannot emit a wrapped fill as a Julia literal"
                    )
                return _julia_literal(val)

            case ntn.Variable(name, _):
                if name in self.constants:
                    return self.generate_julia(self.constants[name], nestingLvl)
                # finch uses '#' in generated names; not valid Julia syntax.
                emitted = self.emit_name(name)
                self.used_extents.add(emitted)
                return emitted

            case _:
                # Dimension, Stack, Value are deliberately unimplemented.
                raise Exception(f"Unhandled node type: {type(prgm)}")


def handle_fills(func: ntn.Function) -> tuple[ntn.Function, tuple[int, ...]]:
    """Rewrite every Dynamic fill in `func`, and every read of one, to a zero of
    its dtype, and report which argument positions carried a Dynamic fill. This
    is a necessary but potentially unsound rewrite which should be removed
    eventually.
    """
    dynamic_args = tuple(
        i
        for i, arg in enumerate(func.args)
        if is_dynamic(getattr(arg.type_, "fill_type", None))
    )

    def rule(node):
        match node:
            case ntn.Literal(DynamicFill() | DynamicFillFType() as fill):
                return ntn.Literal(_zero_value(fill.element_type))
            case ntn.Literal(StaticFill() | StaticFillFType() as fill):
                return ntn.Literal(fill.value)
            case ntn.FillValue(tns):
                match tns.result_type.fill_type:
                    case DynamicFillFType() as fill:
                        return ntn.Literal(_zero_value(fill.element_type))
                    case StaticFillFType() as fill:
                        return ntn.Literal(fill.value)
        return None

    return Rewrite(PostWalk(rule))(func), dynamic_args


class FinchJLCompiler(NotationCompiler):
    def __init__(self, runtime: FinchJLRuntime | None = None):
        self.runtime = DefaultFinchJLRuntime() if runtime is None else runtime

    def __call__(self, prgm: ntn.Module) -> FinchJLLibrary:
        generator = FinchJLGenerator()

        kernel_dict = {}
        for orig_func in prgm.children:
            func, dynamic_args = handle_fills(orig_func)
            generated_prgm = generator(func)
            extents = tuple(
                (pos, axis, casts) for _, pos, axis, casts in generator.extents
            )
            arg_type_strs = tuple(
                ftype_to_jl_type_str(arg.type_)
                for arg in func.args
                if arg.type_ is not None
            )
            # Flat key: source, argument types, pinned fills, and extents all
            # vary independently, so none may be folded into another.
            key = (generated_prgm, arg_type_strs, dynamic_args, extents)
            kernel = self.runtime.get_cached_kernel(key)
            if kernel is None:
                func_name = f"kernel_{uuid.uuid4().hex}"
                jl_code = generated_prgm.replace(func.name.name, func_name, 1)
                jl.seval(jl_code)
                kernel = FinchJLKernel(
                    jl_code,
                    func.name.result_type,
                    func,
                    self.runtime,
                    func_name,
                    dynamic_args,
                    extents,
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
                    extents,
                )
                self.runtime.cache_kernel(key, kernel)
            kernel_dict[func.name.name] = kernel

        return FinchJLLibrary(kernel_dict)
