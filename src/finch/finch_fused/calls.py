"""
Runtime handling of calls made from `jit`-compiled functions.

Every call whose callee is not statically known to accept lazy tensors is routed
through `fused_call`, which decides at runtime how to invoke the callee:

- lazy-aware callables (finch operations, numpy ufuncs, `operator`, ...) receive
  lazy tensors directly.
- Python functions whose source can be parsed are compiled with `jit` in
  transparent mode: their arguments stay lazy and their results are returned
  without being computed, so fusion continues across the call boundary.
- Anything else is opaque: lazy arguments are computed before the call.
"""

import inspect
import logging
import operator
import types
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from finch.symbolic import PostWalk, Rewrite

from .nodes import BinaryOp, Call, Function, If, Literal, While

logger = logging.getLogger(__name__)

_LAZY_AWARE_BUILTINS = frozenset(
    {
        abs,
        dict,
        enumerate,
        getattr,
        hasattr,
        isinstance,
        len,
        list,
        slice,
        tuple,
        type,
        zip,
    }
)

# Methods of LazyTensor that need a concrete value.
_EAGER_METHODS = frozenset(
    {"item", "tolist", "to_numpy", "__bool__", "__float__", "__int__", "__index__"}
)

_UNSUPPORTED_CODE_FLAGS = (
    inspect.CO_GENERATOR
    | inspect.CO_COROUTINE
    | inspect.CO_ASYNC_GENERATOR
    | inspect.CO_ITERABLE_COROUTINE
    | inspect.CO_VARARGS
    | inspect.CO_VARKEYWORDS
)

# Compiled transparent functions, or None for functions that are opaque.
_transparent_cache: dict[types.CodeType, Callable | None] = {}


def _is_lazy_aware(fn: Any) -> bool:
    if isinstance(fn, np.ufunc):
        return True
    try:
        if fn in _LAZY_AWARE_BUILTINS:
            return True
    except TypeError:  # unhashable callable
        return False
    if fn is operator.getitem:
        from finch.interface.lazy import LazyTensor

        return hasattr(LazyTensor, "__getitem__")
    module = getattr(fn, "__module__", None) or ""
    if module in ("operator", "_operator"):
        return True
    if module == "finch" or (
        module.startswith("finch.") and not module.startswith("finch.tests")
    ):
        return True
    if module.startswith("numpy") and hasattr(fn, "_implementation"):
        # numpy functions dispatch to finch through `__array_function__` when the
        # lazy interface implements them.
        from finch.interface import lazy

        return hasattr(lazy, getattr(fn, "__name__", ""))
    return False


def _transparent_jit(func: types.FunctionType) -> Callable | None:
    code = func.__code__
    if code in _transparent_cache:
        return _transparent_cache[code]

    from .dataflow import insert_lazy_and_compute
    from .parser import fused_function_to_python_function, parse_fused_function

    compiled: Callable | None = None
    if not code.co_flags & _UNSUPPORTED_CODE_FLAGS:
        try:
            fused_fn = wrap_calls(parse_fused_function(func, closure_as_params=True))
            compiled = fused_function_to_python_function(
                insert_lazy_and_compute(fused_fn, transparent=True)
            )
        except (OSError, TypeError, SyntaxError, ValueError, NotImplementedError) as e:
            logger.debug("Treating %s as an opaque call: %s", func.__qualname__, e)
    _transparent_cache[code] = compiled
    return compiled


def positional_args(
    func: types.FunctionType, args: tuple, kwargs: dict[str, Any]
) -> tuple:
    """Flatten a call to `func` into positional arguments, applying defaults."""
    code = func.__code__
    if not kwargs and not code.co_kwonlyargcount and len(args) == code.co_argcount:
        return args
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.values())


def _materialize(args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
    from finch.interface import compute
    from finch.interface.lazy import LazyTensor

    lazies: dict[int, LazyTensor] = {}

    def collect(x):
        match x:
            case LazyTensor():
                lazies[id(x)] = x
            case tuple() | list():
                for x_i in x:
                    collect(x_i)
            case dict():
                for x_i in x.values():
                    collect(x_i)

    collect((args, kwargs))
    if not lazies:
        return args, kwargs

    # Compute all lazy arguments together so that they can share work.
    computed = dict(zip(lazies, compute(tuple(lazies.values())), strict=True))

    def rebuild(x):
        match x:
            case LazyTensor():
                return computed[id(x)]
            case tuple():
                return tuple(rebuild(x_i) for x_i in x)
            case list():
                return [rebuild(x_i) for x_i in x]
            case dict():
                return {k: rebuild(v) for k, v in x.items()}
            case x:
                return x

    return rebuild(args), rebuild(kwargs)


def fused_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from finch.interface import compute
    from finch.interface.lazy import LazyTensor

    fn = getattr(fn, "__finch_jit_wrapped__", fn)
    match fn:
        case types.MethodType(__self__=LazyTensor() as self_, __name__=name) if (
            name in _EAGER_METHODS
        ):
            return getattr(compute(self_), name)(*args, **kwargs)
        case types.FunctionType() if not _is_lazy_aware(fn):
            compiled = _transparent_jit(fn)
            if compiled is not None:
                closure = tuple(cell.cell_contents for cell in fn.__closure__ or ())
                return compiled(*closure, *positional_args(fn, args, kwargs))
        case types.MethodType(__func__=types.FunctionType() as func) if (
            not _is_lazy_aware(func)
        ):
            compiled = _transparent_jit(func)
            if compiled is not None:
                closure = tuple(cell.cell_contents for cell in func.__closure__ or ())
                return compiled(
                    *closure, *positional_args(func, (fn.__self__, *args), kwargs)
                )
        case _ if _is_lazy_aware(fn):
            return fn(*args, **kwargs)
    args, kwargs = _materialize(args, kwargs)
    return fn(*args, **kwargs)


def concrete(x: Any) -> Any:
    from finch.interface import compute
    from finch.interface.lazy import LazyTensor

    match x:
        case LazyTensor():
            return compute(x)
        case x:
            return x


_INTERNAL_FNS: tuple[Any, ...] = (fused_call, concrete)


def wrap_calls(prgm: Function) -> Function:
    """
    Route calls that are not statically lazy-aware through `fused_call`, and
    compute values whose truthiness is taken (branch conditions and the operands
    of `and`/`or`), as those may be lazy results of traced calls.
    """
    from .parser import _BOOL_OPS

    logical_ops = tuple(_BOOL_OPS.values())

    def _concrete(expr):
        return Call(Literal(concrete), (expr,))

    def _visitor(node):
        match node:
            case If(cond, then_body, else_body):
                return If(_concrete(cond), then_body, else_body)
            case While(cond, body):
                return While(_concrete(cond), body)
            case BinaryOp(left, Literal(val=op) as op_lit, right) if any(
                op is logical_op for logical_op in logical_ops
            ):
                return BinaryOp(_concrete(left), op_lit, _concrete(right))
            case Call(Literal(val=fn), _, _) if any(
                fn is internal for internal in _INTERNAL_FNS
            ) or _is_lazy_aware(fn):
                return node
            case Call(fn, args, kwargs):
                return Call(Literal(fused_call), (fn, *args), kwargs)
            case node:
                return node

    return cast(Function, Rewrite(PostWalk(_visitor))(prgm))
