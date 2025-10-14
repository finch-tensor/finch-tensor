from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

from .julia import jl
from .typing import JuliaObj

if TYPE_CHECKING:
    from .tensor import Tensor

IterObj = tuple | list | dict | Any


def _recurse(x: IterObj, /, *, f: Callable[[Any], Any]) -> IterObj:
    if isinstance(x, tuple | list):
        return type(x)(_recurse(xi, f=f) for xi in x)
    if isinstance(x, dict):
        ret = {k: _recurse(v, f=f) for k, v in x.items()}
        if type(x) is not dict:
            ret = type(x)(ret)
        return ret
    return f(x)


def _recurse_iter(x: IterObj, /) -> Iterator[Any]:
    if isinstance(x, tuple | list):
        for xi in x:
            yield from _recurse_iter(xi)
        return
    if isinstance(x, dict):
        for xi in x.values():
            yield from _recurse_iter(xi)
        return
    yield x


def _to_lazy_tensor(x: Tensor | Any, /) -> Tensor | Any:
    from .tensor import Tensor

    return x if not isinstance(x, Tensor) else lazy(x)


@dataclass
class _ArgumentIndexer:
    _idx: int = 0

    def index(self, _) -> int:
        ret = self._idx
        self._idx += 1
        return ret


def _recurse_iter_compute(x: IterObj, /, *, compute_kwargs: dict[str, Any]) -> IterObj:
    from .tensor import Tensor

    # Make a recursive iterator of indices.
    idx_obj = _recurse(x, f=_ArgumentIndexer().index)
    jl_computed = []
    py_computed = []

    # Collect lazy tensors; use placeholder
    _placeholder = object()
    for xi in _recurse_iter(x):
        if isinstance(xi, Tensor) and not xi.is_computed():
            jl_computed.append(xi._obj)
            py_computed.append(_placeholder)
        else:
            py_computed.append(xi)
    jl_len = len(jl_computed)
    # This doesn't return an iterable of arrays -- only a single array
    # for `len(jl_computed) == 1`
    jl_computed = jl.Finch.compute(*jl_computed, **compute_kwargs)
    if jl_len == 1:
        jl_computed = (jl_computed,)

    # Replace placeholders with computed tensors.
    jl_computed_iter = iter(jl_computed)
    for i in range(len(py_computed)):
        if py_computed[i] is _placeholder:
            py_computed[i] = Tensor(next(jl_computed_iter))

    # Replace recursive indices by actual computed objects
    return _recurse(idx_obj, f=lambda idx: py_computed[idx])


def compiled(opt=None, *, force_materialization=False, tag: int | None = None):
    def inner(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            from .tensor import Tensor

            args = tuple(args)
            kwargs = dict(kwargs)
            compute_at_end = force_materialization or all(
                t.is_computed()
                for t in _recurse_iter((args, kwargs))
                if isinstance(t, Tensor)
            )
            args = _recurse(args, f=_to_lazy_tensor)
            kwargs = _recurse(kwargs, f=_to_lazy_tensor)
            result = func(*args, **kwargs)
            if not compute_at_end:
                return result
            compute_kwargs = (
                {"ctx": opt.get_julia_scheduler()} if opt is not None else {}
            )
            if tag is not None:
                compute_kwargs["tag"] = tag

            return _recurse_iter_compute(result, compute_kwargs=compute_kwargs)

        return wrapper_func

    return inner


class AbstractScheduler:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def get_julia_scheduler(self) -> JuliaObj:
        pass


class GalleyScheduler(AbstractScheduler):
    def get_julia_scheduler(self) -> JuliaObj:
        return jl.Finch.galley_scheduler(verbose=self.verbose)


class DefaultScheduler(AbstractScheduler):
    def get_julia_scheduler(self) -> JuliaObj:
        return jl.Finch.default_scheduler(verbose=self.verbose)


def set_optimizer(opt: AbstractScheduler) -> None:
    jl.Finch.set_scheduler_b(opt.get_julia_scheduler())


def lazy(tensor: Tensor) -> Tensor:
    from .tensor import Tensor

    if tensor.is_computed():
        return Tensor(jl.Finch.LazyTensor(tensor._obj))
    return tensor


def compute(
    tensor: Tensor, *, opt: AbstractScheduler | None = None, tag: int = -1
) -> Tensor:
    from .tensor import Tensor

    if not tensor.is_computed():
        if opt is None:
            return Tensor(jl.Finch.compute(tensor._obj, tag=tag))
        return Tensor(
            jl.Finch.compute(
                tensor._obj,
                verbose=opt.verbose,
                ctx=opt.get_julia_scheduler(),
                tag=tag,
            )
        )
    return tensor
