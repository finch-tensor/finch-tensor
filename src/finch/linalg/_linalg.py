from ..julia import jl
from ..tensor import Tensor


def vector_norm(
    x: Tensor,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,
) -> Tensor:
    if axis is not None:
        raise ValueError(
            "At the moment only `None` (vector norm of a flattened array) "
            "is supported. Got: {axis}."
        )

    result = Tensor(jl.Finch.norm(x._obj, ord))
    if keepdims:
        result = result.__getitem__(tuple(None for _ in range(x.ndim)))
    return result
