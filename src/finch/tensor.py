from finchlite import EagerTensor

from .julia import jl
from .typing import JuliaObj

# Singleton classes for levels types
# finch tensor lite, formatter stage
# level ftype without the need to create tthe object
# https://github.com/finch-tensor/finch-tensor-lite/blob/main/src/finchlite/autoschedule/formatter.py


class _Display:
    _obj: JuliaObj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)


class FinchJLTensor(_Display, EagerTensor):
    def __init__(self, obj: jl.Finch.Tensor):
        if isinstance(obj, jl.Finch.Tensor):
            self._obj = obj
        else:
            raise ValueError(f"Raw julia object expected. Found: {type(obj)}")

    # TODO: figure out a way to walk through the levels and return the ftype
    @property
    def ftype(self):
        """Returns the ftype of the buffer"""

    @property
    def shape(self) -> tuple:
        """Shape of the tensor."""
        return self.obj.shape
