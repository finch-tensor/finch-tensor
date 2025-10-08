from pathlib import Path

from .julia import jl
from .tensor import Tensor


def read(filename: Path | str) -> Tensor:
    fn = str(filename)
    julia_obj = jl.fread(fn)
    return Tensor(julia_obj)


def write(filename: Path | str, tns: Tensor) -> None:
    fn = str(filename)
    jl.fwrite(fn, tns._obj)
