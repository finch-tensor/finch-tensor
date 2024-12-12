from pathlib import Path

from .julia import add_package, jl
from .tensor import Tensor


def _import_deps(filename: str) -> None:
    fn = filename
    if fn.endswith(".mtx") or fn.endswith(".ttx") or fn.endswith(".tns"):
        add_package(
            "TensorMarket", hash="8b7d4fe7-0b45-4d0d-9dd8-5cc9b23b4b77", version="0.2.0"
        )
        jl.seval("using TensorMarket")
    elif fn.endswith(".bspnpy"):
        add_package("NPZ", hash="15e1cf62-19b3-5cfa-8e77-841668bca605", version="0.4.3")
        jl.seval("using NPZ")
    elif fn.endswith(".bsp.h5"):
        add_package(
            "HDF5", hash="f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f", version="0.17.2"
        )
        jl.seval("using HDF5")
    else:
        raise ValueError(
            "Unsupported file extension. Supported extensions are "
            "`.mtx`, `.ttx`, `.tns`, `.bspnpy`, and `.bsp.h5`."
        )


def read(filename: Path | str) -> Tensor:
    fn = str(filename)
    _import_deps(fn)
    julia_obj = jl.fread(fn)
    return Tensor(julia_obj)


def write(filename: Path | str, tns: Tensor) -> None:
    fn = str(filename)
    _import_deps(fn)
    jl.fwrite(fn, tns._obj)
