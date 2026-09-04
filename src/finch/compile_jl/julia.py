from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

_JULIA_PACKAGES = ("Finch", "HDF5", "NPZ", "TensorMarket", "Random", "Statistics")

_jc: Any | None = None
_jl: Any | None = None


def julia_available() -> bool:
    """
    Whether the Julia backend can be used, i.e. whether both Python packages
    `init_julia` needs are installed. They are optional dependencies; see the
    `julia` extra in `pyproject.toml`.
    """
    return all(
        importlib.util.find_spec(name) is not None for name in ("juliapkg", "juliacall")
    )


JULIA_AVAILABLE = julia_available()


def _julia_exe_from_libjulia(libjulia: str) -> Path | None:
    libpath = Path(libjulia)
    exe_name = "julia.exe" if os.name == "nt" else "julia"
    for bindir in (libpath.parent, libpath.parent.parent / "bin"):
        exe = bindir / exe_name
        if exe.is_file():
            return exe
    return None


def _start_julia() -> tuple[Any, Any]:
    """
    One-time Julia startup: launches juliacall and loads the packages Finch
    needs. Only ever called once per process, by init_julia().
    """
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    import juliapkg

    libjulia = juliapkg.libjulia()
    if libjulia is not None and not os.path.exists(libjulia):
        juliapkg.resolve(force=True)
        libjulia = juliapkg.libjulia()

    julia_exe = _julia_exe_from_libjulia(libjulia)
    if julia_exe is not None:
        os.environ.setdefault("PYTHON_JULIACALL_EXE", str(julia_exe))
        os.environ.setdefault("PYTHON_JULIACALL_PROJECT", juliapkg.project())
        os.environ.setdefault("PYTHON_JULIACALL_LIB", libjulia)
        os.environ.setdefault("PYTHON_JULIACALL_BINDIR", str(julia_exe.parent))

    import juliacall
    from juliacall import Main

    # To change the version of Finch used, see pyjuliapkg and
    # `juliapkg.json` in this package.
    for pkg in _JULIA_PACKAGES:
        Main.seval(f"using {pkg}")

    Main.seval("""
    function wrap_numpy_ptr(ptr_val::Integer, len::Integer, ::Type{T}) where {T}
        unsafe_wrap(Vector{T}, Ptr{T}(UInt(ptr_val)), (len,); own=false)
    end
    """)

    return juliacall, Main


def init_julia() -> tuple[Any, Any]:
    """Starts Julia and loads required packages, exactly once per process."""
    global _jc, _jl
    if _jl is None and JULIA_AVAILABLE:
        _jc, _jl = _start_julia()
    return _jc, _jl


def get_jc() -> Any:
    return init_julia()[0]


def get_jl() -> Any:
    return init_julia()[1]


class _LazyJuliaProxy:
    """
    Delegates attribute access to a Julia handle (Main, or the juliacall
    module), deferring Julia startup until it's actually needed. Julia itself
    only ever starts once -- init_julia() is idempotent -- this class just
    avoids forcing that startup at import time.
    """

    def __init__(self, resolve: Any) -> None:
        self._resolve = resolve

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


jc = _LazyJuliaProxy(get_jc)
jl = _LazyJuliaProxy(get_jl)
