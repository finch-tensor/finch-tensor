from typing import Any, Literal

import juliacall as jc

spmatrix = Any

JuliaObj = jc.AnyValue

DType = jc.AnyValue  # represents jl.DataType

Device = Literal["cpu"] | None

number = int | float | bool | complex
