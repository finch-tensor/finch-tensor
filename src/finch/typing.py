from typing import Any, Literal

import juliacall as jc
import numpy as np


OrderType = Literal["C", "F"] | tuple[int, ...] | None

TupleOf3Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]

JuliaObj = jc.AnyValue

DType = jc.AnyValue  # represents jl.DataType

spmatrix = Any

Device = Literal["cpu"] | None
