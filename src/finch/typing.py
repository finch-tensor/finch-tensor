from typing import Literal, Any

import numpy as np

import juliacall as jc

TupleOf3Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]

spmatrix = Any

JuliaObj = jc.AnyValue

DType = jc.AnyValue  # represents jl.DataType

Device = Literal["cpu"] | None
