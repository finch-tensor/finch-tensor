import numpy as np
import scipy.sparse as sps

import finch
from finch.autoschedule import with_default_scheduler
from finch.compile_jl import COMPILE_JULIA

source = np.array(
    [
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
    ]
)

with with_default_scheduler(COMPILE_JULIA):
    tensor = finch.asarray(sps.csc_array(source), copy=False)

print(type(tensor).__name__)
print(tensor.shape)
print(tensor.to_numpy())

np.testing.assert_array_equal(tensor.to_numpy(), source)
