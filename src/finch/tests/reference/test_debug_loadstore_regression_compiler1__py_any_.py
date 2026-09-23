import _operator, builtins
from numba import njit
import numpy
from numpy import int64, float64


@njit
def finch_access(a: builtins.list, idx: int64) -> int64:
    a_ = a
    a__arr = a_[0]
    if not (((idx >= 0) & (idx < len(a__arr)))):
        raise AssertionError('Finch assertion failed: and_(ge(idx, 0), lt(idx, length(slot(a_, np_buf_t(int64)))))')
    val: int64 = a__arr[idx]
    if not (((idx >= 0) & (idx < len(a__arr)))):
        raise AssertionError('Finch assertion failed: and_(ge(idx, 0), lt(idx, length(slot(a_, np_buf_t(int64)))))')
    val2: int64 = a__arr[idx]
    return val

@njit
def finch_change(a: builtins.list, idx: int64, val: int64) -> int64:
    a_ = a
    a__arr_2 = a_[0]
    if not (((idx >= 0) & (idx < len(a__arr_2)))):
        raise AssertionError('Finch assertion failed: and_(ge(idx, 0), lt(idx, length(slot(a_, np_buf_t(int64)))))')
    a__arr_2[idx] = val
    return 0
