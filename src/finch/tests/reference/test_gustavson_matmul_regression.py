import _operator, builtins
from numba import njit
import numpy
from numpy import int64, float64


@njit
def main(_a_0: builtins.tuple, _b_4: builtins.tuple, _w_8: Numba_BufferizedNDArray, _c_11: builtins.tuple) -> builtins.tuple:
    __a_1: builtins.tuple = _a_0
    __a_1_lvl_lvl_lvl_val_slot = __a_1[0].lvl.lvl.val
    __a_1_lvl_lvl_lvl_val_slot_arr = __a_1_lvl_lvl_lvl_val_slot[0]
    __a_1_lvl_lvl_dimension: int32 = __a_1[0].lvl.dimension
    __a_1_lvl_lvl_ptr_slot = __a_1[0].lvl.ptr
    __a_1_lvl_lvl_ptr_slot_arr = __a_1_lvl_lvl_ptr_slot[0]
    __a_1_lvl_lvl_idx_slot = __a_1[0].lvl.idx
    __a_1_lvl_lvl_idx_slot_arr = __a_1_lvl_lvl_idx_slot[0]
    __a_1_lvl_lvl_qos_fill: int32 = __a_1[0].lvl.qos_fill
    __a_1_lvl_lvl_qos_stop: int32 = __a_1[0].lvl.qos_stop
    __a_1_lvl_lvl_prev_pos: int32 = __a_1[0].lvl.prev_pos
    __a_1_lvl_lvl_frozen: builtins.bool = __a_1[0].lvl.frozen
    __a_1_lvl_dimension: int32 = __a_1[0].dimension
    __a_1_lvl_stride: int32 = __a_1[0].stride
    _a_dim_0_2: int32 = __a_1_lvl_dimension
    _a_dim_1_3: int32 = __a_1_lvl_lvl_dimension
    __b_5: builtins.tuple = _b_4
    __b_5_lvl_lvl_lvl_val_slot = __b_5[0].lvl.lvl.val
    __b_5_lvl_lvl_lvl_val_slot_arr = __b_5_lvl_lvl_lvl_val_slot[0]
    __b_5_lvl_lvl_dimension: int32 = __b_5[0].lvl.dimension
    __b_5_lvl_lvl_ptr_slot = __b_5[0].lvl.ptr
    __b_5_lvl_lvl_ptr_slot_arr = __b_5_lvl_lvl_ptr_slot[0]
    __b_5_lvl_lvl_idx_slot = __b_5[0].lvl.idx
    __b_5_lvl_lvl_idx_slot_arr = __b_5_lvl_lvl_idx_slot[0]
    __b_5_lvl_lvl_qos_fill: int32 = __b_5[0].lvl.qos_fill
    __b_5_lvl_lvl_qos_stop: int32 = __b_5[0].lvl.qos_stop
    __b_5_lvl_lvl_prev_pos: int32 = __b_5[0].lvl.prev_pos
    __b_5_lvl_lvl_frozen: builtins.bool = __b_5[0].lvl.frozen
    __b_5_lvl_dimension: int32 = __b_5[0].dimension
    __b_5_lvl_stride: int32 = __b_5[0].stride
    _b_dim_0_6: int32 = __b_5_lvl_dimension
    _b_dim_1_7: int32 = __b_5_lvl_lvl_dimension
    __w_9: Numba_BufferizedNDArray = _w_8
    __w_9_val_slot = __w_9.val
    __w_9_val_slot_arr = __w_9_val_slot[0]
    __w_9_shape_element_0: int64 = __w_9.shape[0]
    __w_9_strides_element_0: int64 = __w_9.strides[0]
    _w_dim_0_10: int64 = __w_9_shape_element_0
    __c_12: builtins.tuple = _c_11
    __c_12_lvl_lvl_lvl_val_slot = __c_12[0].lvl.lvl.val
    __c_12_lvl_lvl_lvl_val_slot_arr = __c_12_lvl_lvl_lvl_val_slot[0]
    __c_12_lvl_lvl_dimension: int32 = __c_12[0].lvl.dimension
    __c_12_lvl_lvl_ptr_slot = __c_12[0].lvl.ptr
    __c_12_lvl_lvl_ptr_slot_arr = __c_12_lvl_lvl_ptr_slot[0]
    __c_12_lvl_lvl_idx_slot = __c_12[0].lvl.idx
    __c_12_lvl_lvl_idx_slot_arr = __c_12_lvl_lvl_idx_slot[0]
    __c_12_lvl_lvl_qos_fill: int32 = __c_12[0].lvl.qos_fill
    __c_12_lvl_lvl_qos_stop: int32 = __c_12[0].lvl.qos_stop
    __c_12_lvl_lvl_prev_pos: int32 = __c_12[0].lvl.prev_pos
    __c_12_lvl_lvl_frozen: builtins.bool = __c_12[0].lvl.frozen
    __c_12_lvl_dimension: int32 = __c_12[0].dimension
    __c_12_lvl_stride: int32 = __c_12[0].stride
    _c_dim_0_13: int32 = __c_12_lvl_dimension
    _c_dim_1_14: int32 = __c_12_lvl_lvl_dimension
    _w_dim_0_18: int32 = _a_dim_0_2
    _w_dim_1_19: int64 = numpy.int64(_b_dim_1_7)
    _c_dim_0_21: int32 = _w_dim_0_18
    _c_dim_1_22: int32 = numpy.int32(_w_dim_1_19)
    __c_12[0].dimension = _c_dim_0_21
    __c_12_lvl_dimension = __c_12[0].dimension
    __c_12[0].lvl.dimension = _c_dim_1_22
    __c_12_lvl_lvl_dimension = __c_12[0].lvl.dimension
    pos_stop: int64 = (numpy.int64((1 * __c_12_lvl_dimension)) + 1)
    __c_12_lvl_lvl_ptr_slot_arr = numpy.resize(__c_12_lvl_lvl_ptr_slot_arr, pos_stop)
    for p in range(0, pos_stop):
        __c_12_lvl_lvl_ptr_slot_arr[p] = 0
    __c_12[0].lvl.qos_fill = 0
    __c_12_lvl_lvl_qos_fill = __c_12[0].lvl.qos_fill
    __c_12[0].lvl.qos_stop = 0
    __c_12_lvl_lvl_qos_stop = __c_12[0].lvl.qos_stop
    __c_12[0].lvl.prev_pos = 0
    __c_12_lvl_lvl_prev_pos = __c_12[0].lvl.prev_pos
    __c_12[0].lvl.frozen = False
    __c_12_lvl_lvl_frozen = __c_12[0].lvl.frozen
    __c_12_lvl_lvl_lvl_val_slot_arr = numpy.resize(__c_12_lvl_lvl_lvl_val_slot_arr, numpy.int64(0))
    for i in range(0, len(__c_12_lvl_lvl_lvl_val_slot_arr)):
        __c_12_lvl_lvl_lvl_val_slot_arr[i] = 0
    __c_12[0].stride = 1
    __c_12_lvl_stride = __c_12[0].stride
    for _i_15 in range(0, _a_dim_0_2):
        _i_15__pos: int32 = ((0 * __a_1_lvl_dimension) + _i_15)
        _i_15__pos_2: int32 = ((0 * __c_12_lvl_dimension) + _i_15)
        __w_9.shape = (_w_dim_1_19,)
        __w_9_shape_element_0 = __w_9.shape[0]
        __w_9.strides = (1,)
        __w_9_strides_element_0 = __w_9.strides[0]
        __w_9_val_slot_arr = numpy.resize(__w_9_val_slot_arr, (_w_dim_1_19 * 1))
        for i in range(0, len(__w_9_val_slot_arr)):
            __w_9_val_slot_arr[i] = 0
        q: int32 = __a_1_lvl_lvl_ptr_slot_arr[_i_15__pos]
        q_stop: int32 = __a_1_lvl_lvl_ptr_slot_arr[(_i_15__pos + 1)]
        i_stop: int32 = 1
        i_last: int32 = 0
        if (q < q_stop):
            i_stop = __a_1_lvl_lvl_idx_slot_arr[q]
            i_last = __a_1_lvl_lvl_idx_slot_arr[(q_stop - 1)]
        if (__a_1_lvl_lvl_idx_slot_arr[q] < 0):
            q = scansearch(__a_1_lvl_lvl_idx_slot, 0, q, (q_stop - 1))
        i_start: int32 = 0
        while (min(i_stop) < min(_a_dim_1_3, (i_last + 1))):
            if (min(i_stop) == i_stop):
                _j_16__pos: int32 = q
                for _j_16 in range(i_start, min((i_stop + 1), i_stop)):
                    for _k_17 in range(0, _b_dim_1_7):
                        _k_17__pos: int64 = (0 + (__w_9_strides_element_0 * _k_17))
                        __w_9_val_slot_arr[_k_17__pos] = (__w_9_val_slot_arr[_k_17__pos] + 0)
                for _j_16 in range(max(i_start, i_stop), (i_stop + 1)):
                    _j_16__pos_2: int32 = ((0 * __b_5_lvl_dimension) + _j_16)
                    q_2: int32 = __b_5_lvl_lvl_ptr_slot_arr[_j_16__pos_2]
                    q_stop_2: int32 = __b_5_lvl_lvl_ptr_slot_arr[(_j_16__pos_2 + 1)]
                    i_stop_2: int32 = 1
                    i_last_2: int32 = 0
                    if (q_2 < q_stop_2):
                        i_stop_2 = __b_5_lvl_lvl_idx_slot_arr[q_2]
                        i_last_2 = __b_5_lvl_lvl_idx_slot_arr[(q_stop_2 - 1)]
                    if (__b_5_lvl_lvl_idx_slot_arr[q_2] < 0):
                        q_2 = scansearch(__b_5_lvl_lvl_idx_slot, 0, q_2, (q_stop_2 - 1))
                    i_start_2: int32 = 0
                    while (min(i_stop_2) < min(_b_dim_1_7, (i_last_2 + 1))):
                        if (min(i_stop_2) == i_stop_2):
                            _k_17__pos_3: int32 = q_2
                            for _k_17 in range(i_start_2, min((i_stop_2 + 1), i_stop_2)):
                                _k_17__pos_2: int64 = (0 + (__w_9_strides_element_0 * _k_17))
                                __w_9_val_slot_arr[_k_17__pos_2] = (__w_9_val_slot_arr[_k_17__pos_2] + 0)
                            for _k_17 in range(max(i_start_2, i_stop_2), (i_stop_2 + 1)):
                                _k_17__pos_4: int64 = (0 + (__w_9_strides_element_0 * _k_17))
                                __w_9_val_slot_arr[_k_17__pos_4] = (__w_9_val_slot_arr[_k_17__pos_4] + (__a_1_lvl_lvl_lvl_val_slot_arr[numpy.int64(_j_16__pos)] * __b_5_lvl_lvl_lvl_val_slot_arr[numpy.int64(_k_17__pos_3)]))
                            i_start_2 = (i_stop_2 + 1)
                            q_2 = (q_2 + 1)
                            if (q_2 < q_stop_2):
                                i_stop_2 = __b_5_lvl_lvl_idx_slot_arr[q_2]
                            else:
                                i_stop_2 = __b_5_lvl_lvl_dimension
                    for _k_17 in range(max(0, (i_last_2 + 1)), _b_dim_1_7):
                        _k_17__pos_5: int64 = (0 + (__w_9_strides_element_0 * _k_17))
                        __w_9_val_slot_arr[_k_17__pos_5] = (__w_9_val_slot_arr[_k_17__pos_5] + 0)
                i_start = (i_stop + 1)
                q = (q + 1)
                if (q < q_stop):
                    i_stop = __a_1_lvl_lvl_idx_slot_arr[q]
                else:
                    i_stop = __a_1_lvl_lvl_dimension
        for _j_16 in range(max(0, (i_last + 1)), _a_dim_1_3):
            for _k_17 in range(0, _b_dim_1_7):
                _k_17__pos_6: int64 = (0 + (__w_9_strides_element_0 * _k_17))
                __w_9_val_slot_arr[_k_17__pos_6] = (__w_9_val_slot_arr[_k_17__pos_6] + 0)
        qos: int32 = __c_12_lvl_lvl_qos_fill
        for _k_20 in range(0, _w_dim_1_19):
            _k_20__pos: int64 = (0 + (__w_9_strides_element_0 * _k_20))
            if (qos >= __c_12_lvl_lvl_qos_stop):
                if (__c_12_lvl_lvl_qos_stop == 0):
                    __c_12[0].lvl.qos_stop = 1
                    __c_12_lvl_lvl_qos_stop = __c_12[0].lvl.qos_stop
                else:
                    __c_12[0].lvl.qos_stop = (__c_12_lvl_lvl_qos_stop + __c_12_lvl_lvl_qos_stop)
                    __c_12_lvl_lvl_qos_stop = __c_12[0].lvl.qos_stop
                __c_12_lvl_lvl_idx_slot_arr = numpy.resize(__c_12_lvl_lvl_idx_slot_arr, numpy.int64(__c_12_lvl_lvl_qos_stop))
                p_start: int64 = numpy.int64(qos)
                p_stop: int64 = numpy.int64(__c_12_lvl_lvl_qos_stop)
                if (len(__c_12_lvl_lvl_lvl_val_slot_arr) < p_stop):
                    __c_12_lvl_lvl_lvl_val_slot_arr = numpy.resize(__c_12_lvl_lvl_lvl_val_slot_arr, p_stop)
                for p_2 in range(p_start, p_stop):
                    __c_12_lvl_lvl_lvl_val_slot_arr[p_2] = 0
            dirty: builtins.bool = False
            __c_12_lvl_lvl_lvl_val_slot_arr[numpy.int64(qos)] = __w_9_val_slot_arr[_k_20__pos]
            same_x = __c_12_lvl_lvl_lvl_val_slot_arr[numpy.int64(qos)]
            if (not ((same_x == 0) or ((same_x != same_x) and (0 != 0)))):
                dirty = True
            if dirty:
                __c_12_lvl_lvl_idx_slot_arr[numpy.int64(qos)] = numpy.int32(_k_20)
                qos = (qos + 1)
                __c_12[0].lvl.prev_pos = numpy.int32((_i_15__pos_2 + 1))
                __c_12_lvl_lvl_prev_pos = __c_12[0].lvl.prev_pos
        __c_12_lvl_lvl_ptr_slot_arr[numpy.int64((_i_15__pos_2 + 1))] = (__c_12_lvl_lvl_ptr_slot_arr[numpy.int64((_i_15__pos_2 + 1))] + (qos - __c_12_lvl_lvl_qos_fill))
        __c_12[0].lvl.qos_fill = qos
        __c_12_lvl_lvl_qos_fill = __c_12[0].lvl.qos_fill
    pos_stop_2: int64 = numpy.int64((1 * __c_12_lvl_dimension))
    __c_12_lvl_lvl_ptr_slot_arr = numpy.resize(__c_12_lvl_lvl_ptr_slot_arr, (pos_stop_2 + 1))
    if (__c_12_lvl_lvl_frozen == False):
        for p_3 in range(0, pos_stop_2):
            __c_12_lvl_lvl_ptr_slot_arr[(p_3 + 1)] = (__c_12_lvl_lvl_ptr_slot_arr[(p_3 + 1)] + __c_12_lvl_lvl_ptr_slot_arr[p_3])
    qos_stop: int32 = __c_12_lvl_lvl_ptr_slot_arr[pos_stop_2]
    __c_12_lvl_lvl_idx_slot_arr = numpy.resize(__c_12_lvl_lvl_idx_slot_arr, numpy.int64(qos_stop))
    __c_12[0].lvl.qos_fill = qos_stop
    __c_12_lvl_lvl_qos_fill = __c_12[0].lvl.qos_fill
    __c_12[0].lvl.qos_stop = qos_stop
    __c_12_lvl_lvl_qos_stop = __c_12[0].lvl.qos_stop
    __c_12[0].lvl.frozen = True
    __c_12_lvl_lvl_frozen = __c_12[0].lvl.frozen
    if (len(__c_12_lvl_lvl_lvl_val_slot_arr) != numpy.int64(__c_12_lvl_lvl_qos_fill)):
        __c_12_lvl_lvl_lvl_val_slot_arr = numpy.resize(__c_12_lvl_lvl_lvl_val_slot_arr, numpy.int64(__c_12_lvl_lvl_qos_fill))
    __a_1_lvl_lvl_lvl_val_slot[0] = __a_1_lvl_lvl_lvl_val_slot_arr
    __a_1_lvl_lvl_ptr_slot[0] = __a_1_lvl_lvl_ptr_slot_arr
    __a_1_lvl_lvl_idx_slot[0] = __a_1_lvl_lvl_idx_slot_arr
    __b_5_lvl_lvl_lvl_val_slot[0] = __b_5_lvl_lvl_lvl_val_slot_arr
    __b_5_lvl_lvl_ptr_slot[0] = __b_5_lvl_lvl_ptr_slot_arr
    __b_5_lvl_lvl_idx_slot[0] = __b_5_lvl_lvl_idx_slot_arr
    __w_9_val_slot[0] = __w_9_val_slot_arr
    __c_12_lvl_lvl_lvl_val_slot[0] = __c_12_lvl_lvl_lvl_val_slot_arr
    __c_12_lvl_lvl_ptr_slot[0] = __c_12_lvl_lvl_ptr_slot_arr
    __c_12_lvl_lvl_idx_slot[0] = __c_12_lvl_lvl_idx_slot_arr
    main_return: builtins.tuple = (_c_11,)
    return main_return
