function main(v0::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, ElementLevel{0, Int64, Int64, Main.FinchInterop.Buffer{Int64}}}}}, v1::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}, Finch.Extent{Int64, Int64}}, Finch.DiagMask}, v2::Tensor{SparseHashLevel{Int64, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.Buffer{UInt8}, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.PlusOneBuffer{Tuple{Int64, Int64, UInt8}, (1, 2), Main.FinchInterop.Buffer{Tuple{Int64, Int64, UInt8}}}, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.Buffer{Int64}, SparseHashLevel{Int64, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.Buffer{UInt8}, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.PlusOneBuffer{Tuple{Int64, Int64, UInt8}, (1, 2), Main.FinchInterop.Buffer{Tuple{Int64, Int64, UInt8}}}, Main.FinchInterop.PlusOneBuffer{Int64, (1,), Main.FinchInterop.Buffer{Int64}}, Main.FinchInterop.Buffer{Int64}, ElementLevel{0, Int64, Int64, Main.FinchInterop.Buffer{Int64}}}}}, v5::Int64, v6::Int64)
    @inbounds @fastmath(begin
                v0_lvl = v0.lvl
                v0_lvl_stop = v0_lvl.shape
                v0_lvl_2 = v0_lvl.lvl
                v0_lvl_2_ptr = v0_lvl_2.ptr
                v0_lvl_2_idx = v0_lvl_2.idx
                v0_lvl_2_stop = v0_lvl_2.shape
                v0_lvl_3 = v0_lvl_2.lvl
                v0_lvl_3_val = v0_lvl_3.val
                v2_lvl = v2.lvl
                v2_lvl_ptr = v2_lvl.ptr
                v2_lvl_tbl_ctrl = v2_lvl.tbl_ctrl
                v2_lvl_tbl = v2_lvl.tbl
                v2_lvl_key = v2_lvl.key
                v2_lvl_perm = v2_lvl.perm
                v2_lvl_tbl_count = v2_lvl.tbl_count
                v2_lvl_2 = v2_lvl.lvl
                v2_lvl_2_ptr = v2_lvl_2.ptr
                v2_lvl_2_tbl_ctrl = v2_lvl_2.tbl_ctrl
                v2_lvl_2_tbl = v2_lvl_2.tbl
                v2_lvl_2_key = v2_lvl_2.key
                v2_lvl_2_perm = v2_lvl_2.perm
                v2_lvl_2_tbl_count = v2_lvl_2.tbl_count
                v2_lvl_3 = v2_lvl_2.lvl
                v2_lvl_3_val = v2_lvl_3.val
                v0_lvl_2_stop == v6 || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_2_stop) != $(v6))"))
                v0_lvl_stop == v5 || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_stop) != $(v5))"))
                1 == (v1.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v1.dims[1]).start))"))
                v0_lvl_2_stop == (v1.dims[1]).stop || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_2_stop) != $((v1.dims[1]).stop))"))
                1 == (v1.dims[2]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v1.dims[2]).start))"))
                v0_lvl_stop == (v1.dims[2]).stop || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_stop) != $((v1.dims[2]).stop))"))
                Finch.sparse_hash_clear!(v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_key, v2_lvl_perm, v2_lvl.subtables, v2_lvl.pos_multiplier)
                empty!(v2_lvl_key)
                empty!(v2_lvl_perm)
                fill!(v2_lvl_tbl_count, 0)
                v2_lvl_qos_stop = 0
                v2_lvl_pending = 0
                v2_lvl_free_head = zero(Int64)
                Finch.sparse_hash_clear!(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_key, v2_lvl_2_perm, v2_lvl_2.subtables, v2_lvl_2.pos_multiplier)
                empty!(v2_lvl_2_key)
                empty!(v2_lvl_2_perm)
                fill!(v2_lvl_2_tbl_count, 0)
                v2_lvl_2_qos_stop = 0
                v2_lvl_2_pending = 0
                v2_lvl_2_free_head = zero(Int64)
                for v9_5 = 1:v0_lvl_stop
                    v2_lvl_x = Finch.sparse_hash_hash(1, v9_5, v2_lvl.pos_multiplier)
                    v2_lvl_b = Finch.sparse_hash_hash_subtable(v2_lvl_x, v2_lvl.subtables)
                    if 2 * v2_lvl.subtables * (v2_lvl_tbl_count[v2_lvl_b] + 1) > length(v2_lvl_tbl_ctrl)
                        Finch.sparse_hash_resize!(v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_key, max(2 * length(v2_lvl_tbl_ctrl), 4 * v2_lvl.subtables), v2_lvl.subtables, v2_lvl.pos_multiplier, v2_lvl_qos_stop)
                    end
                    v2_lvl_h = Finch.sparse_hash_find(v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_key, 1, v9_5, v2_lvl_x, v2_lvl.subtables)
                    v2_lvl_s = true
                    if v2_lvl_tbl_ctrl[v2_lvl_h] == Finch.SPARSE_HASH_CTRL_EMPTY
                        v2_lvl_tbl_count[v2_lvl_b] += 1
                        if v2_lvl_free_head == 0
                            v2_lvl_qos = (Int64)(v2_lvl_qos_stop += 1)
                        else
                            v2_lvl_qos = v2_lvl_free_head
                            v2_lvl_free_head = (v2_lvl_key[v2_lvl_free_head])[1]
                        end
                        if v2_lvl_qos > length(v2_lvl_key)
                            v2_lvl_q_stop = max(2 * length(v2_lvl_key), v2_lvl_qos)
                            resize!(v2_lvl_key, v2_lvl_q_stop)
                        end
                        v2_lvl_key[v2_lvl_qos] = (1, v9_5, 0x01)
                        v2_lvl_tbl_ctrl[v2_lvl_h] = Finch.sparse_hash_hash_ctrl(v2_lvl_x)
                        v2_lvl_tbl[v2_lvl_h] = v2_lvl_qos
                    else
                        v2_lvl_qos = v2_lvl_tbl[v2_lvl_h]
                        v2_lvl_s = (v2_lvl_key[v2_lvl_qos])[3] != Finch.SPARSE_HASH_KEY_RETAINED
                        v2_lvl_s && Finch.sparse_hash_share!(v2_lvl_key, v2_lvl_qos)
                    end
                    v2_lvl_s && (v2_lvl_pending += 1)
                    v2_lvl_dirty = false
                    v0_lvl_q = (1 - 1) * v0_lvl_stop + v9_5
                    v0_lvl_2_q = v0_lvl_2_ptr[v0_lvl_q]
                    v0_lvl_2_q_stop = v0_lvl_2_ptr[v0_lvl_q + 1]
                    if v0_lvl_2_q < v0_lvl_2_q_stop
                        v0_lvl_2_i1 = v0_lvl_2_idx[v0_lvl_2_q_stop - 1]
                    else
                        v0_lvl_2_i1 = 0
                    end
                    phase_stop = min(v0_lvl_2_stop, v9_5, v0_lvl_2_i1)
                    if phase_stop >= 1
                        if phase_stop < v9_5
                        else
                            if v0_lvl_2_idx[v0_lvl_2_q] < phase_stop
                                v0_lvl_2_q = Finch.scansearch(v0_lvl_2_idx, phase_stop, v0_lvl_2_q, v0_lvl_2_q_stop - 1)
                            end
                            v0_lvl_2_i = v0_lvl_2_idx[v0_lvl_2_q]
                            phase_stop_2 = min(phase_stop, v0_lvl_2_i)
                            if v0_lvl_2_i == phase_stop_2
                                v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                v2_lvl_2_x = Finch.sparse_hash_hash(v2_lvl_qos, phase_stop_2, v2_lvl_2.pos_multiplier)
                                v2_lvl_2_b = Finch.sparse_hash_hash_subtable(v2_lvl_2_x, v2_lvl_2.subtables)
                                if 2 * v2_lvl_2.subtables * (v2_lvl_2_tbl_count[v2_lvl_2_b] + 1) > length(v2_lvl_2_tbl_ctrl)
                                    Finch.sparse_hash_resize!(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_key, max(2 * length(v2_lvl_2_tbl_ctrl), 4 * v2_lvl_2.subtables), v2_lvl_2.subtables, v2_lvl_2.pos_multiplier, v2_lvl_2_qos_stop)
                                end
                                v2_lvl_2_h = Finch.sparse_hash_find(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_key, v2_lvl_qos, phase_stop_2, v2_lvl_2_x, v2_lvl_2.subtables)
                                v2_lvl_2_s = true
                                if v2_lvl_2_tbl_ctrl[v2_lvl_2_h] == Finch.SPARSE_HASH_CTRL_EMPTY
                                    v2_lvl_2_tbl_count[v2_lvl_2_b] += 1
                                    if v2_lvl_2_free_head == 0
                                        v2_lvl_2_qos = (Int64)(v2_lvl_2_qos_stop += 1)
                                    else
                                        v2_lvl_2_qos = v2_lvl_2_free_head
                                        v2_lvl_2_free_head = (v2_lvl_2_key[v2_lvl_2_free_head])[1]
                                    end
                                    if v2_lvl_2_qos > length(v2_lvl_2_key)
                                        v2_lvl_2_old = length(v2_lvl_2_key) + 1
                                        v2_lvl_2_q_stop = max(2 * length(v2_lvl_2_key), v2_lvl_2_qos)
                                        resize!(v2_lvl_2_key, v2_lvl_2_q_stop)
                                        Finch.resize_if_smaller!(v2_lvl_3_val, v2_lvl_2_q_stop)
                                        Finch.fill_range!(v2_lvl_3_val, 0, v2_lvl_2_old, v2_lvl_2_q_stop)
                                    end
                                    v2_lvl_2_key[v2_lvl_2_qos] = (v2_lvl_qos, phase_stop_2, 0x01)
                                    v2_lvl_2_tbl_ctrl[v2_lvl_2_h] = Finch.sparse_hash_hash_ctrl(v2_lvl_2_x)
                                    v2_lvl_2_tbl[v2_lvl_2_h] = v2_lvl_2_qos
                                else
                                    v2_lvl_2_qos = v2_lvl_2_tbl[v2_lvl_2_h]
                                    v2_lvl_2_s = (v2_lvl_2_key[v2_lvl_2_qos])[3] != Finch.SPARSE_HASH_KEY_RETAINED
                                    v2_lvl_2_s && Finch.sparse_hash_share!(v2_lvl_2_key, v2_lvl_2_qos)
                                end
                                v2_lvl_2_s && (v2_lvl_2_pending += 1)
                                v2_lvl_3_val[v2_lvl_2_qos] = v0_lvl_3_val_2
                                true && (v2_lvl_dirty = true)
                                if v2_lvl_2_s
                                    if Finch.sparse_hash_release!(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_key, v2_lvl_2_qos, v2_lvl_2_x, v2_lvl_2.subtables, true, v2_lvl_2_free_head, v2_lvl_2.pos_multiplier)
                                        v2_lvl_2_tbl_count[v2_lvl_2_b] = v2_lvl_2_tbl_count[v2_lvl_2_b] - 1
                                        v2_lvl_2_free_head = (Int64)(v2_lvl_2_qos)
                                    end
                                    v2_lvl_2_pending = v2_lvl_2_pending - 1
                                end
                            end
                        end
                    end
                    v2_lvl_dirty && true
                    if v2_lvl_s
                        if Finch.sparse_hash_release!(v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_key, v2_lvl_qos, v2_lvl_x, v2_lvl.subtables, v2_lvl_dirty, v2_lvl_free_head, v2_lvl.pos_multiplier)
                            v2_lvl_tbl_count[v2_lvl_b] = v2_lvl_tbl_count[v2_lvl_b] - 1
                            v2_lvl_free_head = (Int64)(v2_lvl_qos)
                        end
                        v2_lvl_pending = v2_lvl_pending - 1
                    end
                end
                v2_lvl_pending == 0 || error("SparseHash has unfinished writers during freeze")
                qos_stop = Finch.sparse_hash_freeze!(v2_lvl_ptr, v2_lvl_perm, v2_lvl_key, 1, v2_lvl_qos_stop)
                Finch.sparse_hash_count_buckets!(v2_lvl_tbl_count, v2_lvl_perm, v2_lvl_key, v2_lvl.subtables, v2_lvl.pos_multiplier)
                v2_lvl_2_pending == 0 || error("SparseHash has unfinished writers during freeze")
                qos_stop_2 = Finch.sparse_hash_freeze!(v2_lvl_2_ptr, v2_lvl_2_perm, v2_lvl_2_key, qos_stop, v2_lvl_2_qos_stop)
                Finch.sparse_hash_count_buckets!(v2_lvl_2_tbl_count, v2_lvl_2_perm, v2_lvl_2_key, v2_lvl_2.subtables, v2_lvl_2.pos_multiplier)
                Finch.resize_if_larger!(v2_lvl_3_val, qos_stop_2)
                (v2 = Tensor((SparseHashLevel){Int64}((SparseHashLevel){Int64}(ElementLevel{0, Int64, Int64}(v2_lvl_3_val), v0_lvl_2_stop, v2_lvl_2.subtables, v2_lvl_2.pos_multiplier, v2_lvl_2_ptr, v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_key, v2_lvl_2_perm, v2_lvl_2_tbl_count), v0_lvl_stop, v2_lvl.subtables, v2_lvl.pos_multiplier, v2_lvl_ptr, v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_key, v2_lvl_perm, v2_lvl_tbl_count)),)
            end)
end