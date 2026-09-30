function main(v0::Tensor{DenseLevel{Int64, SparseListLevel{Int64, PlusOneVector{Int64, Vector{Int64}}, PlusOneVector{Int64, Vector{Int64}}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, v1::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}, Finch.Extent{Int64, Int64}}, Finch.DiagMask}, v2::Tensor{SparseHashLevel{Int64, true, PlusOneVector{Int64, Vector{Int64}}, Vector{UInt8}, Vector{Tuple{Int64, Int64, Int64}}, Vector{Int64}, PlusOneVector{Int64, Vector{Int64}}, SparseHashLevel{Int64, true, PlusOneVector{Int64, Vector{Int64}}, Vector{UInt8}, Vector{Tuple{Int64, Int64, Int64}}, Vector{Int64}, PlusOneVector{Int64, Vector{Int64}}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}})
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
                v2_lvl_pool = v2_lvl.pool
                v2_lvl_perm = v2_lvl.perm
                v2_lvl_2 = v2_lvl.lvl
                v2_lvl_2_ptr = v2_lvl_2.ptr
                v2_lvl_2_tbl_ctrl = v2_lvl_2.tbl_ctrl
                v2_lvl_2_tbl = v2_lvl_2.tbl
                v2_lvl_2_pool = v2_lvl_2.pool
                v2_lvl_2_perm = v2_lvl_2.perm
                v2_lvl_3 = v2_lvl_2.lvl
                v2_lvl_3_val = v2_lvl_3.val
                (v1.dims[1]).start == 1 || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).start) != $(1))"))
                (v1.dims[1]).stop == v0_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).stop) != $(v0_lvl_2_stop))"))
                (v1.dims[2]).stop == v0_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[2]).stop) != $(v0_lvl_stop))"))
                1 == (v1.dims[2]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v1.dims[2]).start))"))
                empty!(v2_lvl_tbl_ctrl)
                empty!(v2_lvl_tbl)
                empty!(v2_lvl_pool)
                v2_lvl_qos_stop = 0
                resize!(v2_lvl_perm, 0)
                empty!(v2_lvl_2_tbl_ctrl)
                empty!(v2_lvl_2_tbl)
                empty!(v2_lvl_2_pool)
                v2_lvl_2_qos_stop = 0
                resize!(v2_lvl_2_perm, 0)
                for v3_5 = 1:(v1.dims[2]).stop
                    if v2_lvl_qos_stop == length(v2_lvl_perm)
                        v2_lvl_q_stop = max(length(v2_lvl_perm) << 1, v2_lvl_qos_stop + 1)
                        Finch.resize_if_smaller!(v2_lvl_perm, v2_lvl_q_stop)
                        v2_lvl_tbl_cap = Finch.sparse_hash_table_capacity(v2_lvl_q_stop, v2_lvl.subtables)
                        Finch.sparse_hash_table_resize!(v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_tbl_cap, v2_lvl.subtables)
                    end
                    v2_lvl_tbl_hash = Finch.sparse_hash_hash(1, v3_5)
                    v2_lvl_tbl_ctrl_byte = Finch.sparse_hash_hash_ctrl(v2_lvl_tbl_hash)
                    v2_lvl_tbl_n = length(v2_lvl_tbl)
                    v2_lvl_tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(v2_lvl_tbl_ctrl, v2_lvl_tbl, 1, v3_5, v2_lvl_tbl_hash, v2_lvl_tbl_ctrl_byte, v2_lvl_tbl_n, v2_lvl.subtables)
                    v2_lvl_qos = 0
                    v2_lvl_tbl_found = false
                    if v2_lvl_tbl_slot != 0 && @inbounds(v2_lvl_tbl_ctrl[v2_lvl_tbl_slot]) != Finch.SPARSE_HASH_CTRL_EMPTY
                        @inbounds v2_lvl_tbl_entry = v2_lvl_tbl[v2_lvl_tbl_slot]
                        v2_lvl_qos = Finch.sparse_hash_entry_val(v2_lvl_tbl_entry)
                        v2_lvl_tbl_found = true
                    end
                    if v2_lvl_qos == 0
                        v2_lvl_qos = v2_lvl_qos_stop + 1
                        v2_lvl_qos_stop = v2_lvl_qos
                    end
                    v2_lvl_dirty = false
                    v0_lvl_q = (1 - 1) * v0_lvl_stop + v3_5
                    v0_lvl_2_q = v0_lvl_2_ptr[v0_lvl_q]
                    v0_lvl_2_q_stop = v0_lvl_2_ptr[v0_lvl_q + 1]
                    if v0_lvl_2_q < v0_lvl_2_q_stop
                        v0_lvl_2_i1 = v0_lvl_2_idx[v0_lvl_2_q_stop - 1]
                    else
                        v0_lvl_2_i1 = 0
                    end
                    phase_start = (v1.dims[1]).start
                    phase_stop = min((v1.dims[1]).stop, v3_5, v0_lvl_2_i1)
                    if phase_stop >= phase_start
                        if phase_stop < v3_5
                        else
                            if v0_lvl_2_idx[v0_lvl_2_q] < phase_stop
                                v0_lvl_2_q = Finch.scansearch(v0_lvl_2_idx, phase_stop, v0_lvl_2_q, v0_lvl_2_q_stop - 1)
                            end
                            v0_lvl_2_i = v0_lvl_2_idx[v0_lvl_2_q]
                            phase_stop_2 = min(phase_stop, v0_lvl_2_i)
                            if v0_lvl_2_i == phase_stop_2
                                v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                if v2_lvl_2_qos_stop == length(v2_lvl_2_perm)
                                    v2_lvl_2_old = length(v2_lvl_2_perm) + 1
                                    v2_lvl_2_q_stop = max(length(v2_lvl_2_perm) << 1, v2_lvl_2_qos_stop + 1)
                                    Finch.resize_if_smaller!(v2_lvl_2_perm, v2_lvl_2_q_stop)
                                    v2_lvl_2_tbl_cap = Finch.sparse_hash_table_capacity(v2_lvl_2_q_stop, v2_lvl_2.subtables)
                                    Finch.sparse_hash_table_resize!(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_tbl_cap, v2_lvl_2.subtables)
                                    Finch.resize_if_smaller!(v2_lvl_3_val, v2_lvl_2_q_stop)
                                    Finch.fill_range!(v2_lvl_3_val, 0, v2_lvl_2_old, v2_lvl_2_q_stop)
                                end
                                v2_lvl_2_tbl_hash = Finch.sparse_hash_hash(v2_lvl_qos, phase_stop_2)
                                v2_lvl_2_tbl_ctrl_byte = Finch.sparse_hash_hash_ctrl(v2_lvl_2_tbl_hash)
                                v2_lvl_2_tbl_n = length(v2_lvl_2_tbl)
                                v2_lvl_2_tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_qos, phase_stop_2, v2_lvl_2_tbl_hash, v2_lvl_2_tbl_ctrl_byte, v2_lvl_2_tbl_n, v2_lvl_2.subtables)
                                v2_lvl_2_qos = 0
                                v2_lvl_2_tbl_found = false
                                if v2_lvl_2_tbl_slot != 0 && @inbounds(v2_lvl_2_tbl_ctrl[v2_lvl_2_tbl_slot]) != Finch.SPARSE_HASH_CTRL_EMPTY
                                    @inbounds v2_lvl_2_tbl_entry = v2_lvl_2_tbl[v2_lvl_2_tbl_slot]
                                    v2_lvl_2_qos = Finch.sparse_hash_entry_val(v2_lvl_2_tbl_entry)
                                    v2_lvl_2_tbl_found = true
                                end
                                if v2_lvl_2_qos == 0
                                    v2_lvl_2_qos = v2_lvl_2_qos_stop + 1
                                    v2_lvl_2_qos_stop = v2_lvl_2_qos
                                end
                                v2_lvl_3_val[v2_lvl_2_qos] = v0_lvl_3_val_2
                                if !v2_lvl_2_tbl_found
                                    Finch.sparse_hash_table_insert_at_slot!(v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_tbl_slot, v2_lvl_qos, phase_stop_2, v2_lvl_2_qos, v2_lvl_2_tbl_ctrl_byte)
                                end
                                v2_lvl_dirty = true
                            end
                        end
                    end
                    if v2_lvl_dirty
                        if !v2_lvl_tbl_found
                            Finch.sparse_hash_table_insert_at_slot!(v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_tbl_slot, 1, v3_5, v2_lvl_qos, v2_lvl_tbl_ctrl_byte)
                        end
                    end
                end
                resize!(v2_lvl_ptr, 1 + 1)
                v2_lvl_ptr[1] = 1
                Finch.fill_range!(v2_lvl_ptr, 0, 2, 1 + 1)
                q = 0
                qos_max = 0
                resize!(v2_lvl_perm, length(v2_lvl_tbl_ctrl))
                for h = eachindex(v2_lvl_tbl_ctrl)
                    if v2_lvl_tbl_ctrl[h] != Finch.SPARSE_HASH_CTRL_EMPTY
                        entry = v2_lvl_tbl[h]
                        p = Finch.sparse_hash_entry_pos(entry)
                        v = Finch.sparse_hash_entry_val(entry)
                        q += 1
                        v2_lvl_perm[q] = h
                        qos_max = max(qos_max, v)
                        if p < 1
                            v2_lvl_ptr[p + 2] += 1
                        end
                    end
                end
                resize!(v2_lvl_perm, q)
                for p = 2:1 + 1
                    v2_lvl_ptr[p] += v2_lvl_ptr[p - 1]
                end
                idx_tmp = Vector{Int64}(undef, q)
                @inbounds for q = eachindex(v2_lvl_perm)
                        h = v2_lvl_perm[q]
                        idx_tmp[q] = Finch.sparse_hash_entry_idx(v2_lvl_tbl[h])
                    end
                shuffler = sortperm(idx_tmp)
                @inbounds for q = eachindex(shuffler)
                        shuffler[q] = v2_lvl_perm[shuffler[q]]
                    end
                @inbounds for h = shuffler
                        p = Finch.sparse_hash_entry_pos(v2_lvl_tbl[h])
                        r = v2_lvl_ptr[p + 1]
                        v2_lvl_perm[r] = h
                        v2_lvl_ptr[p + 1] += 1
                    end
                0 == 0 || error("SparseHash pending writer stack is not empty during freeze")
                for v = v2_lvl_pool
                    qos_max = max(qos_max, v)
                end
                resize!(v2_lvl_2_ptr, qos_max + 1)
                v2_lvl_2_ptr[1] = 1
                Finch.fill_range!(v2_lvl_2_ptr, 0, 2, qos_max + 1)
                q_2 = 0
                qos_max_2 = 0
                resize!(v2_lvl_2_perm, length(v2_lvl_2_tbl_ctrl))
                for h_2 = eachindex(v2_lvl_2_tbl_ctrl)
                    if v2_lvl_2_tbl_ctrl[h_2] != Finch.SPARSE_HASH_CTRL_EMPTY
                        entry_2 = v2_lvl_2_tbl[h_2]
                        p_2 = Finch.sparse_hash_entry_pos(entry_2)
                        v_2 = Finch.sparse_hash_entry_val(entry_2)
                        q_2 += 1
                        v2_lvl_2_perm[q_2] = h_2
                        qos_max_2 = max(qos_max_2, v_2)
                        if p_2 < qos_max
                            v2_lvl_2_ptr[p_2 + 2] += 1
                        end
                    end
                end
                resize!(v2_lvl_2_perm, q_2)
                for p_2 = 2:qos_max + 1
                    v2_lvl_2_ptr[p_2] += v2_lvl_2_ptr[p_2 - 1]
                end
                idx_tmp_2 = Vector{Int64}(undef, q_2)
                @inbounds for q_2 = eachindex(v2_lvl_2_perm)
                        h_2 = v2_lvl_2_perm[q_2]
                        idx_tmp_2[q_2] = Finch.sparse_hash_entry_idx(v2_lvl_2_tbl[h_2])
                    end
                shuffler_2 = sortperm(idx_tmp_2)
                @inbounds for q_2 = eachindex(shuffler_2)
                        shuffler_2[q_2] = v2_lvl_2_perm[shuffler_2[q_2]]
                    end
                @inbounds for h_2 = shuffler_2
                        p_2 = Finch.sparse_hash_entry_pos(v2_lvl_2_tbl[h_2])
                        r_2 = v2_lvl_2_ptr[p_2 + 1]
                        v2_lvl_2_perm[r_2] = h_2
                        v2_lvl_2_ptr[p_2 + 1] += 1
                    end
                0 == 0 || error("SparseHash pending writer stack is not empty during freeze")
                for v_2 = v2_lvl_2_pool
                    qos_max_2 = max(qos_max_2, v_2)
                end
                resize!(v2_lvl_3_val, qos_max_2)
                (v2 = Tensor((SparseHashLevel){Int64, true}((SparseHashLevel){Int64, true}(ElementLevel{0, Int64, Int64}(v2_lvl_3_val), (v1.dims[1]).stop, v2_lvl_2.subtables, v2_lvl_2_ptr, v2_lvl_2_tbl_ctrl, v2_lvl_2_tbl, v2_lvl_2_pool, v2_lvl_2_perm), (v1.dims[2]).stop, v2_lvl.subtables, v2_lvl_ptr, v2_lvl_tbl_ctrl, v2_lvl_tbl, v2_lvl_pool, v2_lvl_perm)),)
            end)
end