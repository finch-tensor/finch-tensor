# Finch kernel
eval(let
        v0 = Finch.Tensor(Finch.DenseLevel(Finch.SparseListLevel(Finch.ElementLevel(0, Int64[]), 1, Finch.PlusOneVector(Int64[]), Finch.PlusOneVector(Int64[])), 1))
        v1 = Finch.window(Finch.randommask((1,), 0.5; seed=UInt64(14276969152011380360)), Finch.Extent(1, 1))
        v2 = Finch.window(Finch.randommask((1,), 0.5; seed=UInt64(8095878257575067585)), Finch.Extent(1, 1))
        v3 = Finch.Tensor(Finch.SparseHashLevel{Int,true}(Finch.SparseHashLevel{Int,true}(Finch.ElementLevel(0, Int64[]), 1, 1, Finch.PlusOneVector(Int64[]), UInt8[], Tuple{Int64,Int64,Int64}[], Int64[], Finch.PlusOneVector(Int64[])), 1, 1, Finch.PlusOneVector(Int64[]), UInt8[], Tuple{Int64,Int64,Int64}[], Int64[], Finch.PlusOneVector(Int64[])))
        v4 = 1
        v5 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v3,v4,v5)
        v3 .= 0
        for v10 = 1:v4
            for v11 = 1:v5
                v3[v11,v10] <<Finch.initwrite(0)>>= ((v0[v11,v10] != 0) * Int64(v1[v10]) * Int64(v2[v11]))
            end
        end
        return v3
    end
end)

# Generated Julia
function main(v0::Tensor{DenseLevel{Int64, SparseListLevel{Int64, PlusOneVector{Int64, Vector{Int64}}, PlusOneVector{Int64, Vector{Int64}}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, v1::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}}, Finch.RandomMask{1}}, v2::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}}, Finch.RandomMask{1}}, v3::Tensor{SparseHashLevel{Int64, true, PlusOneVector{Int64, Vector{Int64}}, Vector{UInt8}, Vector{Tuple{Int64, Int64, Int64}}, Vector{Int64}, PlusOneVector{Int64, Vector{Int64}}, SparseHashLevel{Int64, true, PlusOneVector{Int64, Vector{Int64}}, Vector{UInt8}, Vector{Tuple{Int64, Int64, Int64}}, Vector{Int64}, PlusOneVector{Int64, Vector{Int64}}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, v4::Int64, v5::Int64)
    @inbounds @fastmath(begin
                v0_lvl = v0.lvl
                v0_lvl_stop = v0_lvl.shape
                v0_lvl_2 = v0_lvl.lvl
                v0_lvl_2_ptr = v0_lvl_2.ptr
                v0_lvl_2_idx = v0_lvl_2.idx
                v0_lvl_2_stop = v0_lvl_2.shape
                v0_lvl_3 = v0_lvl_2.lvl
                v0_lvl_3_val = v0_lvl_3.val
                p = v1.body.p
                seed = v1.body.seed
                p_2 = v2.body.p
                seed_2 = v2.body.seed
                v3_lvl = v3.lvl
                v3_lvl_ptr = v3_lvl.ptr
                v3_lvl_tbl_ctrl = v3_lvl.tbl_ctrl
                v3_lvl_tbl = v3_lvl.tbl
                v3_lvl_pool = v3_lvl.pool
                v3_lvl_perm = v3_lvl.perm
                v3_lvl_2 = v3_lvl.lvl
                v3_lvl_2_ptr = v3_lvl_2.ptr
                v3_lvl_2_tbl_ctrl = v3_lvl_2.tbl_ctrl
                v3_lvl_2_tbl = v3_lvl_2.tbl
                v3_lvl_2_pool = v3_lvl_2.pool
                v3_lvl_2_perm = v3_lvl_2.perm
                v3_lvl_3 = v3_lvl_2.lvl
                v3_lvl_3_val = v3_lvl_3.val
                v0_lvl_2_stop == v5 || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_2_stop) != $(v5))"))
                v0_lvl_stop == v4 || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_stop) != $(v4))"))
                1 == (v1.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v1.dims[1]).start))"))
                v0_lvl_stop == (v1.dims[1]).stop || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_stop) != $((v1.dims[1]).stop))"))
                1 == (v2.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v2.dims[1]).start))"))
                v0_lvl_2_stop == (v2.dims[1]).stop || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_2_stop) != $((v2.dims[1]).stop))"))
                empty!(v3_lvl_tbl_ctrl)
                empty!(v3_lvl_tbl)
                empty!(v3_lvl_pool)
                v3_lvl_qos_stop = 0
                resize!(v3_lvl_perm, 0)
                empty!(v3_lvl_2_tbl_ctrl)
                empty!(v3_lvl_2_tbl)
                empty!(v3_lvl_2_pool)
                v3_lvl_2_qos_stop = 0
                resize!(v3_lvl_2_perm, 0)
                for v10_5 = 1:v0_lvl_stop
                    if v3_lvl_qos_stop == length(v3_lvl_perm)
                        v3_lvl_q_stop = max(length(v3_lvl_perm) << 1, v3_lvl_qos_stop + 1)
                        Finch.resize_if_smaller!(v3_lvl_perm, v3_lvl_q_stop)
                        v3_lvl_tbl_cap = Finch.sparse_hash_table_capacity(v3_lvl_q_stop, v3_lvl.subtables)
                        Finch.sparse_hash_table_resize!(v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_tbl_cap, v3_lvl.subtables)
                    end
                    v3_lvl_tbl_hash = Finch.sparse_hash_hash(1, v10_5)
                    v3_lvl_tbl_ctrl_byte = Finch.sparse_hash_hash_ctrl(v3_lvl_tbl_hash)
                    v3_lvl_tbl_n = length(v3_lvl_tbl)
                    v3_lvl_tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(v3_lvl_tbl_ctrl, v3_lvl_tbl, 1, v10_5, v3_lvl_tbl_hash, v3_lvl_tbl_ctrl_byte, v3_lvl_tbl_n, v3_lvl.subtables)
                    v3_lvl_qos = 0
                    v3_lvl_tbl_found = false
                    if v3_lvl_tbl_slot != 0 && @inbounds(v3_lvl_tbl_ctrl[v3_lvl_tbl_slot]) != Finch.SPARSE_HASH_CTRL_EMPTY
                        @inbounds v3_lvl_tbl_entry = v3_lvl_tbl[v3_lvl_tbl_slot]
                        v3_lvl_qos = Finch.sparse_hash_entry_val(v3_lvl_tbl_entry)
                        v3_lvl_tbl_found = true
                    end
                    if v3_lvl_qos == 0
                        v3_lvl_qos = v3_lvl_qos_stop + 1
                        v3_lvl_qos_stop = v3_lvl_qos
                    end
                    v3_lvl_dirty = false
                    v0_lvl_q = (1 - 1) * v0_lvl_stop + v10_5
                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed), (UInt64)(v10_5)))) < p
                        v0_lvl_2_q = v0_lvl_2_ptr[v0_lvl_q]
                        v0_lvl_2_q_stop = v0_lvl_2_ptr[v0_lvl_q + 1]
                        if v0_lvl_2_q < v0_lvl_2_q_stop
                            v0_lvl_2_i1 = v0_lvl_2_idx[v0_lvl_2_q_stop - 1]
                        else
                            v0_lvl_2_i1 = 0
                        end
                        phase_stop = min(v0_lvl_2_stop, v0_lvl_2_i1)
                        if phase_stop >= 1
                            if v0_lvl_2_idx[v0_lvl_2_q] < 1
                                v0_lvl_2_q = Finch.scansearch(v0_lvl_2_idx, 1, v0_lvl_2_q, v0_lvl_2_q_stop - 1)
                            end
                            while true
                                v0_lvl_2_i = v0_lvl_2_idx[v0_lvl_2_q]
                                if v0_lvl_2_i < phase_stop
                                    v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                    if v3_lvl_2_qos_stop == length(v3_lvl_2_perm)
                                        v3_lvl_2_old = length(v3_lvl_2_perm) + 1
                                        v3_lvl_2_q_stop = max(length(v3_lvl_2_perm) << 1, v3_lvl_2_qos_stop + 1)
                                        Finch.resize_if_smaller!(v3_lvl_2_perm, v3_lvl_2_q_stop)
                                        v3_lvl_2_tbl_cap = Finch.sparse_hash_table_capacity(v3_lvl_2_q_stop, v3_lvl_2.subtables)
                                        Finch.sparse_hash_table_resize!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_tbl_cap, v3_lvl_2.subtables)
                                        Finch.resize_if_smaller!(v3_lvl_3_val, v3_lvl_2_q_stop)
                                        Finch.fill_range!(v3_lvl_3_val, 0, v3_lvl_2_old, v3_lvl_2_q_stop)
                                    end
                                    v3_lvl_2_tbl_hash = Finch.sparse_hash_hash(v3_lvl_qos, v0_lvl_2_i)
                                    v3_lvl_2_tbl_ctrl_byte = Finch.sparse_hash_hash_ctrl(v3_lvl_2_tbl_hash)
                                    v3_lvl_2_tbl_n = length(v3_lvl_2_tbl)
                                    v3_lvl_2_tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_qos, v0_lvl_2_i, v3_lvl_2_tbl_hash, v3_lvl_2_tbl_ctrl_byte, v3_lvl_2_tbl_n, v3_lvl_2.subtables)
                                    v3_lvl_2_qos = 0
                                    v3_lvl_2_tbl_found = false
                                    if v3_lvl_2_tbl_slot != 0 && @inbounds(v3_lvl_2_tbl_ctrl[v3_lvl_2_tbl_slot]) != Finch.SPARSE_HASH_CTRL_EMPTY
                                        @inbounds v3_lvl_2_tbl_entry = v3_lvl_2_tbl[v3_lvl_2_tbl_slot]
                                        v3_lvl_2_qos = Finch.sparse_hash_entry_val(v3_lvl_2_tbl_entry)
                                        v3_lvl_2_tbl_found = true
                                    end
                                    if v3_lvl_2_qos == 0
                                        v3_lvl_2_qos = v3_lvl_2_qos_stop + 1
                                        v3_lvl_2_qos_stop = v3_lvl_2_qos
                                    end
                                    v3_lvl_2_dirty = false
                                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(v0_lvl_2_i)))) < p_2
                                        v3_lvl_2_dirty = true
                                        v3_lvl_3_val[v3_lvl_2_qos] = v0_lvl_3_val_2 != 0
                                    end
                                    if v3_lvl_2_dirty
                                        if !v3_lvl_2_tbl_found
                                            Finch.sparse_hash_table_insert_at_slot!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_tbl_slot, v3_lvl_qos, v0_lvl_2_i, v3_lvl_2_qos, v3_lvl_2_tbl_ctrl_byte)
                                        end
                                        v3_lvl_dirty = true
                                    end
                                    v0_lvl_2_q += 1
                                else
                                    phase_stop_3 = min(phase_stop, v0_lvl_2_i)
                                    if v0_lvl_2_i == phase_stop_3
                                        v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                        if v3_lvl_2_qos_stop == length(v3_lvl_2_perm)
                                            v3_lvl_2_old = length(v3_lvl_2_perm) + 1
                                            v3_lvl_2_q_stop = max(length(v3_lvl_2_perm) << 1, v3_lvl_2_qos_stop + 1)
                                            Finch.resize_if_smaller!(v3_lvl_2_perm, v3_lvl_2_q_stop)
                                            v3_lvl_2_tbl_cap = Finch.sparse_hash_table_capacity(v3_lvl_2_q_stop, v3_lvl_2.subtables)
                                            Finch.sparse_hash_table_resize!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_tbl_cap, v3_lvl_2.subtables)
                                            Finch.resize_if_smaller!(v3_lvl_3_val, v3_lvl_2_q_stop)
                                            Finch.fill_range!(v3_lvl_3_val, 0, v3_lvl_2_old, v3_lvl_2_q_stop)
                                        end
                                        v3_lvl_2_tbl_hash = Finch.sparse_hash_hash(v3_lvl_qos, phase_stop_3)
                                        v3_lvl_2_tbl_ctrl_byte = Finch.sparse_hash_hash_ctrl(v3_lvl_2_tbl_hash)
                                        v3_lvl_2_tbl_n = length(v3_lvl_2_tbl)
                                        v3_lvl_2_tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_qos, phase_stop_3, v3_lvl_2_tbl_hash, v3_lvl_2_tbl_ctrl_byte, v3_lvl_2_tbl_n, v3_lvl_2.subtables)
                                        v3_lvl_2_qos = 0
                                        v3_lvl_2_tbl_found = false
                                        if v3_lvl_2_tbl_slot != 0 && @inbounds(v3_lvl_2_tbl_ctrl[v3_lvl_2_tbl_slot]) != Finch.SPARSE_HASH_CTRL_EMPTY
                                            @inbounds v3_lvl_2_tbl_entry = v3_lvl_2_tbl[v3_lvl_2_tbl_slot]
                                            v3_lvl_2_qos = Finch.sparse_hash_entry_val(v3_lvl_2_tbl_entry)
                                            v3_lvl_2_tbl_found = true
                                        end
                                        if v3_lvl_2_qos == 0
                                            v3_lvl_2_qos = v3_lvl_2_qos_stop + 1
                                            v3_lvl_2_qos_stop = v3_lvl_2_qos
                                        end
                                        v3_lvl_2_dirty = false
                                        if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(phase_stop_3)))) < p_2
                                            v3_lvl_2_dirty = true
                                            v3_lvl_3_val[v3_lvl_2_qos] = v0_lvl_3_val_2 != 0
                                        end
                                        if v3_lvl_2_dirty
                                            if !v3_lvl_2_tbl_found
                                                Finch.sparse_hash_table_insert_at_slot!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_tbl_slot, v3_lvl_qos, phase_stop_3, v3_lvl_2_qos, v3_lvl_2_tbl_ctrl_byte)
                                            end
                                            v3_lvl_dirty = true
                                        end
                                        v0_lvl_2_q += 1
                                    end
                                    break
                                end
                            end
                        end
                    end
                    if v3_lvl_dirty
                        if !v3_lvl_tbl_found
                            Finch.sparse_hash_table_insert_at_slot!(v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_tbl_slot, 1, v10_5, v3_lvl_qos, v3_lvl_tbl_ctrl_byte)
                        end
                    end
                end
                resize!(v3_lvl_ptr, 1 + 1)
                v3_lvl_ptr[1] = 1
                Finch.fill_range!(v3_lvl_ptr, 0, 2, 1 + 1)
                q = 0
                qos_max = 0
                resize!(v3_lvl_perm, length(v3_lvl_tbl_ctrl))
                for h = eachindex(v3_lvl_tbl_ctrl)
                    if v3_lvl_tbl_ctrl[h] != Finch.SPARSE_HASH_CTRL_EMPTY
                        entry = v3_lvl_tbl[h]
                        p_5 = Finch.sparse_hash_entry_pos(entry)
                        v = Finch.sparse_hash_entry_val(entry)
                        q += 1
                        v3_lvl_perm[q] = h
                        qos_max = max(qos_max, v)
                        if p_5 < 1
                            v3_lvl_ptr[p_5 + 2] += 1
                        end
                    end
                end
                resize!(v3_lvl_perm, q)
                for p_5 = 2:1 + 1
                    v3_lvl_ptr[p_5] += v3_lvl_ptr[p_5 - 1]
                end
                idx_tmp = Vector{Int64}(undef, q)
                @inbounds for q = eachindex(v3_lvl_perm)
                        h = v3_lvl_perm[q]
                        idx_tmp[q] = Finch.sparse_hash_entry_idx(v3_lvl_tbl[h])
                    end
                shuffler = sortperm(idx_tmp)
                @inbounds for q = eachindex(shuffler)
                        shuffler[q] = v3_lvl_perm[shuffler[q]]
                    end
                @inbounds for h = shuffler
                        p_5 = Finch.sparse_hash_entry_pos(v3_lvl_tbl[h])
                        r = v3_lvl_ptr[p_5 + 1]
                        v3_lvl_perm[r] = h
                        v3_lvl_ptr[p_5 + 1] += 1
                    end
                0 == 0 || error("SparseHash pending writer stack is not empty during freeze")
                for v = v3_lvl_pool
                    qos_max = max(qos_max, v)
                end
                resize!(v3_lvl_2_ptr, qos_max + 1)
                v3_lvl_2_ptr[1] = 1
                Finch.fill_range!(v3_lvl_2_ptr, 0, 2, qos_max + 1)
                q_2 = 0
                qos_max_2 = 0
                resize!(v3_lvl_2_perm, length(v3_lvl_2_tbl_ctrl))
                for h_2 = eachindex(v3_lvl_2_tbl_ctrl)
                    if v3_lvl_2_tbl_ctrl[h_2] != Finch.SPARSE_HASH_CTRL_EMPTY
                        entry_2 = v3_lvl_2_tbl[h_2]
                        p_6 = Finch.sparse_hash_entry_pos(entry_2)
                        v_2 = Finch.sparse_hash_entry_val(entry_2)
                        q_2 += 1
                        v3_lvl_2_perm[q_2] = h_2
                        qos_max_2 = max(qos_max_2, v_2)
                        if p_6 < qos_max
                            v3_lvl_2_ptr[p_6 + 2] += 1
                        end
                    end
                end
                resize!(v3_lvl_2_perm, q_2)
                for p_6 = 2:qos_max + 1
                    v3_lvl_2_ptr[p_6] += v3_lvl_2_ptr[p_6 - 1]
                end
                idx_tmp_2 = Vector{Int64}(undef, q_2)
                @inbounds for q_2 = eachindex(v3_lvl_2_perm)
                        h_2 = v3_lvl_2_perm[q_2]
                        idx_tmp_2[q_2] = Finch.sparse_hash_entry_idx(v3_lvl_2_tbl[h_2])
                    end
                shuffler_2 = sortperm(idx_tmp_2)
                @inbounds for q_2 = eachindex(shuffler_2)
                        shuffler_2[q_2] = v3_lvl_2_perm[shuffler_2[q_2]]
                    end
                @inbounds for h_2 = shuffler_2
                        p_6 = Finch.sparse_hash_entry_pos(v3_lvl_2_tbl[h_2])
                        r_2 = v3_lvl_2_ptr[p_6 + 1]
                        v3_lvl_2_perm[r_2] = h_2
                        v3_lvl_2_ptr[p_6 + 1] += 1
                    end
                0 == 0 || error("SparseHash pending writer stack is not empty during freeze")
                for v_2 = v3_lvl_2_pool
                    qos_max_2 = max(qos_max_2, v_2)
                end
                resize!(v3_lvl_3_val, qos_max_2)
                (v3 = Tensor((SparseHashLevel){Int64, true}((SparseHashLevel){Int64, true}(ElementLevel{0, Int64, Int64}(v3_lvl_3_val), v0_lvl_2_stop, v3_lvl_2.subtables, v3_lvl_2_ptr, v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_pool, v3_lvl_2_perm), v0_lvl_stop, v3_lvl.subtables, v3_lvl_ptr, v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_pool, v3_lvl_perm)),)
            end)
end