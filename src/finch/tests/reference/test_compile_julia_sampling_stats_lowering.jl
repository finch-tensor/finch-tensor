# Finch kernel
eval(let
        v0 = Finch.Tensor(Finch.DenseLevel(Finch.SparseListLevel(Finch.ElementLevel(0, FinchInterop.Buffer(Int64[], Val(()))), 1, FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(Int64[], Val((1,)))), 1))
        v1 = Finch.window(Finch.randommask((1,), 0.5; seed=UInt64(14276969152011380360)), Finch.Extent(1, 1))
        v2 = Finch.window(Finch.randommask((1,), 0.5; seed=UInt64(8095878257575067585)), Finch.Extent(1, 1))
        v3 = Finch.Tensor(Finch.SparseHashLevel{Int64}(Finch.SparseHashLevel{Int64}(Finch.ElementLevel(0, FinchInterop.Buffer(Int64[], Val(()))), 1, 1, Finch.SPARSE_HASH_POS_MULTIPLIER, FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(UInt8[], Val(())), FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(Tuple{Int64,Int64,UInt8}[], Val((1, 2))), FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(Int[0])), 1, 1, Finch.SPARSE_HASH_POS_MULTIPLIER, FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(UInt8[], Val(())), FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(Tuple{Int64,Int64,UInt8}[], Val((1, 2))), FinchInterop.Buffer(Int64[], Val((1,))), FinchInterop.Buffer(Int[0])))
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
function main(v0::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, ElementLevel{0, Int64, Int64, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, ()}}}}}, v1::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}}, Finch.RandomMask{1}}, v2::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}}, Finch.RandomMask{1}}, v3::Tensor{SparseHashLevel{Int64, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{UInt8, Tuple{Int64, Tuple{}}, ()}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{Tuple{Int64, Int64, UInt8}, Tuple{Int64, Tuple{Tuple{Int64, Tuple{Int64, Tuple{}}}, Tuple{Int64, Tuple{Int64, Tuple{}}}, Tuple{Int64, Tuple{Int64, Tuple{}}}}}, (1, 2)}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, ()}, SparseHashLevel{Int64, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{UInt8, Tuple{Int64, Tuple{}}, ()}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{Tuple{Int64, Int64, UInt8}, Tuple{Int64, Tuple{Tuple{Int64, Tuple{Int64, Tuple{}}}, Tuple{Int64, Tuple{Int64, Tuple{}}}, Tuple{Int64, Tuple{Int64, Tuple{}}}}}, (1, 2)}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, (1,)}, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, ()}, ElementLevel{0, Int64, Int64, Main.FinchInterop.Buffer{Int64, Tuple{Int64, Tuple{}}, ()}}}}}, v4::Int64, v5::Int64)
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
                v3_lvl_key = v3_lvl.key
                v3_lvl_perm = v3_lvl.perm
                v3_lvl_tbl_count = v3_lvl.tbl_count
                v3_lvl_2 = v3_lvl.lvl
                v3_lvl_2_ptr = v3_lvl_2.ptr
                v3_lvl_2_tbl_ctrl = v3_lvl_2.tbl_ctrl
                v3_lvl_2_tbl = v3_lvl_2.tbl
                v3_lvl_2_key = v3_lvl_2.key
                v3_lvl_2_perm = v3_lvl_2.perm
                v3_lvl_2_tbl_count = v3_lvl_2.tbl_count
                v3_lvl_3 = v3_lvl_2.lvl
                v3_lvl_3_val = v3_lvl_3.val
                v0_lvl_2_stop == v5 || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_2_stop) != $(v5))"))
                v0_lvl_stop == v4 || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_stop) != $(v4))"))
                1 == (v1.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v1.dims[1]).start))"))
                v0_lvl_stop == (v1.dims[1]).stop || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_stop) != $((v1.dims[1]).stop))"))
                1 == (v2.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v2.dims[1]).start))"))
                v0_lvl_2_stop == (v2.dims[1]).stop || throw(DimensionMismatch("mismatched dimension limits ($(v0_lvl_2_stop) != $((v2.dims[1]).stop))"))
                Finch.sparse_hash_clear!(v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_key, v3_lvl_perm, v3_lvl.subtables, v3_lvl.pos_multiplier)
                empty!(v3_lvl_key)
                empty!(v3_lvl_perm)
                fill!(v3_lvl_tbl_count, 0)
                v3_lvl_qos_stop = 0
                v3_lvl_pending = 0
                v3_lvl_free_head = zero(Int64)
                Finch.sparse_hash_clear!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, v3_lvl_2_perm, v3_lvl_2.subtables, v3_lvl_2.pos_multiplier)
                empty!(v3_lvl_2_key)
                empty!(v3_lvl_2_perm)
                fill!(v3_lvl_2_tbl_count, 0)
                v3_lvl_2_qos_stop = 0
                v3_lvl_2_pending = 0
                v3_lvl_2_free_head = zero(Int64)
                for v10_5 = 1:v0_lvl_stop
                    v3_lvl_x = Finch.sparse_hash_hash(1, v10_5, v3_lvl.pos_multiplier)
                    v3_lvl_b = Finch.sparse_hash_hash_subtable(v3_lvl_x, v3_lvl.subtables)
                    if 2 * v3_lvl.subtables * (v3_lvl_tbl_count[v3_lvl_b] + 1) > length(v3_lvl_tbl_ctrl)
                        Finch.sparse_hash_resize!(v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_key, max(2 * length(v3_lvl_tbl_ctrl), 4 * v3_lvl.subtables), v3_lvl.subtables, v3_lvl.pos_multiplier, v3_lvl_qos_stop)
                    end
                    v3_lvl_h = Finch.sparse_hash_find(v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_key, 1, v10_5, v3_lvl_x, v3_lvl.subtables)
                    v3_lvl_s = true
                    if v3_lvl_tbl_ctrl[v3_lvl_h] == Finch.SPARSE_HASH_CTRL_EMPTY
                        v3_lvl_tbl_count[v3_lvl_b] += 1
                        if v3_lvl_free_head == 0
                            v3_lvl_qos = (Int64)(v3_lvl_qos_stop += 1)
                        else
                            v3_lvl_qos = v3_lvl_free_head
                            v3_lvl_free_head = (v3_lvl_key[v3_lvl_free_head])[1]
                        end
                        if v3_lvl_qos > length(v3_lvl_key)
                            v3_lvl_q_stop = max(2 * length(v3_lvl_key), v3_lvl_qos)
                            resize!(v3_lvl_key, v3_lvl_q_stop)
                        end
                        v3_lvl_key[v3_lvl_qos] = (1, v10_5, 0x01)
                        v3_lvl_tbl_ctrl[v3_lvl_h] = Finch.sparse_hash_hash_ctrl(v3_lvl_x)
                        v3_lvl_tbl[v3_lvl_h] = v3_lvl_qos
                    else
                        v3_lvl_qos = v3_lvl_tbl[v3_lvl_h]
                        v3_lvl_s = (v3_lvl_key[v3_lvl_qos])[3] != Finch.SPARSE_HASH_KEY_RETAINED
                        v3_lvl_s && Finch.sparse_hash_share!(v3_lvl_key, v3_lvl_qos)
                    end
                    v3_lvl_s && (v3_lvl_pending += 1)
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
                                    v3_lvl_2_x = Finch.sparse_hash_hash(v3_lvl_qos, v0_lvl_2_i, v3_lvl_2.pos_multiplier)
                                    v3_lvl_2_b = Finch.sparse_hash_hash_subtable(v3_lvl_2_x, v3_lvl_2.subtables)
                                    if 2 * v3_lvl_2.subtables * (v3_lvl_2_tbl_count[v3_lvl_2_b] + 1) > length(v3_lvl_2_tbl_ctrl)
                                        Finch.sparse_hash_resize!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, max(2 * length(v3_lvl_2_tbl_ctrl), 4 * v3_lvl_2.subtables), v3_lvl_2.subtables, v3_lvl_2.pos_multiplier, v3_lvl_2_qos_stop)
                                    end
                                    v3_lvl_2_h = Finch.sparse_hash_find(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, v3_lvl_qos, v0_lvl_2_i, v3_lvl_2_x, v3_lvl_2.subtables)
                                    v3_lvl_2_s = true
                                    if v3_lvl_2_tbl_ctrl[v3_lvl_2_h] == Finch.SPARSE_HASH_CTRL_EMPTY
                                        v3_lvl_2_tbl_count[v3_lvl_2_b] += 1
                                        if v3_lvl_2_free_head == 0
                                            v3_lvl_2_qos = (Int64)(v3_lvl_2_qos_stop += 1)
                                        else
                                            v3_lvl_2_qos = v3_lvl_2_free_head
                                            v3_lvl_2_free_head = (v3_lvl_2_key[v3_lvl_2_free_head])[1]
                                        end
                                        if v3_lvl_2_qos > length(v3_lvl_2_key)
                                            v3_lvl_2_old = length(v3_lvl_2_key) + 1
                                            v3_lvl_2_q_stop = max(2 * length(v3_lvl_2_key), v3_lvl_2_qos)
                                            resize!(v3_lvl_2_key, v3_lvl_2_q_stop)
                                            Finch.resize_if_smaller!(v3_lvl_3_val, v3_lvl_2_q_stop)
                                            Finch.fill_range!(v3_lvl_3_val, 0, v3_lvl_2_old, v3_lvl_2_q_stop)
                                        end
                                        v3_lvl_2_key[v3_lvl_2_qos] = (v3_lvl_qos, v0_lvl_2_i, 0x01)
                                        v3_lvl_2_tbl_ctrl[v3_lvl_2_h] = Finch.sparse_hash_hash_ctrl(v3_lvl_2_x)
                                        v3_lvl_2_tbl[v3_lvl_2_h] = v3_lvl_2_qos
                                    else
                                        v3_lvl_2_qos = v3_lvl_2_tbl[v3_lvl_2_h]
                                        v3_lvl_2_s = (v3_lvl_2_key[v3_lvl_2_qos])[3] != Finch.SPARSE_HASH_KEY_RETAINED
                                        v3_lvl_2_s && Finch.sparse_hash_share!(v3_lvl_2_key, v3_lvl_2_qos)
                                    end
                                    v3_lvl_2_s && (v3_lvl_2_pending += 1)
                                    v3_lvl_2_dirty = false
                                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(v0_lvl_2_i)))) < p_2
                                        v3_lvl_2_dirty = true
                                        v3_lvl_3_val[v3_lvl_2_qos] = v0_lvl_3_val_2 != 0
                                    end
                                    v3_lvl_2_dirty && (v3_lvl_dirty = true)
                                    if v3_lvl_2_s
                                        if Finch.sparse_hash_release!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, v3_lvl_2_qos, v3_lvl_2_x, v3_lvl_2.subtables, v3_lvl_2_dirty, v3_lvl_2_free_head, v3_lvl_2.pos_multiplier)
                                            v3_lvl_2_tbl_count[v3_lvl_2_b] = v3_lvl_2_tbl_count[v3_lvl_2_b] - 1
                                            v3_lvl_2_free_head = (Int64)(v3_lvl_2_qos)
                                        end
                                        v3_lvl_2_pending = v3_lvl_2_pending - 1
                                    end
                                    v0_lvl_2_q += 1
                                else
                                    phase_stop_3 = min(phase_stop, v0_lvl_2_i)
                                    if v0_lvl_2_i == phase_stop_3
                                        v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                        v3_lvl_2_x = Finch.sparse_hash_hash(v3_lvl_qos, phase_stop_3, v3_lvl_2.pos_multiplier)
                                        v3_lvl_2_b = Finch.sparse_hash_hash_subtable(v3_lvl_2_x, v3_lvl_2.subtables)
                                        if 2 * v3_lvl_2.subtables * (v3_lvl_2_tbl_count[v3_lvl_2_b] + 1) > length(v3_lvl_2_tbl_ctrl)
                                            Finch.sparse_hash_resize!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, max(2 * length(v3_lvl_2_tbl_ctrl), 4 * v3_lvl_2.subtables), v3_lvl_2.subtables, v3_lvl_2.pos_multiplier, v3_lvl_2_qos_stop)
                                        end
                                        v3_lvl_2_h = Finch.sparse_hash_find(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, v3_lvl_qos, phase_stop_3, v3_lvl_2_x, v3_lvl_2.subtables)
                                        v3_lvl_2_s = true
                                        if v3_lvl_2_tbl_ctrl[v3_lvl_2_h] == Finch.SPARSE_HASH_CTRL_EMPTY
                                            v3_lvl_2_tbl_count[v3_lvl_2_b] += 1
                                            if v3_lvl_2_free_head == 0
                                                v3_lvl_2_qos = (Int64)(v3_lvl_2_qos_stop += 1)
                                            else
                                                v3_lvl_2_qos = v3_lvl_2_free_head
                                                v3_lvl_2_free_head = (v3_lvl_2_key[v3_lvl_2_free_head])[1]
                                            end
                                            if v3_lvl_2_qos > length(v3_lvl_2_key)
                                                v3_lvl_2_old = length(v3_lvl_2_key) + 1
                                                v3_lvl_2_q_stop = max(2 * length(v3_lvl_2_key), v3_lvl_2_qos)
                                                resize!(v3_lvl_2_key, v3_lvl_2_q_stop)
                                                Finch.resize_if_smaller!(v3_lvl_3_val, v3_lvl_2_q_stop)
                                                Finch.fill_range!(v3_lvl_3_val, 0, v3_lvl_2_old, v3_lvl_2_q_stop)
                                            end
                                            v3_lvl_2_key[v3_lvl_2_qos] = (v3_lvl_qos, phase_stop_3, 0x01)
                                            v3_lvl_2_tbl_ctrl[v3_lvl_2_h] = Finch.sparse_hash_hash_ctrl(v3_lvl_2_x)
                                            v3_lvl_2_tbl[v3_lvl_2_h] = v3_lvl_2_qos
                                        else
                                            v3_lvl_2_qos = v3_lvl_2_tbl[v3_lvl_2_h]
                                            v3_lvl_2_s = (v3_lvl_2_key[v3_lvl_2_qos])[3] != Finch.SPARSE_HASH_KEY_RETAINED
                                            v3_lvl_2_s && Finch.sparse_hash_share!(v3_lvl_2_key, v3_lvl_2_qos)
                                        end
                                        v3_lvl_2_s && (v3_lvl_2_pending += 1)
                                        v3_lvl_2_dirty = false
                                        if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(phase_stop_3)))) < p_2
                                            v3_lvl_2_dirty = true
                                            v3_lvl_3_val[v3_lvl_2_qos] = v0_lvl_3_val_2 != 0
                                        end
                                        v3_lvl_2_dirty && (v3_lvl_dirty = true)
                                        if v3_lvl_2_s
                                            if Finch.sparse_hash_release!(v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, v3_lvl_2_qos, v3_lvl_2_x, v3_lvl_2.subtables, v3_lvl_2_dirty, v3_lvl_2_free_head, v3_lvl_2.pos_multiplier)
                                                v3_lvl_2_tbl_count[v3_lvl_2_b] = v3_lvl_2_tbl_count[v3_lvl_2_b] - 1
                                                v3_lvl_2_free_head = (Int64)(v3_lvl_2_qos)
                                            end
                                            v3_lvl_2_pending = v3_lvl_2_pending - 1
                                        end
                                        v0_lvl_2_q += 1
                                    end
                                    break
                                end
                            end
                        end
                    end
                    v3_lvl_dirty && true
                    if v3_lvl_s
                        if Finch.sparse_hash_release!(v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_key, v3_lvl_qos, v3_lvl_x, v3_lvl.subtables, v3_lvl_dirty, v3_lvl_free_head, v3_lvl.pos_multiplier)
                            v3_lvl_tbl_count[v3_lvl_b] = v3_lvl_tbl_count[v3_lvl_b] - 1
                            v3_lvl_free_head = (Int64)(v3_lvl_qos)
                        end
                        v3_lvl_pending = v3_lvl_pending - 1
                    end
                end
                v3_lvl_pending == 0 || error("SparseHash has unfinished writers during freeze")
                qos_stop = Finch.sparse_hash_freeze!(v3_lvl_ptr, v3_lvl_perm, v3_lvl_key, 1, v3_lvl_qos_stop)
                Finch.sparse_hash_count_buckets!(v3_lvl_tbl_count, v3_lvl_perm, v3_lvl_key, v3_lvl.subtables, v3_lvl.pos_multiplier)
                v3_lvl_2_pending == 0 || error("SparseHash has unfinished writers during freeze")
                qos_stop_2 = Finch.sparse_hash_freeze!(v3_lvl_2_ptr, v3_lvl_2_perm, v3_lvl_2_key, qos_stop, v3_lvl_2_qos_stop)
                Finch.sparse_hash_count_buckets!(v3_lvl_2_tbl_count, v3_lvl_2_perm, v3_lvl_2_key, v3_lvl_2.subtables, v3_lvl_2.pos_multiplier)
                Finch.resize_if_larger!(v3_lvl_3_val, qos_stop_2)
                (v3 = Tensor((SparseHashLevel){Int64}((SparseHashLevel){Int64}(ElementLevel{0, Int64, Int64}(v3_lvl_3_val), v0_lvl_2_stop, v3_lvl_2.subtables, v3_lvl_2.pos_multiplier, v3_lvl_2_ptr, v3_lvl_2_tbl_ctrl, v3_lvl_2_tbl, v3_lvl_2_key, v3_lvl_2_perm, v3_lvl_2_tbl_count), v0_lvl_stop, v3_lvl.subtables, v3_lvl.pos_multiplier, v3_lvl_ptr, v3_lvl_tbl_ctrl, v3_lvl_tbl, v3_lvl_key, v3_lvl_perm, v3_lvl_tbl_count)),)
            end)
end