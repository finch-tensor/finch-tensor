# Finch kernel
eval(let
        v0 = Finch.Tensor(Finch.DenseLevel(Finch.SparseListLevel(Finch.ElementLevel(0, Int64[]), 1, Finch.PlusOneVector(Int64[]), Finch.PlusOneVector(Int64[])), 1))
        v1 = Finch.window(Finch.randommask((1,), 0.5; seed=UInt64(14276969152011380360)), Finch.Extent(1, 1))
        v2 = Finch.window(Finch.randommask((1,), 0.5; seed=UInt64(8095878257575067585)), Finch.Extent(1, 1))
        v3 = Finch.Tensor(Finch.ElementLevel(0, Int64[]))
        v4 = Finch.Tensor(Finch.ElementLevel(0, Int64[]))
        v5 = Finch.Tensor(Finch.ElementLevel(0, Int64[]))
    Finch.@finch_kernel function main(v0,v1,v2,v3,v4,v5)
        v3 .= 0
        for v6 = _
            for v7 = _
                v3[] += ((v0[v7,v6] != 0) * Int64(v1[v6]) * Int64(v2[v7]))
            end
        end
        v4 .= 0
        for v8 = _
            for v9 = _
                v4[] += (((v0[v9,v8] != 0) * Int64(v1[v8]) * Int64(v2[v9])) > 0.0)
            end
        end
        v5 .= 0
        for v10 = _
            for v11 = _
                v5[] += (((v0[v11,v10] != 0) * Int64(v1[v10]) * Int64(v2[v11])) == 1.0)
            end
        end
        return v3,v4,v5
    end
end)

# Generated Julia
function main(v0::Tensor{DenseLevel{Int64, SparseListLevel{Int64, PlusOneVector{Int64, Vector{Int64}}, PlusOneVector{Int64, Vector{Int64}}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, v1::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}}, Finch.RandomMask{1}}, v2::Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}}, Finch.RandomMask{1}}, v3::Tensor{ElementLevel{0, Int64, Int64, Vector{Int64}}}, v4::Tensor{ElementLevel{0, Int64, Int64, Vector{Int64}}}, v5::Tensor{ElementLevel{0, Int64, Int64, Vector{Int64}}})
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
                v3_lvl_val = v3_lvl.val
                v4_lvl = v4.lvl
                v4_lvl_val = v4_lvl.val
                v5_lvl = v5.lvl
                v5_lvl_val = v5_lvl.val
                (v1.dims[1]).start == 1 || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).start) != $(1))"))
                (v1.dims[1]).stop == v0_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).stop) != $(v0_lvl_stop))"))
                (v2.dims[1]).stop == v0_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($((v2.dims[1]).stop) != $(v0_lvl_2_stop))"))
                1 == (v2.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v2.dims[1]).start))"))
                (v1.dims[1]).start == 1 || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).start) != $(1))"))
                (v1.dims[1]).stop == v0_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).stop) != $(v0_lvl_stop))"))
                (v2.dims[1]).stop == v0_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($((v2.dims[1]).stop) != $(v0_lvl_2_stop))"))
                1 == (v2.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v2.dims[1]).start))"))
                (v1.dims[1]).start == 1 || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).start) != $(1))"))
                (v1.dims[1]).stop == v0_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($((v1.dims[1]).stop) != $(v0_lvl_stop))"))
                (v2.dims[1]).stop == v0_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($((v2.dims[1]).stop) != $(v0_lvl_2_stop))"))
                1 == (v2.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v2.dims[1]).start))"))
                Finch.resize_if_smaller!(v3_lvl_val, 1)
                Finch.fill_range!(v3_lvl_val, 0, 1, 1)
                for v6_4 = (v1.dims[1]).start:(v1.dims[1]).stop
                    v0_lvl_q = (1 - 1) * v0_lvl_stop + v6_4
                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed), (UInt64)(v6_4)))) < p
                        v0_lvl_2_q = v0_lvl_2_ptr[v0_lvl_q]
                        v0_lvl_2_q_stop = v0_lvl_2_ptr[v0_lvl_q + 1]
                        if v0_lvl_2_q < v0_lvl_2_q_stop
                            v0_lvl_2_i1 = v0_lvl_2_idx[v0_lvl_2_q_stop - 1]
                        else
                            v0_lvl_2_i1 = 0
                        end
                        phase_stop = min((v2.dims[1]).stop, v0_lvl_2_i1)
                        if phase_stop >= 1
                            if v0_lvl_2_idx[v0_lvl_2_q] < 1
                                v0_lvl_2_q = Finch.scansearch(v0_lvl_2_idx, 1, v0_lvl_2_q, v0_lvl_2_q_stop - 1)
                            end
                            while true
                                v0_lvl_2_i = v0_lvl_2_idx[v0_lvl_2_q]
                                if v0_lvl_2_i < phase_stop
                                    v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(v0_lvl_2_i)))) < p_2
                                        v3_lvl_val[1] = (v0_lvl_3_val_2 != 0) + v3_lvl_val[1]
                                    end
                                    v0_lvl_2_q += 1
                                else
                                    phase_stop_3 = min(phase_stop, v0_lvl_2_i)
                                    if v0_lvl_2_i == phase_stop_3
                                        v0_lvl_3_val_2 = v0_lvl_3_val[v0_lvl_2_q]
                                        if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(phase_stop_3)))) < p_2
                                            v3_lvl_val[1] += v0_lvl_3_val_2 != 0
                                        end
                                        v0_lvl_2_q += 1
                                    end
                                    break
                                end
                            end
                        end
                    end
                end
                Finch.resize_if_smaller!(v4_lvl_val, 1)
                Finch.fill_range!(v4_lvl_val, 0, 1, 1)
                for v8_4 = (v1.dims[1]).start:(v1.dims[1]).stop
                    v0_lvl_q_2 = (1 - 1) * v0_lvl_stop + v8_4
                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed), (UInt64)(v8_4)))) < p
                        v0_lvl_2_q_2 = v0_lvl_2_ptr[v0_lvl_q_2]
                        v0_lvl_2_q_stop_2 = v0_lvl_2_ptr[v0_lvl_q_2 + 1]
                        if v0_lvl_2_q_2 < v0_lvl_2_q_stop_2
                            v0_lvl_2_i1_2 = v0_lvl_2_idx[v0_lvl_2_q_stop_2 - 1]
                        else
                            v0_lvl_2_i1_2 = 0
                        end
                        phase_stop_5 = min((v2.dims[1]).stop, v0_lvl_2_i1_2)
                        if phase_stop_5 >= 1
                            if v0_lvl_2_idx[v0_lvl_2_q_2] < 1
                                v0_lvl_2_q_2 = Finch.scansearch(v0_lvl_2_idx, 1, v0_lvl_2_q_2, v0_lvl_2_q_stop_2 - 1)
                            end
                            while true
                                v0_lvl_2_i_2 = v0_lvl_2_idx[v0_lvl_2_q_2]
                                if v0_lvl_2_i_2 < phase_stop_5
                                    v0_lvl_3_val_3 = v0_lvl_3_val[v0_lvl_2_q_2]
                                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(v0_lvl_2_i_2)))) < p_2
                                        v4_lvl_val[1] = (0.0 < (v0_lvl_3_val_3 != 0)) + v4_lvl_val[1]
                                    end
                                    v0_lvl_2_q_2 += 1
                                else
                                    phase_stop_7 = min(phase_stop_5, v0_lvl_2_i_2)
                                    if v0_lvl_2_i_2 == phase_stop_7
                                        v0_lvl_3_val_3 = v0_lvl_3_val[v0_lvl_2_q_2]
                                        if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(phase_stop_7)))) < p_2
                                            v4_lvl_val[1] += 0.0 < (v0_lvl_3_val_3 != 0)
                                        end
                                        v0_lvl_2_q_2 += 1
                                    end
                                    break
                                end
                            end
                        end
                    end
                end
                Finch.resize_if_smaller!(v5_lvl_val, 1)
                Finch.fill_range!(v5_lvl_val, 0, 1, 1)
                for v10_4 = (v1.dims[1]).start:(v1.dims[1]).stop
                    v0_lvl_q_3 = (1 - 1) * v0_lvl_stop + v10_4
                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed), (UInt64)(v10_4)))) < p
                        v0_lvl_2_q_3 = v0_lvl_2_ptr[v0_lvl_q_3]
                        v0_lvl_2_q_stop_3 = v0_lvl_2_ptr[v0_lvl_q_3 + 1]
                        if v0_lvl_2_q_3 < v0_lvl_2_q_stop_3
                            v0_lvl_2_i1_3 = v0_lvl_2_idx[v0_lvl_2_q_stop_3 - 1]
                        else
                            v0_lvl_2_i1_3 = 0
                        end
                        phase_stop_9 = min((v2.dims[1]).stop, v0_lvl_2_i1_3)
                        if phase_stop_9 >= 1
                            if v0_lvl_2_idx[v0_lvl_2_q_3] < 1
                                v0_lvl_2_q_3 = Finch.scansearch(v0_lvl_2_idx, 1, v0_lvl_2_q_3, v0_lvl_2_q_stop_3 - 1)
                            end
                            while true
                                v0_lvl_2_i_3 = v0_lvl_2_idx[v0_lvl_2_q_3]
                                if v0_lvl_2_i_3 < phase_stop_9
                                    v0_lvl_3_val_4 = v0_lvl_3_val[v0_lvl_2_q_3]
                                    if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(v0_lvl_2_i_3)))) < p_2
                                        v5_lvl_val[1] = ((v0_lvl_3_val_4 != 0) == 1.0) + v5_lvl_val[1]
                                    end
                                    v0_lvl_2_q_3 += 1
                                else
                                    phase_stop_11 = min(phase_stop_9, v0_lvl_2_i_3)
                                    if v0_lvl_2_i_3 == phase_stop_11
                                        v0_lvl_3_val_4 = v0_lvl_3_val[v0_lvl_2_q_3]
                                        if (Finch).randommask_uniform((Finch).randommask_mix(xor((Finch).randommask_mix(seed_2), (UInt64)(phase_stop_11)))) < p_2
                                            v5_lvl_val[1] += (v0_lvl_3_val_4 != 0) == 1.0
                                        end
                                        v0_lvl_2_q_3 += 1
                                    end
                                    break
                                end
                            end
                        end
                    end
                end
                resize!(v5_lvl_val, 1)
                resize!(v4_lvl_val, 1)
                resize!(v3_lvl_val, 1)
                (v3 = Tensor(ElementLevel{0, Int64, Int64}(v3_lvl_val)), v4 = Tensor(ElementLevel{0, Int64, Int64}(v4_lvl_val)), v5 = Tensor(ElementLevel{0, Int64, Int64}(v5_lvl_val)))
            end)
end