# Finch kernel
eval(let
        v0 = Finch.Tensor(Finch.DenseLevel(Finch.SparseListLevel(Finch.ElementLevel(0, Int64[]), 1, Finch.PlusOneVector(Int64[]), Finch.PlusOneVector(Int64[])), 1))
        v1 = Finch.swizzle(Finch.window(Finch.splitmask(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v2 = Finch.swizzle(Finch.window(Finch.splitmask(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v3 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v4 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
        v5 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
    Finch.@finch_kernel function main(v0,v1,v2,v3,v4,v5)
        v3 .= 0
        for v6 = _
            for v7 = _
                for v8 = _
                    for v9 = _
                        v3[v8,v6] += ((v0[v9,v7] != 0) * Int64(v1[v6,v7]) * Int64(v2[v8,v9]))
                    end
                end
            end
        end
        v4 .= 0
        for v10 = _
            for v11 = _
                v4[v10] += Int64(v1[v10,v11])
            end
        end
        v5 .= 0
        for v12 = _
            for v13 = _
                v5[v12] += Int64(v2[v12,v13])
            end
        end
        return v3,v4,v5
    end
end)

# Generated Julia
function main(v0::Tensor{DenseLevel{Int64, SparseListLevel{Int64, PlusOneVector{Int64, Vector{Int64}}, PlusOneVector{Int64, Vector{Int64}}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, v1::Finch.SwizzleArray{(2, 1), Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}, Finch.Extent{Int64, Int64}}, Finch.SplitMask{Int64}}}, v2::Finch.SwizzleArray{(2, 1), Finch.WindowedArray{Tuple{Finch.Extent{Int64, Int64}, Finch.Extent{Int64, Int64}}, Finch.SplitMask{Int64}}}, v3::Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, v4::Tensor{DenseLevel{Int64, ElementLevel{0, Int64, Int64, Vector{Int64}}}}, v5::Tensor{DenseLevel{Int64, ElementLevel{0, Int64, Int64, Vector{Int64}}}})
    @inbounds @fastmath(begin
                v0_lvl = v0.lvl
                v0_lvl_stop = v0_lvl.shape
                v0_lvl_2 = v0_lvl.lvl
                v0_lvl_2_ptr = v0_lvl_2.ptr
                v0_lvl_2_idx = v0_lvl_2.idx
                v0_lvl_2_stop = v0_lvl_2.shape
                v0_lvl_3 = v0_lvl_2.lvl
                v0_lvl_3_val = v0_lvl_3.val
                P = v1.body.body.P
                stop = v1.body.body.stop
                P_2 = v2.body.body.P
                stop_2 = v2.body.body.stop
                v3_lvl = v3.lvl
                v3_lvl_2 = v3_lvl.lvl
                v3_lvl_3 = v3_lvl_2.lvl
                v3_lvl_3_val = v3_lvl_3.val
                v4_lvl = v4.lvl
                v4_lvl_2 = v4_lvl.lvl
                v4_lvl_2_val = v4_lvl_2.val
                v5_lvl = v5.lvl
                v5_lvl_2 = v5_lvl.lvl
                v5_lvl_2_val = v5_lvl_2.val
                (v1.body.dims[1]).start == 1 || throw(DimensionMismatch("mismatched dimension limits ($((v1.body.dims[1]).start) != $(1))"))
                (v1.body.dims[1]).stop == v0_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($((v1.body.dims[1]).stop) != $(v0_lvl_stop))"))
                (v2.body.dims[1]).stop == v0_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($((v2.body.dims[1]).stop) != $(v0_lvl_2_stop))"))
                1 == (v2.body.dims[1]).start || throw(DimensionMismatch("mismatched dimension limits ($(1) != $((v2.body.dims[1]).start))"))
                pos_stop = (v2.body.dims[2]).stop * (v1.body.dims[2]).stop
                Finch.resize_if_smaller!(v3_lvl_3_val, pos_stop)
                Finch.fill_range!(v3_lvl_3_val, 0, 1, pos_stop)
                for v6_4 = (v1.body.dims[2]).start:(v1.body.dims[2]).stop
                    v3_lvl_q = (1 - 1) * (v1.body.dims[2]).stop + v6_4
                    phase_start_2 = max((v1.body.dims[1]).start, 1 + fld(stop * (-1 + v6_4), P))
                    phase_stop_2 = min((v1.body.dims[1]).stop, fld(stop * v6_4, P))
                    if phase_stop_2 >= phase_start_2
                        for v7_6 = phase_start_2:phase_stop_2
                            v0_lvl_q = (1 - 1) * v0_lvl_stop + v7_6
                            for v8_4 = (v2.body.dims[2]).start:(v2.body.dims[2]).stop
                                v3_lvl_2_q = (v3_lvl_q - 1) * (v2.body.dims[2]).stop + v8_4
                                v0_lvl_2_q = v0_lvl_2_ptr[v0_lvl_q]
                                v0_lvl_2_q_stop = v0_lvl_2_ptr[v0_lvl_q + 1]
                                if v0_lvl_2_q < v0_lvl_2_q_stop
                                    v0_lvl_2_i1 = v0_lvl_2_idx[v0_lvl_2_q_stop - 1]
                                else
                                    v0_lvl_2_i1 = 0
                                end
                                phase_start_5 = max(1, 1 + fld(stop_2 * (-1 + v8_4), P_2))
                                phase_stop_5 = min((v2.body.dims[1]).stop, v0_lvl_2_i1, fld(stop_2 * v8_4, P_2))
                                if phase_stop_5 >= phase_start_5
                                    if v0_lvl_2_idx[v0_lvl_2_q] < phase_start_5
                                        v0_lvl_2_q = Finch.scansearch(v0_lvl_2_idx, phase_start_5, v0_lvl_2_q, v0_lvl_2_q_stop - 1)
                                    end
                                    while true
                                        v0_lvl_2_i = v0_lvl_2_idx[v0_lvl_2_q]
                                        if v0_lvl_2_i < phase_stop_5
                                            v0_lvl_3_val_3 = v0_lvl_3_val[v0_lvl_2_q]
                                            v3_lvl_3_val[v3_lvl_2_q] = (v0_lvl_3_val_3 != 0) + v3_lvl_3_val[v3_lvl_2_q]
                                            v0_lvl_2_q += 1
                                        else
                                            phase_stop_7 = min(phase_stop_5, v0_lvl_2_i)
                                            if v0_lvl_2_i == phase_stop_7
                                                v0_lvl_3_val_3 = v0_lvl_3_val[v0_lvl_2_q]
                                                v3_lvl_3_val[v3_lvl_2_q] += v0_lvl_3_val_3 != 0
                                                v0_lvl_2_q += 1
                                            end
                                            break
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                Finch.resize_if_smaller!(v4_lvl_2_val, (v1.body.dims[2]).stop)
                Finch.fill_range!(v4_lvl_2_val, 0, 1, (v1.body.dims[2]).stop)
                for v10_4 = (v1.body.dims[2]).start:(v1.body.dims[2]).stop
                    v4_lvl_q = (1 - 1) * (v1.body.dims[2]).stop + v10_4
                    phase_start_12 = max((v1.body.dims[1]).start, 1 + fld(stop * (-1 + v10_4), P))
                    phase_stop_13 = min((v1.body.dims[1]).stop, fld(stop * v10_4, P))
                    if phase_stop_13 >= phase_start_12
                        v4_lvl_2_val[v4_lvl_q] = 1 + -phase_start_12 + phase_stop_13 + v4_lvl_2_val[v4_lvl_q]
                    end
                end
                Finch.resize_if_smaller!(v5_lvl_2_val, (v2.body.dims[2]).stop)
                Finch.fill_range!(v5_lvl_2_val, 0, 1, (v2.body.dims[2]).stop)
                for v12_4 = (v2.body.dims[2]).start:(v2.body.dims[2]).stop
                    v5_lvl_q = (1 - 1) * (v2.body.dims[2]).stop + v12_4
                    phase_start_15 = max((v2.body.dims[1]).start, 1 + fld(stop_2 * (-1 + v12_4), P_2))
                    phase_stop_16 = min((v2.body.dims[1]).stop, fld(stop_2 * v12_4, P_2))
                    if phase_stop_16 >= phase_start_15
                        v5_lvl_2_val[v5_lvl_q] = 1 + -phase_start_15 + phase_stop_16 + v5_lvl_2_val[v5_lvl_q]
                    end
                end
                resize!(v5_lvl_2_val, (v2.body.dims[2]).stop)
                resize!(v4_lvl_2_val, (v1.body.dims[2]).stop)
                resize!(v3_lvl_3_val, (v2.body.dims[2]).stop * (v1.body.dims[2]).stop)
                (v3 = Tensor((DenseLevel){Int64}((DenseLevel){Int64}(ElementLevel{0, Int64, Int64}(v3_lvl_3_val), (v2.body.dims[2]).stop), (v1.body.dims[2]).stop)), v4 = Tensor((DenseLevel){Int64}(ElementLevel{0, Int64, Int64}(v4_lvl_2_val), (v1.body.dims[2]).stop)), v5 = Tensor((DenseLevel){Int64}(ElementLevel{0, Int64, Int64}(v5_lvl_2_val), (v2.body.dims[2]).stop)))
            end)
end