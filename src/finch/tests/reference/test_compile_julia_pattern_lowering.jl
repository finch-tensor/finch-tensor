# EyeTensor

eval(let
        v0 = Finch.window(Finch.offset(Finch.diagmask, 0, 1), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0.0
        for v3 = _
            for v4 = _
                v2[v4,v3] = (Float64(v0[v4,v3]) + v1[v4,v3])
            end
        end
        return v2
    end
end)

# UpperTriangleTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.offset(Finch.uptrimask, 0, 1), Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0.0
        for v3 = _
            for v4 = _
                v2[v4,v3] = (Float64(v0[v4,v3]) + v1[v4,v3])
            end
        end
        return v2
    end
end)

# PairSumTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.pairsummask, Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0.0
        for v3 = _
            for v4 = _
                v2[v4,v3] = (Float64(v0[v4,v3]) + v1[v4,v3])
            end
        end
        return v2
    end
end)

# RollTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.rollmask(1, -4), Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0.0
        for v3 = _
            for v4 = _
                v2[v4,v3] = (Float64(v0[v4,v3]) + v1[v4,v3])
            end
        end
        return v2
    end
end)

# OneHotMaskTensor

eval(let
        v0 = Finch.window(Finch.onehotmask(3), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0
        for v3 = _
            v2[v3] = (Bool(v0[v3]) + v1[v3])
        end
        return v2
    end
end)

# ReshapeMaskTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.reshapemask((2, 3), (3, 2)), Finch.Extent(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1)), 4, 3, 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0
        for v3 = _
            for v4 = _
                for v5 = _
                    for v6 = _
                        v2[v6,v5,v4,v3] = (Bool(v0[v6,v5,v4,v3]) + v1[v6,v5,v4,v3])
                    end
                end
            end
        end
        return v2
    end
end)

# ChunkMaskTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.chunkmask(1, 3), Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0
        for v3 = _
            for v4 = _
                v2[v4,v3] = (Bool(v0[v4,v3]) + v1[v4,v3])
            end
        end
        return v2
    end
end)

# SplitMaskTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.splitmask(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1)), 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
    Finch.@finch_kernel function main(v0,v1,v2)
        v2 .= 0
        for v3 = _
            for v4 = _
                v2[v4,v3] = (Bool(v0[v4,v3]) + v1[v4,v3])
            end
        end
        return v2
    end
end)