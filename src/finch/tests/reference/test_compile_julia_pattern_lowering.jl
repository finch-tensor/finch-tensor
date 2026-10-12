# EyeTensor

eval(let
        v0 = Finch.window(Finch.offset(Finch.diagmask, 0, 1), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0.0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0.0)>>= (Float64(v0[v12,v11]) + v1[v12,v11])
            end
        end
        return v2
    end
end)

# UpperTriangleTensor

eval(let
        v0 = Finch.window(Finch.offset(Finch.lotrimask, 0, -1), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0.0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0.0)>>= (Float64(v0[v12,v11]) + v1[v12,v11])
            end
        end
        return v2
    end
end)

# PairSumTensor

eval(let
        v0 = Finch.window(Finch.repeatmask(2), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0.0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0.0)>>= (Float64(v0[v12,v11]) + v1[v12,v11])
            end
        end
        return v2
    end
end)

# RollTensor

eval(let
        v0 = Finch.window(Finch.rollmask(1, 4), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0.0, Float64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0.0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0.0)>>= (Float64(v0[v12,v11]) + v1[v12,v11])
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
        v4 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v4)
        v2 .= 0
        for v7 = 1:v4
            v2[v7] <<Finch.initwrite(0)>>= (Bool(v0[v7]) + v1[v7])
        end
        return v2
    end
end)

# ReshapeMaskTensor

eval(let
        v0 = Finch.swizzle(Finch.window(Finch.reshapemask((2, 3), (3, 2)), Finch.Extent(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1)), 4, 3, 2, 1)
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1), 1), 1))
        v7 = 1
        v8 = 1
        v9 = 1
        v10 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v7,v8,v9,v10)
        v2 .= 0
        for v19 = 1:v7
            for v20 = 1:v8
                for v21 = 1:v9
                    for v22 = 1:v10
                        v2[v22,v21,v20,v19] <<Finch.initwrite(0)>>= (Bool(v0[v22,v21,v20,v19]) + v1[v22,v21,v20,v19])
                    end
                end
            end
        end
        return v2
    end
end)

# ChunkMaskTensor

eval(let
        v0 = Finch.window(Finch.chunkmask(1, 3), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0)>>= (Bool(v0[v12,v11]) + v1[v12,v11])
            end
        end
        return v2
    end
end)

# SplitMaskTensor

eval(let
        v0 = Finch.window(Finch.splitmask(1, 1), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0)>>= (Bool(v0[v12,v11]) + v1[v12,v11])
            end
        end
        return v2
    end
end)

# RandomMaskTensor

eval(let
        v0 = Finch.window(Finch.randommask((), 0.4; seed=UInt64(42)))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
        v3 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v3)
        v2 .= 0
        for v6 = 1:v3
            v2[v6] <<Finch.initwrite(0)>>= (Bool(v0[]) + v1[v6])
        end
        return v2
    end
end)

# RandomMaskTensor

eval(let
        v0 = Finch.window(Finch.randommask((1,), 0.25; seed=UInt64(42)), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1))
        v4 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v4)
        v2 .= 0
        for v7 = 1:v4
            v2[v7] <<Finch.initwrite(0)>>= (Bool(v0[v7]) + v1[v7])
        end
        return v2
    end
end)

# RandomMaskTensor

eval(let
        v0 = Finch.window(Finch.randommask((1, 1), 0.5; seed=UInt64(18446744073709551615)), Finch.Extent(1, 1), Finch.Extent(1, 1))
        v1 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v2 = Finch.Tensor(Finch.DenseLevel(Finch.DenseLevel(Finch.ElementLevel(0, Int64[]), 1), 1))
        v5 = 1
        v6 = 1
    Finch.@finch_kernel function main(v0,v1,v2,v5,v6)
        v2 .= 0
        for v11 = 1:v5
            for v12 = 1:v6
                v2[v12,v11] <<Finch.initwrite(0)>>= (Int64(v0[v12,v11]) + v1[v12,v11])
            end
        end
        return v2
    end
end)