module FinchInterop

using Base: @propagate_inbounds

"""
Resizable storage with an explicit owner. Borrowed storage is shared until resize;
the first resize detaches it, preserving the surviving prefix. Exporting storage
borrows it again so existing views remain valid across subsequent resizes.
"""
mutable struct Buffer{T} <: AbstractVector{T}
    data::Vector{T}
    owner::Any
    writable::Bool
end

Buffer(data::Vector{T}) where {T} = Buffer{T}(data, data, true)

function borrow(::Type{T}, bytes, writable=true) where {T}
    isbitstype(T) && sizeof(T) > 0 ||
        throw(ArgumentError("buffer elements must be nonempty isbits"))
    eltype(bytes) == UInt8 && stride(bytes, 1) == 1 && length(bytes) % sizeof(T) == 0 ||
        throw(ArgumentError("buffer storage must be contiguous bytes"))
    data = unsafe_wrap(Vector{T}, Ptr{T}(pointer(bytes)), length(bytes) ÷ sizeof(T); own=false)
    Buffer{T}(data, bytes, writable)
end

owns_storage(buf::Buffer) = isnothing(buf.owner)

function owner(buf::Buffer)
    if owns_storage(buf)
        buf.owner = buf.data
    end
    return buf.owner
end

# Export the allocation's owner so views retain it after this buffer detaches.
storage(buf::Buffer) = owner(buf)

Base.IndexStyle(::Type{<:Buffer}) = IndexLinear()
Base.size(buf::Buffer) = size(buf.data)
Base.similar(::Buffer, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
Base.dataids(buf::Buffer) = Base.dataids(buf.data)
Base.copy(buf::Buffer{T}) where {T} = Buffer{T}(copy(buf.data), nothing, true)

function Base.copyto!(dest::Buffer, dstart::Integer, src::Buffer, sstart::Integer, n::Integer)
    if dest !== src
        return invoke(copyto!, Tuple{AbstractArray,Integer,AbstractArray,Integer,Integer},
                      dest, dstart, src, sstart, n)
    end
    n == 0 || dest.writable || throw(ReadOnlyMemoryError())
    # The AbstractArray fallback copies the entire source to handle aliasing.
    # Hash checkpoints copy short ranges repeatedly; the vector handles
    # overlap directly, keeping their total cost linear in the buffer length.
    copyto!(dest.data, dstart, src.data, sstart, n)
    return dest
end

@propagate_inbounds Base.getindex(buf::Buffer, i::Int) = buf.data[i]

@propagate_inbounds function Base.setindex!(buf::Buffer, value, i::Int)
    @boundscheck buf.writable || throw(ReadOnlyMemoryError())
    buf.data[i] = value
    return buf
end

function Base.resize!(buf::Buffer{T}, n::Integer) where {T}
    n >= 0 || throw(ArgumentError("buffer length must be nonnegative"))
    n == length(buf) && return buf
    if owns_storage(buf)
        resize!(buf.data, n)
    else
        data = Vector{T}(undef, n)
        copyto!(data, 1, buf.data, 1, min(n, length(buf.data)))
        buf.data = data
        buf.owner = nothing
        buf.writable = true
    end
    return buf
end

Base.empty!(buf::Buffer) = resize!(buf, 0)
function Base.push!(buf::Buffer, value)
    resize!(buf, length(buf) + 1)
    buf[end] = value
    return buf
end

end
