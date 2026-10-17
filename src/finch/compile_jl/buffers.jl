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

"""
Shifts the index fields `I` of each element up by one on load and down by one
on store, so Julia sees 1-based indices over 0-based storage.
"""
struct PlusOneBuffer{T,I,A<:AbstractVector{T}} <: AbstractVector{T}
    data::A
end

PlusOneBuffer(data::AbstractVector{T}, ::Val{I}) where {T,I} =
    PlusOneBuffer{T,I,typeof(data)}(data)

@inline offset(value, op) = op(value, one(value))
@inline offset(value::Tuple, op) = map(x -> offset(x, op), value)
@inline shift(value, ::Val, op) = offset(value, op)
@inline shift(value::Tuple, ::Val{I}, op) where {I} =
    ntuple(i -> i in I ? offset(value[i], op) : value[i], Val(length(value)))

@propagate_inbounds Base.getindex(buf::PlusOneBuffer{T,I}, i::Int) where {T,I} =
    shift(buf.data[i], Val(I), +)

@propagate_inbounds function Base.setindex!(buf::PlusOneBuffer{T,I}, value, i::Int) where {T,I}
    buf.data[i] = shift(value, Val(I), -)
    return buf
end

Base.IndexStyle(::Type{<:PlusOneBuffer}) = IndexLinear()
Base.size(buf::PlusOneBuffer) = size(buf.data)
Base.dataids(buf::PlusOneBuffer) = Base.dataids(buf.data)
Base.copy(buf::PlusOneBuffer{T,I}) where {T,I} = PlusOneBuffer(copy(buf.data), Val(I))
Base.resize!(buf::PlusOneBuffer, n::Integer) = (resize!(buf.data, n); buf)
Base.empty!(buf::PlusOneBuffer) = (empty!(buf.data); buf)
Base.push!(buf::PlusOneBuffer{T,I}, value) where {T,I} =
    (push!(buf.data, shift(value, Val(I), -)); buf)

# Both sides store the same shifted representation, so forward to the storage.
function Base.copyto!(dest::PlusOneBuffer{T,I}, dstart::Integer,
                      src::PlusOneBuffer{T,I}, sstart::Integer, n::Integer) where {T,I}
    copyto!(dest.data, dstart, src.data, sstart, n)
    return dest
end

end
