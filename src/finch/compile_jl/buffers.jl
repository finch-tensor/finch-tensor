module FinchInterop

using Base: @propagate_inbounds

"""
Resizable storage with an explicit owner. Borrowed storage is shared until resize;
the first resize detaches it, preserving the surviving prefix. Exporting storage
borrows it again so existing views remain valid across subsequent resizes.

The byte layout supports both packed NumPy records and aligned Julia tuples.
Index fields are shifted on access, without rewriting the borrowed allocation.
"""
mutable struct Buffer{T,L,I} <: AbstractVector{T}
    data::Vector{UInt8}
    layout::L
    owner::Any
    shifted::Bool
    writable::Bool
end

layout(::Type{T}) where {T} = (sizeof(T), ())
layout(::Type{T}) where {T<:Tuple} =
    (sizeof(T), ntuple(i -> (Int(fieldoffset(T, i)), layout(fieldtype(T, i))), fieldcount(T)))

function Buffer(data::Vector{T}, ::Val{I}=Val(())) where {T,I}
    borrow(T, reinterpret(UInt8, data), layout(T), Val(I), false)
end

valid_layout(::Type{T}, layout) where {T} = layout == (sizeof(T), ())
function valid_layout(::Type{T}, layout) where {T<:Tuple}
    length(layout[2]) == fieldcount(T) || return false
    all(enumerate(layout[2])) do (i, (offset, child))
        0 <= offset <= layout[1] - child[1] && valid_layout(fieldtype(T, i), child)
    end
end

function borrow(::Type{T}, bytes, layout::L, ::Val{I}, shifted, writable=true) where {T,L,I}
    isbitstype(T) || throw(ArgumentError("buffer elements must be isbits"))
    eltype(bytes) == UInt8 && stride(bytes, 1) == 1 ||
        throw(ArgumentError("buffer storage must be contiguous bytes"))
    layout[1] > 0 || throw(ArgumentError("buffer elements must have nonzero size"))
    valid_layout(T, layout) && length(bytes) % layout[1] == 0 ||
        throw(ArgumentError("invalid buffer layout"))
    data = unsafe_wrap(Vector{UInt8}, pointer(bytes), length(bytes); own=false)
    Buffer{T,L,I}(data, layout, bytes, shifted, writable)
end

owns_storage(buf::Buffer) = isnothing(buf.owner)

function owner(buf::Buffer)
    if owns_storage(buf)
        buf.owner = buf.data
    end
    return buf.owner
end

function storage(buf::Buffer)
    # Export the allocation's owner so views retain it after this buffer detaches.
    return (owner(buf), buf.layout)
end

Base.IndexStyle(::Type{<:Buffer}) = IndexLinear()
Base.size(buf::Buffer) = (length(buf.data) ÷ buf.layout[1],)
Base.similar(::Buffer, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
Base.dataids(buf::Buffer) = Base.dataids(buf.data)
Base.copy(buf::Buffer{T,L,I}) where {T,L,I} =
    Buffer{T,L,I}(copy(buf.data), buf.layout, nothing, buf.shifted, true)

@generated function load(::Type{T}, ptr, layout) where {T}
    if T <: Tuple
        fields = [:(load($(fieldtype(T, i)), ptr + layout[2][$i][1], layout[2][$i][2]))
                  for i in 1:fieldcount(T)]
        return Expr(:tuple, fields...)
    end
    return :(unsafe_load(Ptr{T}(ptr)))
end

@generated function store!(ptr, value::T, layout) where {T}
    if T <: Tuple
        fields = [:(store!(ptr + layout[2][$i][1], value[$i], layout[2][$i][2]))
                  for i in 1:fieldcount(T)]
        return Expr(:block, fields..., :(nothing))
    end
    return :(unsafe_store!(Ptr{T}(ptr), value))
end

@inline offset(value, op) = op(value, one(value))
@inline offset(value::Tuple, op) = map(x -> offset(x, op), value)
@inline shift(value, ::Val, op) = offset(value, op)
@inline shift(value::Tuple, ::Val{I}, op) where {I} =
    ntuple(i -> i in I ? offset(value[i], op) : value[i], Val(length(value)))

@propagate_inbounds function Base.getindex(buf::Buffer{T,L,I}, i::Int) where {T,L,I}
    @boundscheck checkbounds(buf, i)
    value = GC.@preserve buf load(T, pointer(buf.data) + (i - 1) * buf.layout[1], buf.layout)
    return buf.shifted ? shift(value, Val(I), +) : value
end

@propagate_inbounds function Base.setindex!(buf::Buffer{T,L,I}, value, i::Int) where {T,L,I}
    @boundscheck checkbounds(buf, i)
    buf.writable || throw(ReadOnlyMemoryError())
    value = convert(T, value)
    value = buf.shifted ? shift(value, Val(I), -) : value
    GC.@preserve buf store!(pointer(buf.data) + (i - 1) * buf.layout[1], value, buf.layout)
    return buf
end

function Base.resize!(buf::Buffer, n::Integer)
    n >= 0 || throw(ArgumentError("buffer length must be nonnegative"))
    n == length(buf) && return buf
    nbytes = Base.checked_mul(Int(n), buf.layout[1])
    if owns_storage(buf)
        resize!(buf.data, nbytes)
    else
        data = Vector{UInt8}(undef, nbytes)
        copyto!(data, 1, buf.data, 1, min(nbytes, length(buf.data)))
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
