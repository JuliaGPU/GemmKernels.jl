# a simple immutable array type backed by stack memory
#
# similar to StaticArrays, but immutable to prevent optimization bugs (JuliaLang/julia#41800)

struct LocalArray{S <: Tuple, T, N, L} <: AbstractArray{T,N}
    data::NTuple{L,T}

    LocalArray{S,T,N,L}(::UndefInitializer) where {S,T,N,L} = new{S,T,N,L}()
    LocalArray{S,T,N,L}(data::NTuple{L,T}) where {S,T,N,L} = new{S,T,N,L}(data)
end

@inline @generated function LocalArray{S,T}(args...) where {S,T}
    dims = (S.parameters...,)
    N = length(dims)
    L = prod(dims)
    @assert isbitstype(T)
    quote
        LocalArray{S, T, $N, $L}(args...)
    end
end

# array interface
Base.IndexStyle(::Type{<:LocalArray}) = IndexLinear()
Base.size(x::LocalArray{S}) where {S} = (S.parameters...,)

# indexing
Base.@propagate_inbounds function Base.getindex(v::LocalArray, i::Int)
    @boundscheck checkbounds(v,i)
    @inbounds v.data[i]
end
Base.@propagate_inbounds function Base.setindex(v::LocalArray{S,T,N,L}, val, i::Int) where {S,T,N,L}
    @boundscheck checkbounds(v,i)
    new_data = Base.setindex(v.data, convert(T, val), i)
    LocalArray{S,T,N,L}(new_data)
end
## XXX: Base's setindex doesn't have a ND version
Base.@propagate_inbounds function Base.setindex(v::LocalArray{S,T,N,L}, val, is::Int...) where {S,T,N,L}
    @boundscheck checkbounds(v,is...)
    I = CartesianIndex(is...)
    i = LinearIndices(v)[I]
    new_data = Base.setindex(v.data, convert(T, val), i)
    LocalArray{S,T,N,L}(new_data)
end

# helper macro
"""
    @immutable local_array[...]
    @immutable local_array[...] = ...

Helper macro that rewrites array indexing operations on a `LocalArray` to call non-mutating
versions of the array indexing functions. Instead, the macro will generate an expression
that overwrites the array with the new value.
"""
macro immutable(ex)
    # is this an assignment?
    if Meta.isexpr(ex, :(=))
        ex, val = ex.args
    else
        val = nothing
    end
    Meta.isexpr(ex, :ref) || error("@immutable only works with indexing operations")

    # getindex does not need to be rewritten
    val === nothing && return ex

    # rewrite setindex to overwrite the array
    arr, idxs... = ex.args
    quote
        $(esc(arr)) = Base.setindex($(esc(arr)), $(esc(val)), $(map(esc, idxs)...))
    end
end
