module Tiling

using GemmKernels: BitArrayIndex, constant, variadic

# -----------
# Tile object
# -----------

export Tile

"""
    Tile{size, names, N}

A [`Tile`](@ref) represents a part of a multidimensional tensor that is
contiguous and aligned to the tensor's dimensions.

Note that the dimensions of this [`Tile`](@ref) are named. Similar to a
[`NamedTuple`](@ref), the names are stored as a type parameter `names`.

A [`Tile`](@ref) contains several fields:
- `index`: The offset of this tile relative to its parent tensor.
- `size`: A [`NamedTuple`](@ref) representing the size of the tile along each
          dimension.

You can also project a [`Tile`](@ref) (i.e. drop certain dimensions) by
accessing a special "field" of which the name is derived from the dimensions
you intend to keep.

For example, to drop the `K` dimension of a tile containing `M`, `N` and `K`
dimensions, you can use the syntax `tile.MN`.
"""
struct Tile{size, names, N}
    index::NamedTuple{names, NTuple{N, BitArrayIndex}}
end

@inline _size(tile::Tile{size, names}) where {size, names} = size
@inline _names(tile::Tile{size, names}) where {size, names} = names

"""
    Tile(; kw_args...)

Creates a new [`Tile`](@ref) of the given `size`, with zero `index`.
The `size` for each dimension must be specified by a keyword argument.

# Example
```julia
GemmKernels.Tiling.Tile(M = 24, N = 16, K = 4)
```
"""
Tile(; kw_args...) = Tile((; kw_args...))

"""
    Tile(size::NamedTuple{names})

Creates a new [`Tile`](@ref) of the given `size`, with zero `base` and
`offset`.

# Arguments
- `size`: A `NamedTuple` representing the size of the [`Tile`](@ref).

# Example
```julia
GemmKernels.Tiling.Tile((M = 24, N = 16, K = 4))
```
"""
@inline Tile(size::NamedTuple{names}) where {names} = Tile{size, names, length(size)}(map(x -> constant(0), size))

# ------------
# Pretty print
# ------------

function Base.show(io::IO, tile::Tile{size, names}) where {size, names}
    print(io, "index: ", tile.index, '\n')
    print(io, "size:  ", tile.size)
end

# --------------------------
# Projection & transposition
# --------------------------

@inline _projection_impl(index::NamedTuple{names, NTuple{N, T}}, size::NamedTuple{names, NTuple{N, U}}) where {names, N, T, U} = Tile{size, names, N}(index)

@inline _base_part(x::BitArrayIndex) = x.variadic_part
@inline _offset_part(x::BitArrayIndex) = x.known_one

@generated function _getproperty_impl(tile::Tile{size, names, N}, ::Val{sym}) where {names, sym, N, size}
    if sym == :index
        # fields
        return :(getfield(tile, sym))
    # Fallback for old code that uses the base and offset fields.
    elseif sym == :base
        return :(map(_base_part, getfield(tile, :index)))
    elseif sym == :offset
        return :(map(_offset_part, getfield(tile, :index)))
    elseif sym == :size
        # size
        return size
    else
        # tile projection
        sym_str = String(sym)
        names = ntuple(i -> Symbol(sym_str[i]), length(sym_str))
        return :( _projection_impl(NamedTuple{$names}(getfield(tile, :index)),
                                   NamedTuple{$names}(size)) )
    end
end

@generated function Base.transpose(tile::Tile{size, names}) where {names, size}
    new_names = reverse(names)
    return :( _projection_impl(NamedTuple{$new_names}(tile.index), NamedTuple{$new_names}(size)) )
end

@inline Base.getproperty(tile::Tile{size, names, N}, sym::Symbol) where {names, size, N} = _getproperty_impl(tile, Val(sym))

# -------------
# Linearisation
# -------------

export linearise

"""
    linearise(tile, dims)

Convert a multidimensional tile to a linear index with respect to a
tensor with dimensions `dims`.

# Arguments
- `tile`: The tile to linearise.
- `dims`: The dimensions of the parent tensor.
"""
@inline function linearise(tile::Tile, dims)
    ind = Tuple(map(x -> convert(Int, x), tile.index)) .+ 1
    @inbounds return LinearIndices(Tuple(dims))[ind...]
end

# -----------
# Translation
# -----------

export translate

"""
    translate(tile, offset)

Translate (i.e. move) a [`Tile`](@ref) by a variadic or constant `offset`.

# Arguments
- `tile`: The [`Tile`](@ref) to translate.
- `offset`: The `offset` in each dimension.
"""
function translate end

@inline function translate(tile::Tile{size, names, N}, offset::NamedTuple{names, NTuple{N, BitArrayIndex}}) where {names, size, N}
    new_index = map(+, tile.index, offset)
    return Tile{size, names, N}(new_index)
end

# -------------
# TileIterators
# -------------

export TileIterator

"""
    TileIterator{names, N, R}

A [`TileIterator`](@ref) represents an iterator over a set of [`Tile`](@ref)s.

See also: [`subdivide`](@ref), [`parallelise`](@ref).
"""
struct TileIterator{tile_size, parent_size, names, N, S, idxs, col_major}
    parent::Tile{parent_size, names, N}
    subtile_indices::S
    idx::Int32
end

# ----------------
# Parallelisation
# ----------------

export parallelise, subdivide

"""
    parallelise(tile, tiling_size, idx, size)

Split the given `tile` in subtiles of size `tiling_size` across a group of
cooperating entities (e.g. warps, threads, ...).

Unlike [`subdivide`](@ref), the `tile` need not be completely covered by
`count` tiles of size `tiling_size`. If that's not the case, the subtiles
are evenly parallelised across all cooperating entities.

Returns a [`TileIterator`](@ref) that iterates over the [`Tile`](@ref)s of
the calling entity.

# Arguments
- `tile`: The [`Tile`](@ref) to parallelise.
- `tiling_size`: A `NamedTuple` indicating the size of a subtile along each dimension.
- `idx`: The identity of the calling entity.
- `idxs`: The number of cooperating entities.
"""
@inline function parallelise(tile::Tile{size, names, N}, tiling_size::Tile{tile_sz, names, N}, idx, idxs, col_major::Bool=true) where {names, size, N, tile_sz}
    # Transpose
    tile = col_major ? tile : transpose(tile)
    tiling_size = col_major ? tiling_size : transpose(tiling_size)

    # Number of tiles along each dimension
    num_tiles = map(div, Tuple(_size(tile)), Tuple(_size(tiling_size)))

    parent = tile
    subtile_indices = CartesianIndices(num_tiles)

    return TileIterator{_size(tiling_size), _size(tile), _names(tile), N, typeof(subtile_indices), idxs, col_major}(parent, subtile_indices, convert(Int32, idx))
end

"""
    subdivide(tile, tiling_size, idx, count)

Split the given `tile` in subtiles of size `tiling_size` across a group of
`count` cooperating entities (e.g. warps, threads, ...).

The given `tile` must be completely covered by `count` tiles of size
`tiling_size`.

Returns the [`Tile`](@ref) that the calling entity is responsible for.

# Arguments
- `tile`: The [`Tile`](@ref) to subdivide.
- `tiling_size`: A `NamedTuple` indicating the size of a subtile along each dimension.
- `idx`: The identity of the calling entity.
- `count`: The number of cooperating entities.
"""
@inline function subdivide(tile::Tile{size, names}, tiling_size::Tile{tile_sz, names}, idx, count) where {names, size, tile_sz}
    iter = iterate(parallelise(tile, tiling_size, idx, count))::Tuple{Tile,Any}
    @boundscheck begin
        iter === nothing && throw(BoundsError())
    end
    @inbounds iter[1]
end

@inline function Base.iterate(it::TileIterator{tile_size, parent_size, names, N, S, idxs, col_major}, state = 1) where {tile_size, parent_size, names, N, S, idxs, col_major}
    if idxs > length(it.subtile_indices) && it.idx > length(it.subtile_indices)
        # the number of cooperating entities exceeds the number of subtiles.
        # the short-circuiting check against a static value is crucial for performance,
        # as it allows removing the dynamic check in many cases.
        return nothing
    end
    if state > length(it.subtile_indices)
        # we've exhausted the iterator
        return nothing
    end

    # Calculate index in number of tiles
    @inbounds variadic_part = map(x -> variadic(x-1), Tuple(it.subtile_indices[it.idx])) .* Tuple(tile_size)

    # Mask away bits in the variadic part that we know are 0.
    #
    # Suppose we parallelise a M x N = 16 x 32 tile in 8 x 4 tiles over 8 warps.
    # We know that bit 4 up until 63 of M and N are 0. This is
    # because in a single iteration, all warps handle a 16 x 16 tile,
    # so bits higher than position 4 will be set by the constant offset.

    mask = map((x, y) -> constant(x*y-1), Tuple(it.subtile_indices[idxs]), Tuple(tile_size))

    @inbounds variadic_part = variadic_part .& mask
    @inbounds constant_part = map(x -> constant(x-1), Tuple(it.subtile_indices[state])) .* Tuple(tile_size)
    @inbounds index = Tuple(it.parent.index) .+ variadic_part .+ constant_part

    # Create tile
    tile = Tile{tile_size, names, N}(NamedTuple{names}(index))

    # Transpose
    tile = col_major ? tile : transpose(tile)

    return (tile, state + idxs)
end

end
