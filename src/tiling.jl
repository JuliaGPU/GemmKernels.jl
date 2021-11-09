module Tiling

# -----------
# Tile object
# -----------

export Tile

"""
    Tile{size, names, T}

A [`Tile`](@ref) represents a part of a multidimensional tensor that is
contiguous and aligned to the tensor's dimensions.

Note that the dimensions of this [`Tile`](@ref) are named. Similar to a
[`NamedTuple`](@ref), the names are stored as a type parameter `names`.

A [`Tile`](@ref) contains several fields:
- `index`: A [`NamedTuple`](@ref) that represents the "first" multidimensional
           index of the parent tensor that this tile contains.
- `base`: The part of the `index` that depends on runtime values, such as the
          `threadIdx`.
- `offset`: The part of the `index` that is known at compile-time.
- `size`: A [`NamedTuple`](@ref) representing the size of the tile along each
          dimension.

You can also project a [`Tile`](@ref) (i.e. drop certain dimensions) by
accessing a special "field" of which the name is derived from the dimensions
you intend to keep.

For example, to drop the `K` dimension of a tile containing `M`, `N` and `K`
dimensions, you can use the syntax `tile.MN`.
"""
struct Tile{size, names, T}
    base::NamedTuple{names, T}
    offset::NamedTuple{names, T}
end

@inline _size(tile::Tile{size, names, T}) where {size, names, T} = size
@inline _names(tile::Tile{size, names, T}) where {size, names, T} = names

"""
    Tile(; kw_args...)

Creates a new [`Tile`](@ref) of the given `size`, with zero `base` and
`offset`. The `size` for each dimension must be specified by a keyword
argument.

# Example
```julia
GemmKernels.Tiling.Tile(M = 24, N = 16, K = 4)
```
"""
Tile(; kw_args...) = Tile((; kw_args...))

"""
    Tile(size::NamedTuple{names, T})

Creates a new [`Tile`](@ref) of the given `size`, with zero `base` and
`offset`.

# Arguments
- `size`: A `NamedTuple` representing the size of the [`Tile`](@ref).

# Example
```julia
GemmKernels.Tiling.Tile((M = 24, N = 16, K = 4))
```
"""
@inline Tile(size::NamedTuple{names, T}) where {names, T} = Tile{size, names, T}(map(x -> 0, size), map(x -> 0, size))

# ------------
# Pretty print
# ------------

function Base.show(io::IO, tile::Tile{size, names, T}) where {size, names, T}
    print(io, "base:   ", tile.base, '\n')
    print(io, "offset: ", tile.offset, '\n')
    print(io, "size:   ", tile.size)
end

# --------------------------
# Projection & transposition
# --------------------------

@inline _projection_impl(base::NamedTuple{names, T}, offset::NamedTuple{names, T}, size::NamedTuple{names, T}) where {names, T} = Tile{size, names, T}(base, offset)

@generated function _getproperty_impl(tile::Tile{size, names, T}, ::Val{sym}) where {names, T, sym, size}
    if sym == :base || sym == :offset
        # fields
        return :(getfield(tile, sym))
    elseif sym == :size
        # size
        return size
    elseif sym == :index
        # index: sum of base and offset
        return :(map(+, getfield(tile, :base), getfield(tile, :offset)))
    else
        # tile projection
        sym_str = String(sym)
        names = ntuple(i -> Symbol(sym_str[i]), length(sym_str))
        return :( _projection_impl(NamedTuple{$names}(getfield(tile, :base)),
                                   NamedTuple{$names}(getfield(tile, :offset)),
                                   NamedTuple{$names}(size)) )
    end
end

@generated function Base.transpose(tile::Tile{size, names, T}) where {names, T, size}
    new_names = reverse(names)
    return :( _projection_impl(NamedTuple{$new_names}(tile.base), NamedTuple{$new_names}(tile.offset), NamedTuple{$new_names}(size)) )
end

@inline Base.getproperty(tile::Tile{size, names, T}, sym::Symbol) where {names, T, size} = _getproperty_impl(tile, Val(sym))

# -------------
# Linearisation
# -------------

export linearise

"""
    linearise(coord, dims)

Convert a multidimensional coordinate to a linear index with respect to a
tensor with dimensions `dims`.

# Arguments
- `coord`: The coordinate (base, offset, or index) to linearise.
- `dims`: The dimensions of the parent tensor.
"""
@inline function linearise(coord, dims)
    ind = Tuple(coord) .+ 1
    @inbounds return LinearIndices(Tuple(dims))[ind...]
end

# -----------
# Translation
# -----------

export translate_base, translate_offset

"""
    translate_base(tile, offset)

Translate (i.e. move) a [`Tile`](@ref) by a constant `offset`.
The `offset` is added to the [`Tile`](@ref)'s base.

# Arguments
- `tile`: The [`Tile`](@ref) to translate.
- `offset`: The `offset` in each dimension.
"""
function translate_base end

"""
    translate_offset(tile, offset)

Translate (i.e. move) a [`Tile`](@ref) by a constant `offset`.
The `offset` is added to the [`Tile`](@ref)'s base.

# Arguments
- `tile`: The [`Tile`](@ref) to translate.
- `offset`: The `offset` in each dimension.
"""
function translate_offset end

@inline function translate_base(tile::Tile{size, names, T}, offset::NamedTuple{names, T}) where {names, T, size}
    new_base = map(+, tile.base, offset)
    return Tile{size, names, T}(new_base, tile.offset)
end

@inline translate_base(tile::Tile{size, names, T}, offset::Tuple) where {names, T, size} = translate_base(tile, NamedTuple{names}(offset))

@inline function translate_offset(tile::Tile{size, names, T}, offset::NamedTuple{names, T}) where {names, T, size}
    new_offset = map(+, tile.offset, offset)
    return Tile{size, names, T}(tile.base, new_offset)
end

@inline translate_offset(tile::Tile{size, names, T}, offset::Tuple) where {names, T, size} = translate_offset(tile, NamedTuple{names}(offset))

# -------------
# TileIterators
# -------------

export TileIterator

"""
    TileIterator{names, T, N, R}

A [`TileIterator`](@ref) represents an iterator over a set of [`Tile`](@ref)s.

See also: [`subdivide`](@ref), [`parallellise`](@ref).
"""
struct TileIterator{tile_size, parent_size, names, T, S, col_major}
    parent::Tile{parent_size, names, T}
    subtile_indices::S
    idx::Int32
    step::Int32
end

# ----------------
# Parallellisation
# ----------------

export parallellise, subdivide

"""
    parallellise(tile, tiling_size, idx, size)

Split the given `tile` in subtiles of size `tiling_size` across a group of
cooperating entities (e.g. warps, threads, ...).

Unlike [`subdivide`](@ref), the `tile` need not be completely covered by
`count` tiles of size `tiling_size`. If that's not the case, the subtiles
are evenly parallellised across all cooperating entities.

Returns a [`TileIterator`](@ref) that iterates over the [`Tile`](@ref)s of
the calling entity.

# Arguments
- `tile`: The [`Tile`](@ref) to parallellise.
- `tiling_size`: A `NamedTuple` indicating the size of a subtile along each dimension.
- `idx`: The identity of the calling entity.
- `count`: The number of cooperating entities.
"""
@inline function parallellise(tile::Tile{size, names, T}, tiling_size::Tile{tile_sz, names, T}, idx, count, col_major::Bool=true) where {names, T, size, tile_sz}
    # Transpose
    tile = col_major ? tile : transpose(tile)
    tiling_size = col_major ? tiling_size : transpose(tiling_size)

    # Number of tiles along each dimension
    num_tiles = map(div, Tuple(_size(tile)), Tuple(_size(tiling_size)))

    parent = tile
    subtile_indices = CartesianIndices(num_tiles)
    step = count

    return TileIterator{_size(tiling_size), _size(tile), _names(tile), T, typeof(subtile_indices), col_major}(parent, subtile_indices, convert(Int32, idx), convert(Int32, step))
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
@inline function subdivide(tile::Tile{size, names, T}, tiling_size::Tile{tile_sz, names, T}, idx, count) where {names, T, size, tile_sz}
    iter = iterate(parallellise(tile, tiling_size, idx, count))::Tuple{Tile,Any}
    iter === nothing && throw(BoundsError())
    @inbounds iter[1]
end

@inline function Base.iterate(it::TileIterator{tile_size, parent_size, names, T, S, col_major}, state = 1) where {tile_size, parent_size, names, T, S, col_major}
    if state > length(it.subtile_indices)
        return nothing
    end

    # Calculate base and offset in number of tiles
    @inbounds base   = Tuple(it.parent.base)   .+ (Tuple(it.subtile_indices[it.idx]) .- 1) .* Tuple(tile_size)
    @inbounds offset = Tuple(it.parent.offset) .+ (Tuple(it.subtile_indices[state])  .- 1) .* Tuple(tile_size)

    # Create tile
    tile = Tile{tile_size, names, T}(NamedTuple{names, T}(base), NamedTuple{names, T}(offset))

    # Transpose
    tile = col_major ? tile : transpose(tile)

    return (tile, state + it.step)
end

end
