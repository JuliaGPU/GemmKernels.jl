export Layout
module Layout

using CUDA
using KernelAbstractions.Extras: @unroll
using GemmKernels.Tiling

# ---------------------
# Customise computation
# ---------------------

@inline threadblock_condition(layout_a, layout_b, block_i, block_j, block_k, block_tile) = true

# ----------------------
# Explicit vectorisation
# ----------------------

struct Vec{N, T} end

@inline @generated function vloada(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}, i::Integer = 1) where {N, T, AS}
    alignment = sizeof(T) * N

    return quote
        vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, VecElement{T}}, AS}, ptr)
        return unsafe_load(vec_ptr, (i-1) ÷ N + 1, Val($alignment))
    end
end

@inline @generated function vstorea!(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}, x, i::Integer = 1) where {N, T, AS}
    alignment = sizeof(T) * N

    return quote
        vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, VecElement{T}}, AS}, ptr)
        return unsafe_store!(vec_ptr, x, (i-1) ÷ N + 1, Val($alignment))
    end
end

# -----------
# Layout base
# -----------

abstract type LayoutBase{T} end

@inline eltype(::Type{<:LayoutBase{T}}) where {T} = T
@inline physical_size(::Type{<:LayoutBase{T}}, logical_size::NamedTuple) where {T} = Tuple(logical_size)

# --------------
# Padded layouts
# --------------

abstract type Padded{L, P} end

@inline function pad_logical_coord(::Type{<:Padded{L, P}}, crd::NamedTuple) where {L, P}
    t = Tuple(crd)
    return typeof(crd)((Base.first(t) + P, Base.tail(t)...))
end

@inline eltype(::Type{<:Padded{L, P}}) where {L, P} = eltype(L)
@inline physical_size(::Type{<:Padded{L, P}}, logical_size::NamedTuple) where {L, P} = physical_size(L, pad_logical_coord(Padded{L, P}, logical_size))
@inline load(::Type{<:Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = load(L, workspace, tile)
@inline store!(::Type{<:Padded{L, P}}, workspace, value, tile::Tile) where {L, P} = store!(L, workspace, value, tile::Tile)

# ---------------
# AlignedColMajor
# ---------------

abstract type AlignedColMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{<:Padded{AlignedColMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[1] + P, logical_size[2])
@inline physical_size(::Type{<:AlignedColMajor{T}}, logical_size::NamedTuple) where {T} = (logical_size[1] , logical_size[2])

@inline fragtype(::Type{<:AlignedColMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{16 ÷ sizeof(T), VecElement{T}}

@inline function load(::Type{<:AlignedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    linear_base = linearise(tile.base, Base.size(workspace))
    linear_offset = linearise(tile.offset, Base.size(workspace))

    return vloada(Vec{N, T}, pointer(workspace), linear_base + linear_offset - 1)
end

@inline function store!(::Type{<:AlignedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    linear_base = linearise(tile.base, Base.size(workspace))
    linear_offset = linearise(tile.offset, Base.size(workspace))

    vstorea!(Vec{N, T}, pointer(workspace), value, linear_base + linear_offset - 1)
end

# --------
# Diagonal
# --------

abstract type Diagonal{T} <: LayoutBase{T} end

@inline function load(::Type{<:Diagonal{T}}, workspace, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    # The row index is given by t.index[1] + (k - 1), the column index is given by t.index[2] (0-based).
    # Only load on the diagonal, i.e. if row and column are equal.
    # Note that t.index[2] is 0-based, so we need to add 1 before loading from workspace.
    return ntuple(k -> VecElement{Float16}(tile.index[1] + k - 1 == tile.index[2] ? @inbounds(workspace[tile.index[2] + 1]) : 0), Val(8))
end

@inline threadblock_condition(layout_a::Type{<:Diagonal{T}}, layout_b, block_i, block_j, block_k, block_tile) where {T} = abs(block_i - block_k) <= block_tile.size.K

# ---------------
# AlignedRowMajor
# ---------------

abstract type AlignedRowMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{<:Padded{AlignedRowMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[2] + P, logical_size[1])

@inline fragtype(::Type{<:AlignedRowMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{16 ÷ sizeof(T), VecElement{T}}

@inline function load(::Type{<:AlignedRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    linear_base = linearise(reverse(Tuple(tile.base)), Base.size(workspace))
    linear_offset = linearise(reverse(Tuple(tile.offset)), Base.size(workspace))

    return vloada(Vec{N, T}, pointer(workspace), linear_base + linear_offset - 1)
end

@inline function store!(::Type{<:AlignedRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    linear_base = linearise(reverse(Tuple(tile.base)), Base.size(workspace))
    linear_offset = linearise(reverse(Tuple(tile.offset)), Base.size(workspace))

    vstorea!(Vec{N, T}, pointer(workspace), value, linear_base + linear_offset - 1)
end

# -------------------
# InterleavedColMajor
# -------------------

abstract type InterleavedColMajor{T} <: LayoutBase{T} end

@inline fragtype(::Type{<:InterleavedColMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], Complex{T}}

@inline function load(::Type{<:InterleavedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            @inbounds val = workspace[t.index[1] + 1, t.index[2] + 1]
            x = Base.setindex(x, val, (i - 1) * tile.size[2] + j)
        end
    end

    return x
end

@inline function store!(::Type{<:InterleavedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            @inbounds workspace[t.index[1] + 1, t.index[2] + 1] = val
        end
    end
end

# -------------------
# InterleavedRowMajor
# -------------------

abstract type InterleavedRowMajor{T} <: LayoutBase{T} end

@inline fragtype(::Type{<:InterleavedRowMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], Complex{T}}

@inline function load(::Type{<:InterleavedRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            @inbounds val = workspace[t.index[2] + 1, t.index[1] + 1]
            x = Base.setindex(x, val, (i - 1) * tile.size[2] + j)
        end
    end

    return x
end

@inline function store!(::Type{<:InterleavedRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            @inbounds workspace[t.index[2] + 1, t.index[1] + 1] = val
        end
    end
end

# -------------
# SplitColMajor
# -------------

abstract type SplitColMajor{T} <: LayoutBase{T} end

@inline function physical_size(::Type{<:SplitColMajor{T}}, logical_size::NamedTuple) where {T}
    t = Tuple(logical_size)
    return (t..., 2)
end

@inline fragtype(::Type{<:SplitColMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], Complex{T}}

@inline function load(::Type{<:SplitColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            @inbounds val = workspace[t.index[1] + 1, t.index[2] + 1, 1] + im *
                            workspace[t.index[1] + 1, t.index[2] + 1, 2]
            x = Base.setindex(x, val, (i - 1) * tile.size[2] + j)
        end
    end

    return x
end

@inline function store!(::Type{<:SplitColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            @inbounds workspace[t.index[1] + 1, t.index[2] + 1, 1] = val.re
            @inbounds workspace[t.index[1] + 1, t.index[2] + 1, 2] = val.im
        end
    end
end

# -------------
# SplitRowMajor
# -------------

abstract type SplitRowMajor{T} <: LayoutBase{T} end

@inline function physical_size(::Type{<:Padded{SplitRowMajor{T}, P}}, logical_size::NamedTuple) where {T, P}
    return (logical_size[2] + P, logical_size[1], 2)
end

@inline fragtype(::Type{<:SplitRowMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], Complex{T}}

@inline function load(::Type{<:SplitRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            @inbounds val = workspace[t.index[2] + 1, t.index[1] + 1, 1] + im *
                            workspace[t.index[2] + 1, t.index[1] + 1, 2]
            x = Base.setindex(x, val, (i - 1) * tile.size[2] + j)
        end
    end

    return x
end

@inline function store!(::Type{<:SplitRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            @inbounds workspace[t.index[2] + 1, t.index[1] + 1, 1] = val.re
            @inbounds workspace[t.index[2] + 1, t.index[1] + 1, 2] = val.im
        end
    end
end

end
