export Layout
module Layout

using CUDA
using GPUifyLoops: @unroll
using GemmKernels.Tiling
using StaticArrays

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
    vec_len = (sizeof(T) * N) ÷ sizeof(Float32)

    return quote
        vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{$vec_len, VecElement{Float32}}, AS}, ptr)
        return unsafe_load(vec_ptr, (i-1) ÷ N + 1, Val($alignment))
    end
end

@inline @generated function vstorea!(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}, x, i::Integer = 1) where {N, T, AS}
    alignment = sizeof(T) * N
    vec_len = (sizeof(T) * N) ÷ sizeof(Float32)

    return quote
        vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{$vec_len, VecElement{Float32}}, AS}, ptr)
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

struct Padded{L, P} end

@inline function pad_logical_coord(::Type{Padded{L, P}}, crd::NamedTuple) where {L, P}
    t = Tuple(crd)
    return typeof(crd)((Base.first(t) + P, Base.tail(t)...))
end

@inline eltype(::Type{Padded{L, P}}) where {L, P} = eltype(L)
@inline physical_size(::Type{Padded{L, P}}, logical_size::NamedTuple) where {L, P} = physical_size(L, pad_logical_coord(Padded{L, P}, logical_size))
@inline load(::Type{Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = load(L, workspace, tile)
@inline store!(::Type{Padded{L, P}}, workspace, value, tile::Tile) where {L, P} = store!(L, workspace, value, tile::Tile)

# ---------------
# AlignedColMajor
# ---------------

struct AlignedColMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{Padded{AlignedColMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[1] + P, logical_size[2])

# TODO: cleanup vectorisation
@inline function load(::Type{AlignedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)
    N = (sizeof(T) * vec_len) ÷ sizeof(Float32)
    res = MArray{Tuple{size[1] ÷ vec_len, size[2]}, NTuple{N, VecElement{Float32}}}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            linear_base = linearise(t.base, Base.size(workspace))
            linear_offset = linearise(t.offset, Base.size(workspace))

            @inbounds res[i, j] = vloada(Vec{vec_len, T}, pointer(workspace), linear_base + linear_offset - 1)
        end
    end

    return res
end

@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            linear_base = linearise(t.base, Base.size(workspace))
            linear_offset = linearise(t.offset, Base.size(workspace))

            @inbounds vstorea!(Vec{vec_len, T}, pointer(workspace), value[i, j], linear_base + linear_offset - 1)
        end
    end
end

# --------
# Diagonal
# --------

struct Diagonal{T} <: LayoutBase{T} end

@inline bitcast_helper(x::NTuple{8, VecElement{Float16}}) = Base.llvmcall(
    "
    %ret = bitcast <8 x i16> %0 to <4 x float>
    ret <4 x float> %ret
    ", NTuple{4, VecElement{Float32}}, Tuple{NTuple{8, VecElement{Float16}}}, x)

@inline function load(::Type{Diagonal{T}}, workspace, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)
    N = (sizeof(T) * vec_len) ÷ sizeof(Float32)
    res = MArray{Tuple{size[1] ÷ vec_len, size[2]}, NTuple{N, VecElement{Float32}}}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            # The row index is given by t.index[1] + (k - 1), the column index is given by t.index[2] (0-based).
            # Only load on the diagonal, i.e. if row and column are equal.
            # Note that t.index[2] is 0-based, so we need to add 1 before loading from workspace.
            # TODO: Remove the <4 x float> everywhere, so we don't have to do this ugly casting all over the place.
            @inbounds res[i, j] = bitcast_helper(ntuple(k -> VecElement{Float16}(t.index[1] + k - 1 == t.index[2] ? @inbounds(workspace[t.index[2] + 1]) : 0), Val(8)))
        end
    end

    return res
end

@inline threadblock_condition(layout_a::Type{Diagonal{T}}, layout_b, block_i, block_j, block_k, block_tile) where {T} = abs(block_i - block_k) <= block_tile.size.K

# ---------------
# AlignedRowMajor
# ---------------

struct AlignedRowMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{Padded{AlignedRowMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[2] + P, logical_size[1])

# TODO: cleanup vectorisation
@inline function load(::Type{AlignedRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)
    N = (sizeof(T) * vec_len) ÷ sizeof(Float32)
    res = MArray{Tuple{size[1], size[2] ÷ vec_len}, NTuple{N, VecElement{Float32}}}(undef)

    @unroll for i = 1 : size[1]
        @unroll for j = 1 : vec_len : size[2]
            t = translate_offset(tile, (i - 1, j - 1))

            linear_base = linearise(reverse(Tuple(t.base)), Base.size(workspace))
            linear_offset = linearise(reverse(Tuple(t.offset)), Base.size(workspace))

            @inbounds res[i, j] = vloada(Vec{vec_len, T}, pointer(workspace), linear_base + linear_offset - 1)
        end
    end

    return res
end

@inline function store!(::Type{AlignedRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)

    @unroll for i = 1 : size[1]
        @unroll for j = 1 : vec_len : size[2]
            t = translate_offset(tile, (i - 1, j - 1))

            linear_base = linearise(reverse(Tuple(t.base)), Base.size(workspace))
            linear_offset = linearise(reverse(Tuple(t.offset)), Base.size(workspace))

            @inbounds vstorea!(Vec{vec_len, T}, pointer(workspace), value[i, j], linear_base + linear_offset - 1)
        end
    end
end

# -------------------
# InterleavedColMajor
# -------------------

struct InterleavedColMajor{T} <: LayoutBase{T} end

@inline function load(::Type{InterleavedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    res = MArray{Tuple{tile.size[1], tile.size[2]}, Complex{T}}(undef)

    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds res[i, j] = workspace[t.index[1] + 1, t.index[2] + 1]
        end
    end

    return res
end

@inline function store!(::Type{InterleavedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds workspace[t.index[1] + 1, t.index[2] + 1] = value[i, j]
        end
    end
end

# -------------------
# InterleavedRowMajor
# -------------------

struct InterleavedRowMajor{T} <: LayoutBase{T} end

@inline function load(::Type{InterleavedRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    res = MArray{Tuple{tile.size[1], tile.size[2]}, Complex{T}}(undef)

    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds res[i, j] = workspace[t.index[2] + 1, t.index[1] + 1]
        end
    end

    return res
end

@inline function store!(::Type{InterleavedRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for i = 1 : size[1]
        @unroll for j = 1 : size[2]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds workspace[t.index[2] + 1, t.index[1] + 1] = value[i, j]
        end
    end
end

# -------------
# SplitColMajor
# -------------

struct SplitColMajor{T} <: LayoutBase{T} end

@inline function physical_size(::Type{SplitColMajor{T}}, logical_size::NamedTuple) where {T}
    t = Tuple(logical_size)
    return (t..., 2)
end

@inline function load(::Type{SplitColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    res = MArray{Tuple{tile.size[1], tile.size[2]}, Complex{T}}(undef)

    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds res[i,j] = workspace[t.index[1] + 1, t.index[2] + 1, 1] + workspace[t.index[1] + 1, t.index[2] + 1, 2] * im
        end
    end

    return res
end

@inline function store!(::Type{SplitColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds workspace[t.index[1] + 1, t.index[2] + 1, 1] = value[i, j].re
            @inbounds workspace[t.index[1] + 1, t.index[2] + 1, 2] = value[i, j].im
        end
    end
end

# -------------
# SplitRowMajor
# -------------

struct SplitRowMajor{T} <: LayoutBase{T} end

@inline function physical_size(::Type{Padded{SplitRowMajor{T}, P}}, logical_size::NamedTuple) where {T, P}
    return (logical_size[2] + P, logical_size[1], 2)
end

@inline function load(::Type{SplitRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    res = MArray{Tuple{tile.size[1], tile.size[2]}, Complex{T}}(undef)

    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds res[i,j] = workspace[t.index[2] + 1, t.index[1] + 1, 1] + workspace[t.index[2] + 1, t.index[1] + 1, 2] * im
        end
    end

    return res
end

@inline function store!(::Type{SplitRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for i = 1 : tile.size[1]
        @unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))

            @inbounds workspace[t.index[2] + 1, t.index[1] + 1, 1] = value[i, j].re
            @inbounds workspace[t.index[2] + 1, t.index[1] + 1, 2] = value[i, j].im
        end
    end
end

end
