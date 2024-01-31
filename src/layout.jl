export Layout
module Layout

using CUDA
using LLVMLoopInfo: @loopinfo
using Base.Cartesian: @ntuple
using GemmKernels.Tiling
using GemmKernels: LocalArray, @immutable, b, tid, @unrolled, variadic, constant, vstorea!, Vec

# ---------------------
# Customise computation
# ---------------------

@inline threadblock_condition(layout_a, layout_b, block_i, block_j, block_k, block_tile) = true

# -----------
# Layout base
# -----------

abstract type LayoutBase{T} end

@inline eltype(::Type{<:LayoutBase{T}}) where {T} = T
@inline physical_size(::Type{<:LayoutBase{T}}, logical_size::NamedTuple) where {T} = Tuple(logical_size)

# ----
# Zero
# ----

abstract type Zero{T} <: LayoutBase{T} end

@inline function load(::Type{<:Zero{T}}, workspace, tile::Tile{size}) where {T, size}
    N = size[1] * size[2]
    return ntuple(i -> VecElement{T}(zero(T)), Val(N))
end

@inline store!(::Type{<:Zero{T}}, workspace, value, tile::Tile) where {T} = return

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
@inline Base.@propagate_inbounds load(::Type{<:Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = load(L, workspace, tile)
@inline Base.@propagate_inbounds store!(::Type{<:Padded{L, P}}, workspace, value, tile::Tile) where {L, P} = store!(L, workspace, value, tile::Tile)

# --------
# ColMajor
# --------

abstract type ColMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{<:Padded{ColMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[1] + P, logical_size[2])

@inline fragtype(::Type{<:ColMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], T}

@inline Base.@propagate_inbounds function load(::Type{<:ColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> VecElement(zero(T)), tile.size[1] * tile.size[2])

    @loopinfo unroll for j = 1 : tile.size[2]
        @loopinfo unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            if checkbounds(Bool, workspace, t.index[1] + 1, t.index[2] + 1)
                @inbounds val = workspace[t.index[1] + 1, t.index[2] + 1]
                @inbounds @immutable x[(i - 1) * tile.size[2] + j] = VecElement(val)
            end
        end
    end
    return x
end

@inline Base.@propagate_inbounds function store!(::Type{<:ColMajor{T}}, workspace, values, tile::Tile{size}) where {T, size}
    @loopinfo unroll for j = 1 : tile.size[2]
        @loopinfo unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            @inbounds val = values[(i - 1) * tile.size[2] + j]
            if checkbounds(Bool, workspace, t.index[1] + 1, t.index[2] + 1)
                @inbounds workspace[t.index[1] + 1, t.index[2] + 1] = val.value
            end
        end
    end

    return
end

# --------
# RowMajor
# --------

abstract type RowMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{<:Padded{RowMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[1], logical_size[2] + P)

@inline fragtype(::Type{<:RowMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], T}

@inline Base.@propagate_inbounds function load(::Type{<:RowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> VecElement(zero(T)), tile.size[1] * tile.size[2])

    @loopinfo unroll for i = 1 : tile.size[1]
        @loopinfo unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            if checkbounds(Bool, workspace, t.index[2] + 1, t.index[1] + 1)
                @inbounds val = workspace[t.index[2] + 1, t.index[1] + 1]
                @inbounds @immutable x[(i - 1) * tile.size[2] + j] = VecElement(val)
            end
        end
    end

    return x
end

@inline Base.@propagate_inbounds function store!(::Type{<:RowMajor{T}}, workspace, values, tile::Tile{size}) where {T, size}
    @loopinfo unroll for i = 1 : tile.size[1]
        @loopinfo unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            @inbounds val = value[(i - 1) * tile.size[2] + j]
            if checkbounds(Bool, workspace, t.index[2] + 1, t.index[1] + 1)
                @inbounds workspace[t.index[2] + 1, t.index[1] + 1] = val
            end
        end
    end

    return
end

# ---------------------
# UnsafeAlignedColMajor
# ---------------------

# assumes that memory is aligned, and that tiles exactly cover the workspace

abstract type UnsafeAlignedColMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{<:Padded{UnsafeAlignedColMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[1] + P, logical_size[2])

@inline fragtype(::Type{<:UnsafeAlignedColMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], VecElement{T}}

@inline Base.@propagate_inbounds function load(::Type{<:UnsafeAlignedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    N = size[1] * size[2]

    linear_base = linearise(tile.base, Base.size(workspace))
    linear_offset = linearise(tile.offset, Base.size(workspace))
    linear_idx = linear_base + linear_offset - 1

    @boundscheck checkbounds(workspace, linear_idx:(linear_idx+N-1))
    return vloada(Vec{N, T}, pointer(workspace, linear_idx))
end

@inline Base.@propagate_inbounds function store!(::Type{<:UnsafeAlignedColMajor{T}}, workspace, values, tile::Tile{size}) where {T, size}
    N = size[1] * size[2]

    linear_base = linearise(tile.base, Base.size(workspace))
    linear_offset = linearise(tile.offset, Base.size(workspace))
    linear_idx = linear_base + linear_offset - 1

    @boundscheck checkbounds(workspace, linear_idx:(linear_idx+length(values)-1))
    vstorea!(Vec{N, T}, pointer(workspace, linear_idx), values)
    return
end

# --------
# Diagonal
# --------

abstract type Diagonal{T} <: LayoutBase{T} end

@inline Base.@propagate_inbounds function load(::Type{<:Diagonal{T}}, workspace, tile::Tile{size}) where {T, size}
    N = size[1] * size[2]

    # The row index is given by t.index[1] + (k - 1), the column index is given by t.index[2] (0-based).
    # Only load on the diagonal, i.e. if row and column are equal.
    # Note that t.index[2] is 0-based, so we need to add 1 before loading from workspace.
    return ntuple(k -> VecElement{Float16}(tile.index[1] + k - 1 == tile.index[2] ? workspace[tile.index[2] + 1] : 0), Val(8))
end

@inline threadblock_condition(layout_a::Type{<:Diagonal{T}}, layout_b, block_i, block_j, block_k, block_tile) where {T} = abs(block_i - block_k) <= block_tile.size.K

# ---------------------
# UnsafeAlignedRowMajor
# ---------------------

# assumes that memory is aligned, and that tiles exactly cover the workspace

abstract type UnsafeAlignedRowMajor{T} <: LayoutBase{T} end

@inline physical_size(::Type{<:Padded{UnsafeAlignedRowMajor{T}, P}}, logical_size::NamedTuple) where {T, P} = (logical_size[2] + P, logical_size[1])

@inline fragtype(::Type{<:UnsafeAlignedRowMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], VecElement{T}}

@inline Base.@propagate_inbounds function load(::Type{<:UnsafeAlignedRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    N = size[1] * size[2]

    linear_base = linearise(reverse(Tuple(tile.base)), Base.size(workspace))
    linear_offset = linearise(reverse(Tuple(tile.offset)), Base.size(workspace))
    linear_idx = linear_base + linear_offset - 1

    @boundscheck checkbounds(workspace, linear_idx:(linear_idx+N-1))
    return vloada(Vec{N, T}, pointer(workspace, linear_idx))
end

@inline Base.@propagate_inbounds function store!(::Type{<:UnsafeAlignedRowMajor{T}}, workspace, values, tile::Tile{size}) where {T, size}
    N = size[1] * size[2]

    linear_base = linearise(reverse(Tuple(tile.base)), Base.size(workspace))
    linear_offset = linearise(reverse(Tuple(tile.offset)), Base.size(workspace))

    linear_idx = linear_base + linear_offset - 1
    @boundscheck checkbounds(workspace, linear_idx:(linear_idx+length(values)-1))
    vstorea!(Vec{N, T}, pointer(workspace, linear_idx), values)
end

# -------------------
# InterleavedColMajor
# -------------------

abstract type InterleavedColMajor{T} <: LayoutBase{T} end

@inline fragtype(::Type{<:InterleavedColMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], Complex{T}}

@inline Base.@propagate_inbounds function load(::Type{<:InterleavedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @loopinfo unroll for j = 1 : tile.size[2]
        @loopinfo unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            val = workspace[t.index[1] + 1, t.index[2] + 1]
            @immutable x[(i - 1) * tile.size[2] + j] = val
        end
    end

    return x
end

@inline Base.@propagate_inbounds function store!(::Type{<:InterleavedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @loopinfo unroll for j = 1 : tile.size[2]
        @loopinfo unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            workspace[t.index[1] + 1, t.index[2] + 1] = val
        end
    end
end

# -------------------
# InterleavedRowMajor
# -------------------

abstract type InterleavedRowMajor{T} <: LayoutBase{T} end

@inline fragtype(::Type{<:InterleavedRowMajor{T}}, tile_size::NamedTuple) where {T} = NTuple{tile_size[1] * tile_size[2], Complex{T}}

@inline Base.@propagate_inbounds function load(::Type{<:InterleavedRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @loopinfo unroll for i = 1 : tile.size[1]
        @loopinfo unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            val = workspace[t.index[2] + 1, t.index[1] + 1]
            @immutable x[(i - 1) * tile.size[2] + j] = val
        end
    end

    return x
end

@inline Base.@propagate_inbounds function store!(::Type{<:InterleavedRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @loopinfo unroll for i = 1 : tile.size[1]
        @loopinfo unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            workspace[t.index[2] + 1, t.index[1] + 1] = val
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

@inline Base.@propagate_inbounds function load(::Type{<:SplitColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @loopinfo unroll for j = 1 : tile.size[2]
        @loopinfo unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            val = workspace[t.index[1] + 1, t.index[2] + 1, 1] + im *
                  workspace[t.index[1] + 1, t.index[2] + 1, 2]
            @immutable x[(i - 1) * tile.size[2] + j] = val
        end
    end

    return x
end

@inline Base.@propagate_inbounds function store!(::Type{<:SplitColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @loopinfo unroll for j = 1 : tile.size[2]
        @loopinfo unroll for i = 1 : tile.size[1]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            workspace[t.index[1] + 1, t.index[2] + 1, 1] = val.re
            workspace[t.index[1] + 1, t.index[2] + 1, 2] = val.im
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

@inline Base.@propagate_inbounds function load(::Type{<:SplitRowMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    x = ntuple(i -> zero(Complex{T}), tile.size[1] * tile.size[2])

    @loopinfo unroll for i = 1 : tile.size[1]
        @loopinfo unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            val = workspace[t.index[2] + 1, t.index[1] + 1, 1] + im *
                  workspace[t.index[2] + 1, t.index[1] + 1, 2]
            @immutable x[(i - 1) * tile.size[2] + j] = val
        end
    end

    return x
end

@inline Base.@propagate_inbounds function store!(::Type{<:SplitRowMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @loopinfo unroll for i = 1 : tile.size[1]
        @loopinfo unroll for j = 1 : tile.size[2]
            t = translate_offset(tile, (i - 1, j - 1))
            val = value[(i - 1) * tile.size[2] + j]
            workspace[t.index[2] + 1, t.index[1] + 1, 1] = val.re
            workspace[t.index[2] + 1, t.index[1] + 1, 2] = val.im
        end
    end
end

# ----------------------
# Volta swizzled layouts
# ----------------------

# TODO: Generalise this to:
# - Other configurations than NN
# - Other architectures
# - Other tile sizes?

abstract type SwizzledLayout{T} <: LayoutBase{T} end

abstract type VoltaSwizzledOperandA{T} <: SwizzledLayout{T} end
abstract type VoltaSwizzledOperandB{T} <: SwizzledLayout{T} end

@inline function swizzle(::Type{VoltaSwizzledOperandA{Float16}}, m, k)
    # TODO: Generalise this to larger extents?
    # m : 7 bits
    # k : 5 bits
    offset = b(k, 0, 0) +
             b(k, 1, 1) +
             (b(k, 3, 2) ⊻ b(m, 2, 2)) +
             (b(k, 4, 3) ⊻ b(m, 3, 3) ⊻ b(m, 4, 3)) +
             b(m, 0, 4) +
             b(m, 1, 5) +
             b(m, 4, 6) +
             b(m, 5, 7) +
             b(m, 6, 8) +
             b(k, 2, 9) +
             b(k, 3, 10) +
             b(k, 4, 11) +
             b(k, 5, 12)

    return offset
end

@inline function swizzle(::Type{VoltaSwizzledOperandB{Float16}}, k, n)
    # TODO: Generalise this to larger extents?
    # k: 5 bits
    # n: 8 bits
    offset = b(n, 0, 0) +
             b(n, 1, 1) +
             b(n, 2, 2) +
             (b(n, 3, 3) ⊻ b(k, 0, 3)) +
             (b(n, 4, 4) ⊻ b(k, 1, 4)) +
             b(n, 5, 5) +
             b(n, 6, 6) +
             b(n, 7, 7) +
             b(n, 3, 8) +
             b(n, 4, 9) +
             b(k, 2, 10) +
             b(k, 3, 11) +
             b(k, 4, 12) +
             b(k, 5, 13)

    return offset
end

@inline Base.@propagate_inbounds function store!(L::Type{VoltaSwizzledOperandA{Float16}}, workspace, value, tile::Tile)
    # TODO: Boundschecking logic.
    m = variadic(tile.base.M) + constant(tile.offset.M)
    k = variadic(tile.base.K) + constant(tile.offset.K)

    @inbounds vstorea!(Vec{4, Float16}, workspace, Layout.swizzle(L, m, k), value)
end

@inline Base.@propagate_inbounds function store!(L::Type{VoltaSwizzledOperandB{Float16}}, workspace, value, tile::Tile)
    # TODO: Boundchecking logic.
    k = variadic(tile.base.K) + constant(tile.offset.K)
    n = variadic(tile.base.N) + constant(tile.offset.N)

    @inbounds vstorea!(Vec{8, Float16}, workspace, Layout.swizzle(L, k, n), value)
end

end
