export Layout
module Layout

using CUDA
using LLVMLoopInfo: @loopinfo
using GemmKernels.Tiling
using GemmKernels: LocalArray, @immutable
using Base.Cartesian: @ntuple

# ---------------------
# Customise computation
# ---------------------

@inline threadblock_condition(layout_a, layout_b, block_i, block_j, block_k, block_tile) = true

# ----------------------
# Explicit vectorisation
# ----------------------

struct Vec{N, T} end

@noinline function throw_alignmenterror(ptr)
    @cuprintln "ERROR: AlignmentError: Pointer $ptr is not properly aligned for vectorized load/store"
    error()
end

@inline function checkalignment(ptr, alignment)
    checkalignment(Bool, ptr, alignment) || throw_alignmenterror(ptr)
    nothing
end

@inline function checkalignment(::Type{Bool}, ptr, alignment)
    Int(ptr) % alignment == 0
end

@inline @generated function vloada(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}, i::Integer = 1) where {N, T, AS}
    alignment = sizeof(T) * N

    return quote
        vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, VecElement{T}}, AS}, ptr)
        @boundscheck checkalignment(vec_ptr)
        return unsafe_load(vec_ptr, (i-1) ÷ N + 1, Val($alignment))
    end
end

@inline @generated function vstorea!(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS},
                                     x::NTuple{M,<:Any}) where {N, T, AS, M}
    alignment = sizeof(T) * N

    ex = quote end

    # we may be storing more values than we can using a single vectorized operation
    # (e.g., when types mismatch, storing 8 Float16s in a Float32 shared memory layout)
    for offset = 0:N:M-1
        append!(ex.args, (quote
            y = @ntuple $N j -> VecElement{T}(x[j+$offset].value)
            vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, VecElement{T}}, AS}, ptr)
            @boundscheck checkalignment(vec_ptr, $alignment)
            unsafe_store!(vec_ptr, y, $offset ÷ N + 1, Val($alignment))
        end).args)
    end

    return ex
end

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

end
