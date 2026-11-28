module GETTLayout

using CUDA
using GemmKernels.Layout
using GemmKernels.Tiling
using LLVMLoopInfo: @loopinfo

@inline function sloada(::Type{Layout.Vec{NUMEL, T}}, workspace, offset::Int,
                        strideOverExtent::Int) where {NUMEL, T}
    return ntuple(Val(NUMEL)) do i
        @inbounds VecElement{T}(workspace[offset + (i - 1) * strideOverExtent])
    end
end

@inline function sstorea!(::Type{Layout.Vec{NUMEL, T}}, workspace, val, offset::Int,
                          strideOverExtent::Int) where {NUMEL, T}
    for i in 1 : NUMEL
        @inbounds workspace[offset + (i - 1) * strideOverExtent] = val[i].value
    end
end

abstract type GETTLayoutColMajor{T, divT1, modT1, strides1, divT2, modT2, strides2,
                                 isLoadOrStoreStrided, strideOverExtent
                                } <: Layout.UnsafeAlignedColMajor{T} end

abstract type GETTLayoutRowMajor{T, divT1, modT1, strides1, divT2, modT2, strides2,
                                 isLoadOrStoreStrided, strideOverExtent
                                } <: Layout.UnsafeAlignedRowMajor{T} end

Base.@propagate_inbounds @inline function Layout.load(
        ::Union{
            Type{GETTLayoutColMajor{T, divT1, modT1, strides1, divT2, modT2, strides2,
                                    isLoadOrStoreStrided, strideOverExtent}},
            Type{GETTLayoutRowMajor{T, divT1, modT1, strides1, divT2, modT2, strides2,
                                    isLoadOrStoreStrided, strideOverExtent}}
        }, workspace, tile::Tile{size}
    ) where {T, divT1, modT1, strides1, divT2, modT2, strides2,
             isLoadOrStoreStrided, strideOverExtent, size}

    NUMEL = 16 ÷ sizeof(T)

    G1 = tile.base[1] + tile.offset[1]
    G2 = tile.base[2] + tile.offset[2]

    offset = 1

    @loopinfo unroll for i in eachindex(divT1)
        stride_offset = (G1 ÷ divT1[i]) % modT1[i]
        offset += stride_offset * strides1[i]
    end

    @loopinfo unroll for i in eachindex(divT2)
        stride_offset = (G2 ÷ divT2[i]) % modT2[i]
        offset += stride_offset * strides2[i]
    end

    if (isLoadOrStoreStrided == false)
        return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
    else
        return GETTLayout.sloada(Layout.Vec{NUMEL, T}, workspace, offset, strideOverExtent)
    end
end

Base.@propagate_inbounds @inline function Layout.store!(
        ::Union{
            Type{GETTLayoutColMajor{T, divT1, modT1, strides1, divT2, modT2, strides2,
                                    isLoadOrStoreStrided, strideOverExtent}},
            Type{GETTLayoutRowMajor{T, divT1, modT1, strides1, divT2, modT2, strides2,
                                    isLoadOrStoreStrided, strideOverExtent}}
        }, workspace, value, tile::Tile{size}
    ) where {T, divT1, modT1, strides1, divT2, modT2, strides2,
             isLoadOrStoreStrided, strideOverExtent, size}

    NUMEL = 16 ÷ sizeof(T)

    G1 = tile.base[1] + tile.offset[1]
    G2 = tile.base[2] + tile.offset[2]

    offset = 1

    @loopinfo unroll for i in eachindex(divT1)
        stride_offset = (G1 ÷ divT1[i]) % modT1[i]
        offset += stride_offset * strides1[i]
    end

    @loopinfo unroll for i in eachindex(divT2)
        stride_offset = (G2 ÷ divT2[i]) % modT2[i]
        offset += stride_offset * strides2[i]
    end

    if (isLoadOrStoreStrided == false)
        return Layout.vstorea!(Layout.Vec{NUMEL, T}, pointer(workspace), value, offset)
    else
        GETTLayout.sstorea!(Layout.Vec{NUMEL, T}, workspace, value, offset, strideOverExtent)
    end
end


# Asynchronous copies (see Layout.copy_async!): only non-strided GETT layouts store the
# elements of a thread's tile contiguously.
Layout.contiguous_tiles(
        ::Union{
            Type{GETTLayoutColMajor{T, divT1, modT1, strides1, divT2, modT2, strides2, isLoadOrStoreStrided, strideOverExtent}},
            Type{GETTLayoutRowMajor{T, divT1, modT1, strides1, divT2, modT2, strides2, isLoadOrStoreStrided, strideOverExtent}}
        }
    ) where {T, divT1, modT1, strides1, divT2, modT2, strides2, isLoadOrStoreStrided, strideOverExtent} =
    isLoadOrStoreStrided == false

Base.@propagate_inbounds @inline function Layout.tile_pointer(
        ::Union{
            Type{GETTLayoutColMajor{T, divT1, modT1, strides1, divT2, modT2, strides2, isLoadOrStoreStrided, strideOverExtent}},
            Type{GETTLayoutRowMajor{T, divT1, modT1, strides1, divT2, modT2, strides2, isLoadOrStoreStrided, strideOverExtent}}
        },
        workspace, tile::Tile
    ) where {T, divT1, modT1, strides1, divT2, modT2, strides2, isLoadOrStoreStrided, strideOverExtent}

    G1 = tile.base[1] + tile.offset[1]
    G2 = tile.base[2] + tile.offset[2]

    offset = 1

    @loopinfo unroll for i in eachindex(divT1)
        stride_offset = (G1 ÷ divT1[i]) % modT1[i]
        offset += stride_offset * strides1[i]
    end

    @loopinfo unroll for i in eachindex(divT2)
        stride_offset = (G2 ÷ divT2[i]) % modT2[i]
        offset += stride_offset * strides2[i]
    end

    return pointer(workspace) + (offset - 1) * sizeof(T)
end

# TODO: Write test. One example given below.
# precomputeGETTLayoutConstants([16, 512, 32], ([1, 3], [2]))
# result: ((1, 3), (2,), (1, 16), (1,), (16, 512, 32), (1, 8192), (16,))

export precomputeGETTLayoutConstants
function precomputeGETTLayoutConstants(
    extent::Vector{Int},
    dimensions::Tuple{Vector{Int}, Vector{Int}},
    isLoadOrStoreStrided::Bool,
    strideOver::Union{Vector{Int}, Nothing} = nothing,
)
    # 1. Convert the tensor strides from Tuple{Vector{Int}, Vector{int}}  to two separate
    # Tuple{Int, Int, ...}.

    # → For the A matrix this will contain the tensor strides corresponding to the M stride.
    # e.g. for D[A, B, C] = A[B, D, A] * B[D, C] this will be (1, 3), since B and A belong
    # to the M stride.
    dimensionsT1 = Tuple(x for x in dimensions[1])
    # → analogous, T2_stride will be equal to (2,) for the above example, since D belongs to
    # the K stride.
    dimensionsT2 = Tuple(x for x in dimensions[2])


    # 2. Precompute the divisors used to calculate the tensor stride offsets.
    # → T1_stride_offset = (M ÷ divT1[i]) % T1_mod[i]
    divT1 = Vector{Int}(undef, length(dimensionsT1))
    div = 1
    for (idx, stride_idx) in enumerate(dimensionsT1)
        divT1[idx] = div
        div *= extent[stride_idx]
    end
    divT1 = Tuple(x for x in divT1)

    # 2b. Do the same for T2_stride.
    divT2 = Vector{Int}(undef, length(dimensionsT2))
    div = 1
    for (idx, stride_idx) in enumerate(dimensionsT2)
        divT2[idx] = div
        div *= extent[stride_idx]
    end
    divT2 = Tuple(x for x in divT2)


    # 3. Precompute the moduli used to calculate the tensor stride offsets.
    # These are simply the extents of the tensor. Again, converted to Tuple{Int, Int, ...}.
    modT1 = Tuple(extent[dimensionsT1[i]] for i in eachindex(dimensionsT1))
    modT2 = Tuple(extent[dimensionsT2[i]] for i in eachindex(dimensionsT2))


    # 4. Precompute the multiplicative terms used to calculate the GEMM stride offsets.
    # → offset += T1_stride_offset * strides1[i]
    strides1 = Vector{Int}(undef, length(dimensionsT1))
    for (idx, stride_idx) in enumerate(dimensionsT1)
        strides1[idx] = 1
        for j = 1 : (stride_idx - 1)
            strides1[idx] *= extent[j]
        end
    end
    strides1 = Tuple(x for x in strides1)

    # 4b. Do the same for strides2.
    strides2 = Vector{Int}(undef, length(dimensionsT2))
    for (idx, stride_idx) in enumerate(dimensionsT2)
        strides2[idx] = 1
        for j = 1 : (stride_idx - 1)
            strides2[idx] *= extent[j]
        end
    end
    strides2 = Tuple(x for x in strides2)

    # 5. Convert the Bool to an Int.
    isLoadOrStoreStrided = Int(isLoadOrStoreStrided)

    # 5.b If the load or store is strided, then precompute the size of the dimensions to
    # stride over.
    strideOverExtent = 1
    if (isnothing(strideOver) == false && isLoadOrStoreStrided == true)
        for stride_idx in strideOver
            strideOverExtent *= extent[stride_idx]
        end
    end

    return (
        divT1, divT2,
        modT1, modT2,
        strides1, strides2,
        isLoadOrStoreStrided, strideOverExtent,
    )
end

# Leaner address arithmetic (opt-in, see `createGETTLayout`'s `indexing`). The layouts above
# compute every element offset as a sum over the tensor modes of
# `(G ÷ divisor) % extent * stride`, in signed 64-bit arithmetic. The GPU has no 64-bit
# multiply, so each division or modulo by a constant takes several instructions, and in
# kernels like TCCG TC22 they make up most of the main loop. These layouts compute the same
# offsets in unsigned 32-bit arithmetic (for tensors with fewer than 2^31 elements) and without
# the modulo of the last mode of each dimension, which is redundant: G is smaller than the
# product of the extents, so the last quotient is smaller than its extent. The original layouts
# stay the default, as the generated code is sensitive to such changes: on an RTX 6000 Ada,
# these made 8 of 9 TCCG contractions 2-27% faster but one 7% slower, so the tuner chooses.
# (Unsigned 64-bit arithmetic was tried too: it was erratic, up to 25x slower.)
abstract type GETTLayoutLeanColMajor{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                                     isLoadOrStoreStrided, strideOverExtent
                                    } <: Layout.UnsafeAlignedColMajor{T} end

abstract type GETTLayoutLeanRowMajor{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                                     isLoadOrStoreStrided, strideOverExtent
                                    } <: Layout.UnsafeAlignedRowMajor{T} end

const GETTLayoutLean{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                     isLoadOrStoreStrided, strideOverExtent} = Union{
    Type{GETTLayoutLeanColMajor{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                                isLoadOrStoreStrided, strideOverExtent}},
    Type{GETTLayoutLeanRowMajor{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                                isLoadOrStoreStrided, strideOverExtent}}}

# The 1-based element offset of logical position (G1, G2), as straight-line code in IT.
@generated function lean_offset(::Type{IT}, G1, G2, ::Val{divT1}, ::Val{modT1}, ::Val{strides1},
                                ::Val{divT2}, ::Val{modT2}, ::Val{strides2}) where
                                {IT, divT1, modT1, strides1, divT2, modT2, strides2}
    ex = quote
        g1 = G1 % $IT
        g2 = G2 % $IT
        offset = one($IT)
    end
    for (g, divT, modT, strides) in ((:g1, divT1, modT1, strides1), (:g2, divT2, modT2, strides2))
        n = length(divT)
        for i in 1:n
            q = divT[i] == 1 ? g : :(div($g, $(IT(divT[i]))))
            t = i == n ? q : :(rem($q, $(IT(modT[i]))))
            term = strides[i] == 1 ? t : :($t * $(IT(strides[i])))
            push!(ex.args, :(offset += $term))
        end
    end
    push!(ex.args, :(return Int(offset)))
    return ex
end

@inline lean_offset(::GETTLayoutLean{T, IT, divT1, modT1, strides1, divT2, modT2, strides2},
                    tile) where {T, IT, divT1, modT1, strides1, divT2, modT2, strides2} =
    lean_offset(IT, tile.base[1] + tile.offset[1], tile.base[2] + tile.offset[2],
                Val(divT1), Val(modT1), Val(strides1), Val(divT2), Val(modT2), Val(strides2))

Base.@propagate_inbounds @inline function Layout.load(
        L::GETTLayoutLean{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                          isLoadOrStoreStrided, strideOverExtent},
        workspace, tile::Tile{size}
    ) where {T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
             isLoadOrStoreStrided, strideOverExtent, size}
    NUMEL = size[1] * size[2]
    offset = lean_offset(L, tile)
    if (isLoadOrStoreStrided == false)
        return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
    else
        return GETTLayout.sloada(Layout.Vec{NUMEL, T}, workspace, offset, strideOverExtent)
    end
end

Base.@propagate_inbounds @inline function Layout.store!(
        L::GETTLayoutLean{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                          isLoadOrStoreStrided, strideOverExtent},
        workspace, value, tile::Tile{size}
    ) where {T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
             isLoadOrStoreStrided, strideOverExtent, size}
    NUMEL = size[1] * size[2]
    offset = lean_offset(L, tile)
    if (isLoadOrStoreStrided == false)
        return Layout.vstorea!(Layout.Vec{NUMEL, T}, pointer(workspace), value, offset)
    else
        GETTLayout.sstorea!(Layout.Vec{NUMEL, T}, workspace, value, offset, strideOverExtent)
    end
end

Layout.contiguous_tiles(
        ::GETTLayoutLean{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                         isLoadOrStoreStrided, strideOverExtent}
    ) where {T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
             isLoadOrStoreStrided, strideOverExtent} =
    isLoadOrStoreStrided == false

Base.@propagate_inbounds @inline Layout.tile_pointer(
        L::GETTLayoutLean{T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                          isLoadOrStoreStrided, strideOverExtent},
        workspace, tile::Tile
    ) where {T, IT, divT1, modT1, strides1, divT2, modT2, strides2,
             isLoadOrStoreStrided, strideOverExtent} =
    pointer(workspace) + (lean_offset(L, tile) - 1) * sizeof(T)

# Partial edge tiles (opt-in, see `createGETTLayout`'s `edge`). The layouts above require the
# GEMM dimensions to be multiples of the block shape, so the tensors are padded: whole modes,
# which for a 128-wide tile over two modes of 72 means padding one to 80, i.e. 11% more work
# per dimension. GETTLayoutMasked instead lets the last row and column of thread blocks extend
# past the edge: it wraps a GETT layout, loads zeros for the tiles of each thread whose
# logical coordinate lies beyond `bound1` (first dimension) or `bound2` (second dimension),
# and skips their stores. A bound of 0 means that dimension is not masked. A thread's tile
# never straddles a bound: `setUpGETTKernel` requires the bounds to be multiples of the tile
# sizes of the copies.
abstract type GETTLayoutMasked{T, L, bound1, bound2} <: Layout.LayoutBase{T} end

@inline function in_bounds(::Type{GETTLayoutMasked{T, L, bound1, bound2}}, tile) where {T, L, bound1, bound2}
    (bound1 == 0 || tile.base[1] + tile.offset[1] < bound1) &&
        (bound2 == 0 || tile.base[2] + tile.offset[2] < bound2)
end

Base.@propagate_inbounds @inline function Layout.load(
        M::Type{GETTLayoutMasked{T, L, bound1, bound2}}, workspace, tile::Tile{size}
    ) where {T, L, bound1, bound2, size}
    NUMEL = size[1] * size[2]
    if in_bounds(M, tile)
        return Layout.load(L, workspace, tile)
    else
        return ntuple(_ -> VecElement{T}(zero(T)), Val(NUMEL))
    end
end

Base.@propagate_inbounds @inline function Layout.store!(
        M::Type{GETTLayoutMasked{T, L, bound1, bound2}}, workspace, value, tile::Tile
    ) where {T, L, bound1, bound2}
    if in_bounds(M, tile)
        Layout.store!(L, workspace, value, tile)
    end
    return
end

# the register fragments of the kernels that copy through registers
@inline Layout.fragtype(::Type{GETTLayoutMasked{T, L, bound1, bound2}}, tile_size::NamedTuple) where {T, L, bound1, bound2} =
    Layout.fragtype(L, tile_size)

# Asynchronous copies: the tiles within the edges are copied with cp.async, and those beyond
# them get zeros with a plain store to shared memory, which the asynchronous-copy kernel
# allows (see its copy_tile: the stage buffer is only read after a later barrier).
Layout.contiguous_tiles(::Type{GETTLayoutMasked{T, L, bound1, bound2}}) where {T, L, bound1, bound2} =
    Layout.contiguous_tiles(L)

Base.@propagate_inbounds @inline function Layout.copy_async!(
        M::Type{GETTLayoutMasked{T, L, bound1, bound2}}, shared_layout,
        shared_workspace, global_workspace, shared_tile::Tile{size}, global_tile::Tile
    ) where {T, L, bound1, bound2, size}
    if in_bounds(M, global_tile)
        Layout.copy_async!(L, shared_layout, shared_workspace, global_workspace, shared_tile, global_tile)
    else
        zeros = ntuple(_ -> VecElement{T}(zero(T)), Val(size[1] * size[2]))
        Layout.store!(shared_layout, shared_workspace, zeros, shared_tile)
    end
    return
end

# The address arithmetic of the GETT layouts (see GETTLayoutLean):
#   :default   the original layouts
#   :lean32    unsigned 32-bit, without the redundant modulos (tensors < 2^31 elements)
const GETT_INDEXING = (:default, :lean32)

function createGETTLayout(
    DT::DataType,
    extent::Vector{Int},
    dimensions::Tuple{Vector{Int}, Vector{Int}},
    isColMajor::Bool,
    isLoadOrStoreStrided::Bool,
    strideOver::Union{Vector{Int}, Nothing} = nothing;
    indexing::Symbol = :default,
    bounds::Tuple{Int, Int} = (0, 0),
)
    (
        divT1, divT2,
        modT1, modT2,
        strides1, strides2,
        isLoadOrStoreStrided, strideOverExtent
    ) = precomputeGETTLayoutConstants(extent, dimensions, isLoadOrStoreStrided, strideOver)

    L = if indexing == :default
        if (isColMajor == true)
            GETTLayoutColMajor{DT, divT1, modT1, strides1, divT2, modT2, strides2,
                               isLoadOrStoreStrided, strideOverExtent}
        else
            GETTLayoutRowMajor{DT, divT1, modT1, strides1, divT2, modT2, strides2,
                               isLoadOrStoreStrided, strideOverExtent}
        end
    else
        indexing in GETT_INDEXING ||
            throw(ArgumentError("indexing must be one of $(GETT_INDEXING), but is :$indexing"))
        IT = UInt32
        if (isColMajor == true)
            GETTLayoutLeanColMajor{DT, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                                   isLoadOrStoreStrided, strideOverExtent}
        else
            GETTLayoutLeanRowMajor{DT, IT, divT1, modT1, strides1, divT2, modT2, strides2,
                                   isLoadOrStoreStrided, strideOverExtent}
        end
    end

    # partial edge tiles: mask the dimensions with a non-zero bound
    return bounds == (0, 0) ? L : GETTLayoutMasked{DT, L, bounds[1], bounds[2]}
end

end
