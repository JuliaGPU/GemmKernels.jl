module GETTLayout

using CUDA
using GemmKernels.Layout
using GemmKernels.Tiling
using KernelAbstractions.Extras: @unroll

@inline function sloada(::Type{Layout.Vec{NUMEL, T}}, workspace, offset::Int, strideOverExtent::Int) where {NUMEL, T}
    return ntuple(Val(NUMEL)) do i
        @inbounds VecElement{T}(workspace[offset + (i - 1) * strideOverExtent])
    end
end

@inline function sstorea!(::Type{Layout.Vec{NUMEL, T}}, workspace, val, offset::Int, strideOverExtent::Int) where {NUMEL, T}
    for i in 1 : NUMEL
        @inbounds workspace[offset + (i - 1) * strideOverExtent] = val[i].value
    end
end

abstract type GETTLayoutColMajor{T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent} <: Layout.AlignedColMajor{T} end

abstract type GETTLayoutRowMajor{T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent} <: Layout.AlignedRowMajor{T} end

@inline function Layout.load(
        ::Union{
            Type{GETTLayoutColMajor{T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent}},
            Type{GETTLayoutRowMajor{T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent}}
        }, workspace, tile::Tile{size}
    ) where {T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent, size}

    NUMEL = 16 ÷ sizeof(T)

    G1 = tile.base[1] + tile.offset[1]
    G2 = tile.base[2] + tile.offset[2]

    offset = 1

    @unroll for i in eachindex(dimensionsT1)
        stride_offset = (G1 ÷ divT1[i]) % modT[dimensionsT1[i]]
        offset += stride_offset * mulG1[i]
    end

    @unroll for i in eachindex(dimensionsT2)
        stride_offset = (G2 ÷ divT2[i]) % modT[dimensionsT2[i]]
        offset += stride_offset * mulG2[i]
    end

    if (isLoadOrStoreStrided == false)
        return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
    else
        return GETTLayout.sloada(Layout.Vec{NUMEL, T}, workspace, offset, strideOverExtent)
    end
end

@inline function Layout.store!(
        ::Union{
            Type{GETTLayoutColMajor{T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent}},
            Type{GETTLayoutRowMajor{T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent}}
        }, workspace, value, tile::Tile{size}
    ) where {T, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent, size}

    NUMEL = 16 ÷ sizeof(T)

    G1 = tile.base[1] + tile.offset[1]
    G2 = tile.base[2] + tile.offset[2]

    offset = 1

    @unroll for i in eachindex(dimensionsT1)
        stride_offset = (G1 ÷ divT1[i]) % modT[dimensionsT1[i]]
        offset += stride_offset * mulG1[i]
    end

    @unroll for i in eachindex(dimensionsT2)
        stride_offset = (G2 ÷ divT2[i]) % modT[dimensionsT2[i]]
        offset += stride_offset * mulG2[i]
    end

    if (isLoadOrStoreStrided == false)
        return Layout.vstorea!(Layout.Vec{NUMEL, T}, pointer(workspace), value, offset)
    else
        GETTLayout.sstorea!(Layout.Vec{NUMEL, T}, workspace, value, offset, strideOverExtent)
    end
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
    # e.g. for D[A, B, C] = A[B, D, A] * B[D, C] this will be (1, 3), since B and A belong to the
    # M stride.
    dimensionsT1 = Tuple(x for x in dimensions[1])
    # → analogous, T2_stride will be equal to (2,) for the above example, since D belongs to the
    # K stride.
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
    modT = Tuple(x for x in extent)


    # 4. Precompute the multiplicative terms used to calculate the GEMM stride offsets.
    # → offset += T1_stride_offset * mulG1[i]
    mulG1 = Vector{Int}(undef, length(dimensionsT1))
    for (idx, stride_idx) in enumerate(dimensionsT1)
        mulG1[idx] = 1
        for j = 1 : (stride_idx - 1) 
            mulG1[idx] *= extent[j]
        end
    end
    mulG1 = Tuple(x for x in mulG1)

    # 4b. Do the same for mulG2.
    mulG2 = Vector{Int}(undef, length(dimensionsT2))
    for (idx, stride_idx) in enumerate(dimensionsT2)
        mulG2[idx] = 1
        for j = 1 : (stride_idx - 1) 
            mulG2[idx] *= extent[j]
        end
    end
    mulG2 = Tuple(x for x in mulG2)

    # 5. Convert the Bool to an Int.
    isLoadOrStoreStrided = Int(isLoadOrStoreStrided)

    # 5.b If the load or store is strided, then precompute the size of the dimensions to stride 
    # over.
    strideOverExtent = 1
    if (isnothing(strideOver) == false && isLoadOrStoreStrided == true)
        for stride_idx in strideOver
            strideOverExtent *= extent[stride_idx]
        end
    end

    return (
        dimensionsT1, dimensionsT2,
        divT1, divT2,
        modT,
        mulG1, mulG2,
        isLoadOrStoreStrided, strideOverExtent,
    )
end

function createGETTLayout(
    DT::DataType,
    extent::Vector{Int},
    dimensions::Tuple{Vector{Int}, Vector{Int}},
    isColMajor::Bool,
    isLoadOrStoreStrided::Bool,
    strideOver::Union{Vector{Int}, Nothing} = nothing,
)
    (
        dimensionsT1, dimensionsT2,
        divT1, divT2,
        modT,
        mulG1, mulG2,
        isLoadOrStoreStrided, strideOverExtent
    ) = precomputeGETTLayoutConstants(extent, dimensions, isLoadOrStoreStrided, strideOver)

    if (isColMajor == true)
        return GETTLayoutColMajor{DT, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent}
    else
        return GETTLayoutRowMajor{DT, dimensionsT1, divT1, mulG1, dimensionsT2, divT2, mulG2, modT, isLoadOrStoreStrided, strideOverExtent}
    end
end

end