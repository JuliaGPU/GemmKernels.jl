export TensorLayout
module TensorLayout

using CUDA
using GemmKernels.Layout
using GemmKernels.Tiling
using KernelAbstractions.Extras: @unroll

@inline function sloada(::Type{Layout.Vec{NUMEL, T}}, workspace, offset::Int, strided_over_size::Int) where {NUMEL, T}
    return ntuple(Val(NUMEL)) do i
        @inbounds VecElement{T}(workspace[offset + (i - 1) * strided_over_size])
    end
end

@inline function sstorea!(::Type{Layout.Vec{NUMEL, T}}, workspace, val, offset::Int, strided_over_size::Int) where {NUMEL, T}
    for i in 1 : NUMEL
        @inbounds workspace[offset + (i - 1) * strided_over_size] = val[i].value
    end
end

abstract type TensorLayoutColMajor{T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size} <: Layout.AlignedColMajor{T} end

abstract type TensorLayoutRowMajor{T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size} <: Layout.AlignedRowMajor{T} end

@inline function Layout.load(
        ::Union{
            Type{TensorLayoutColMajor{T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size}},
            Type{TensorLayoutRowMajor{T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size}}
        }, workspace, tile::Tile{size}
    ) where {T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size, size}

    NUMEL = 16 ÷ sizeof(T)

    M = tile.base[1] + tile.offset[1]
    K = tile.base[2] + tile.offset[2]

    offset = 1

    @unroll for i in eachindex(TM_strides)
        stride_offset = (M ÷ TM_div[i]) % T_mod[TM_strides[i]]
        offset += stride_offset * GM_mul[i]
    end

    @unroll for i in eachindex(TK_strides)
        stride_offset = (K ÷ TK_div[i]) % T_mod[TK_strides[i]]
        offset += stride_offset * GK_mul[i]
    end

    if (is_load_strided == false)
        return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
    else
        return TensorLayout.sloada(Layout.Vec{NUMEL, T}, workspace, offset, strided_over_size)
    end
end

@inline function Layout.store!(
        ::Union{
            Type{TensorLayoutColMajor{T, TM_strides, TM_div, GM_mul, TN_strides, TN_div, GN_mul, T_mod, is_store_strided, strided_over_size}},
            Type{TensorLayoutRowMajor{T, TM_strides, TM_div, GM_mul, TN_strides, TN_div, GN_mul, T_mod, is_store_strided, strided_over_size}}
        }, workspace, value, tile::Tile{size}
    ) where {T, TM_strides, TM_div, GM_mul, TN_strides, TN_div, GN_mul, T_mod, is_store_strided, strided_over_size, size}

    NUMEL = 16 ÷ sizeof(T)

    M = tile.base.M + tile.offset.M
    N = tile.base.N + tile.offset.N

    offset = 1

    @unroll for i in eachindex(TM_strides)
        stride_offset = (M ÷ TM_div[i]) % T_mod[TM_strides[i]]
        offset += stride_offset * GM_mul[i]
    end

    @unroll for i in eachindex(TN_strides)
        stride_offset = (N ÷ TN_div[i]) % T_mod[TN_strides[i]]

        offset += stride_offset * GN_mul[i]
    end

    if (is_store_strided == false)
        return Layout.vstorea!(Layout.Vec{NUMEL, T}, pointer(workspace), value, offset)
    else
        TensorLayout.sstorea!(Layout.Vec{NUMEL, T}, workspace, value, offset, strided_over_size)
    end
end


# TODO: Write test. One example given below.
# precomputeGETTLayoutConstants([16, 512, 32], ([1, 3], [2]))
# result: ((1, 3), (2,), (1, 16), (1,), (16, 512, 32), (1, 8192), (16,))

# TODO: Write docstring.

export precomputeGETTLayoutConstants

function precomputeGETTLayoutConstants(
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_load_or_store_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    # 1. Convert the tensor strides from Tuple{Vector{Int}, Vector{int}}  to two separate 
    # Tuple{Int, Int, ...}.
    # ? @eval cannot work with Vector, but can work with Tuple, because the size of Tuple is 
    # ? known at compile time. 

    # → For the A matrix this will contain the tensor strides corresponding to the M stride.
    # e.g. for D[A, B, C] = A[B, D, A] * B[D, C] this will be (1, 3), since B and A belong to the
    # M stride.
    T1_strides = Tuple(x for x in T_strides[1])
    # → analogous, T2_stride will be equal to (2,) for the above example, since D belongs to the
    # K stride.
    T2_strides = Tuple(x for x in T_strides[2])


    # 2. Precompute the divisors used to calculate the tensor stride offsets.
    # → T1_stride_offset = (M ÷ T1_div[i]) % T1_mod[i] 
    T1_div = Vector{Int}(undef, length(T1_strides))
    div = 1
    for (idx, stride_idx) in enumerate(T1_strides)
        T1_div[idx] = div
        div *= T_strides_sizes[stride_idx]
    end
    T1_div = Tuple(x for x in T1_div)

    # 2b. Do the same for T2_stride.
    T2_div = Vector{Int}(undef, length(T2_strides))
    div = 1
    for (idx, stride_idx) in enumerate(T2_strides)
        T2_div[idx] = div
        div *= T_strides_sizes[stride_idx]
    end
    T2_div = Tuple(x for x in T2_div)


    # 3. Precompute the moduli used to calculate the tensor stride offsets.
    # These are simply the sizes of the tensor strides. Again, converted to Tuple{Int, Int, ...}.
    T_mod = Tuple(x for x in T_strides_sizes)


    # 4. Precompute the multiplicative terms used to calculate the GEMM stride offsets.
    # → offset += T1_stride_offset * G1_mul[i]
    G1_mul = Vector{Int}(undef, length(T1_strides))
    for (idx, stride_idx) in enumerate(T1_strides)
        G1_mul[idx] = 1
        for j = 1 : (stride_idx - 1) 
            G1_mul[idx] *= T_strides_sizes[j]
        end
    end
    G1_mul = Tuple(x for x in G1_mul)

    # 4b. Do the same for G2_mul.
    G2_mul = Vector{Int}(undef, length(T2_strides))
    for (idx, stride_idx) in enumerate(T2_strides)
        G2_mul[idx] = 1
        for j = 1 : (stride_idx - 1) 
            G2_mul[idx] *= T_strides_sizes[j]
        end
    end
    G2_mul = Tuple(x for x in G2_mul)

    # 5. Convert the Bool to an Int.
    is_load_or_store_strided = Int(is_load_or_store_strided)

    # 5.b If the load or store is strided, then precompute the size of the dimensions to stride 
    # over.
    strided_over_size = 1
    if (isnothing(load_or_store_strided_over) == false && is_load_or_store_strided == true)
        for stride_idx in load_or_store_strided_over
            strided_over_size *= T_strides_sizes[stride_idx]
        end
    end

    return (
        T1_strides, T2_strides,
        T1_div, T2_div,
        T_mod,
        G1_mul, G2_mul,
        is_load_or_store_strided, strided_over_size,
    )
end

function createALayout(
    T::DataType,
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_col_major::Bool,
    is_load_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    (
        TM_strides, TK_strides,
        TM_div, TK_div,
        T_mod,
        GM_mul, GK_mul,
        is_load_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_load_strided, load_or_store_strided_over)

    if (is_col_major == true)
        return TensorLayoutColMajor{T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size}
    else
        return TensorLayoutRowMajor{T, TM_strides, TM_div, GM_mul, TK_strides, TK_div, GK_mul, T_mod, is_load_strided, strided_over_size}
    end
end

function createBLayout(
    T::DataType,
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_col_major::Bool,
    is_load_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    (
        TK_strides, TN_strides,
        TK_div, TN_div,
        T_mod,
        GK_mul, GN_mul,
        is_load_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_load_strided, load_or_store_strided_over)

    if (is_col_major == true)
        return TensorLayoutColMajor{T, TK_strides, TK_div, GK_mul, TN_strides, TN_div, GN_mul, T_mod, is_load_strided, strided_over_size}
    else
        return TensorLayoutRowMajor{T, TK_strides, TK_div, GK_mul, TN_strides, TN_div, GN_mul, T_mod, is_load_strided, strided_over_size}
    end
end

function createCLayout(
    T::DataType,
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_load_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)

    (
        TM_strides, TN_strides,
        TM_div, TN_div,
        T_mod,
        GM_mul, GN_mul,
        is_load_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_load_strided, load_or_store_strided_over)

    # TODO: Add RowMajor support (goes along with attempt at more vectorised C loads and D stores)
    return TensorLayoutColMajor{T, TM_strides, TM_div, GM_mul, TN_strides, TN_div, GN_mul, T_mod, is_load_strided, strided_over_size}
end

# TODO: Make this also use the strided_over contant. It will probably be more efficient.
function createDLayout(
    T::DataType,
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_store_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    (
        TM_strides, TN_strides,
        TM_div, TN_div,
        T_mod,
        GM_mul, GN_mul,
        is_store_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_store_strided, load_or_store_strided_over)

    return TensorLayoutColMajor{T, TM_strides, TM_div, GM_mul, TN_strides, TN_div, GN_mul, T_mod, is_store_strided, strided_over_size}
end

end