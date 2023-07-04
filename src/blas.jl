module BLAS

using CUDA
using GemmKernels
using LinearAlgebra

# Select the best kernel
kernel(layout_a, layout_b) = Kernel.matmul_singlestage
kernel(::Type{Layout.UnsafeAlignedColMajor{T}}, ::Type{Layout.UnsafeAlignedColMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.UnsafeAlignedColMajor{T}}, ::Type{Layout.UnsafeAlignedRowMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.UnsafeAlignedRowMajor{T}}, ::Type{Layout.UnsafeAlignedColMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.UnsafeAlignedRowMajor{T}}, ::Type{Layout.UnsafeAlignedRowMajor{T}}) where {T} = Kernel.matmul_pipelined

# Based on https://github.com/JuliaGPU/CUDA.jl/blob/bd5a2a8800e91eb6a7df89eb5dd4bb8fc503541d/lib/cublas/wrappers.jl#L743-L769
function gemmEx!(transA::Char, transB::Char, alpha::Number, A::CuMatrix, B::CuMatrix,
                 beta::Number, C::CuMatrix; wmma::Union{Bool,Nothing}=nothing)
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch("Dimensions do not match"))
    end

    transpose_a = (transA == 'T')
    transpose_b = (transB == 'T')
    a_layout_base = transpose_a ? Layout.RowMajor : Layout.ColMajor
    b_layout_base = transpose_b ? Layout.RowMajor : Layout.ColMajor
    a_aligned_layout_base = transpose_a ? Layout.UnsafeAlignedRowMajor : Layout.UnsafeAlignedColMajor
    b_aligned_layout_base = transpose_b ? Layout.UnsafeAlignedRowMajor : Layout.UnsafeAlignedColMajor

    # determine operator to use
    wmma_types = [
        (Float16, Float16, Float16),
        (Float16, Float16, Float32),
        # TODO: more, and device-capability dependent
    ]
    compute_type = promote_type(eltype(A), eltype(B))
    supports_wmma = something(wmma, (compute_type, compute_type, eltype(C)) in wmma_types)

    # determine shared memory layouts
    ## padded to avoid bank conflicts
    if supports_wmma
        # in the case of WMMA, the shared memory needs to have the correct type already,
        # as we'll use WMMA intrinsics to load from it.
        shared_a_layout = Layout.Padded{a_aligned_layout_base{compute_type}, 8}
        shared_b_layout = Layout.Padded{b_aligned_layout_base{compute_type}, 8}
    else
        shared_a_layout = Layout.Padded{a_aligned_layout_base{eltype(A)}, 8}
        shared_b_layout = Layout.Padded{b_aligned_layout_base{eltype(B)}, 8}
    end
    ## outputs are never transposed, and padding them doesn't seem worth it
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # determine block shape
    # XXX: heuristic should take much more into account (GEMM size, at least)
    block_shape = if supports_wmma
        GemmKernels.heuristic_block_shape(shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout)
    else
        # XXX: heuristic for FPU
        (M = 128, N = 128, K = 32)
    end

    # determine global memory layouts
    ## check if tiles begin at aligned addresses, allowing use of vectorized loads & stores
    a_aligned = (stride(A, 2) * sizeof(eltype(A))) % 16 == 0
    b_aligned = (stride(B, 2) * sizeof(eltype(B))) % 16 == 0
    c_aligned = (stride(C, 2) * sizeof(eltype(C))) % 16 == 0
    ## if alpha is zero, we don't need to load A or B
    if iszero(alpha)
        global_a_layout = Layout.Zero{eltype(A)}
        global_b_layout = Layout.Zero{eltype(B)}
    else
        global_a_layout = if a_aligned && m%block_shape.M == 0 &&  k%block_shape.K == 0
            a_aligned_layout_base{eltype(A)}
        else
            a_layout_base{eltype(A)}
        end
        global_b_layout = if b_aligned && k%block_shape.K == 0 && n%block_shape.N == 0
            b_aligned_layout_base{eltype(B)}
        else
            b_layout_base{eltype(B)}
        end
    end
    ## if beta is zero, we don't need to load C
    global_c_layout = if iszero(beta)
        Layout.Zero{eltype(C)}
    else
        if c_aligned && m%block_shape.M == 0 && n%block_shape.N == 0
            Layout.UnsafeAlignedColMajor{eltype(C)}
        else
            Layout.ColMajor{eltype(C)}
        end
    end
    global_d_layout = if c_aligned && m%block_shape.M == 0 && n%block_shape.N == 0
        Layout.UnsafeAlignedColMajor{eltype(C)}
    else
        Layout.ColMajor{eltype(C)}
    end

    conf = if supports_wmma
        GemmKernels.get_config(;
            gemm_shape = (M = m, N = n, K = k), block_shape,
            operator = Operator.WMMAOp{16, 16, 16, compute_type, eltype(C)},

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = !transpose_a,
            is_b_col_major = !transpose_b
        )
    else
        GemmKernels.get_config(;
            gemm_shape = (M = m, N = n, K = k), block_shape,
            operator = Operator.FPUOp{8, 8, 1, compute_type, eltype(C)},

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = !transpose_a,
            is_b_col_major = !transpose_b
        )
    end

    alpha = convert(compute_type, alpha)
    beta = convert(eltype(C), beta)
    GemmKernels.matmul(A, B, C, C, conf;
                       transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                       kernel = kernel(global_a_layout, global_b_layout)
                      )
    C
end

end
