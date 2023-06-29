module BLAS

using CUDA
using GemmKernels
using LinearAlgebra

# Convert matrix to type compatible with kernel
convert_matrix(mat) = mat
convert_matrix(mat::Diagonal{T, A}) where {T, A} = mat.diag

# Select the best kernel
kernel(layout_a, layout_b) = Kernel.matmul_singlestage
kernel(::Type{Layout.AlignedColMajor{T}}, ::Type{Layout.AlignedColMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.AlignedColMajor{T}}, ::Type{Layout.AlignedRowMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.AlignedRowMajor{T}}, ::Type{Layout.AlignedColMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.AlignedRowMajor{T}}, ::Type{Layout.AlignedRowMajor{T}}) where {T} = Kernel.matmul_pipelined

# Based on https://github.com/JuliaGPU/CUDA.jl/blob/bd5a2a8800e91eb6a7df89eb5dd4bb8fc503541d/lib/cublas/wrappers.jl#L743-L769
function gemmEx!(transA::Char, transB::Char, alpha::Number, A, B, beta::Number, C)
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)

    if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch("Dimensions do not match"))
    end

    transpose_a = (transA == 'T')
    transpose_b = (transB == 'T')

    a_layout_base = transpose_a ? Layout.AlignedRowMajor : Layout.AlignedColMajor
    b_layout_base = transpose_b ? Layout.AlignedRowMajor : Layout.AlignedColMajor

    # determine global memory layouts
    ## if alpha is zero, we don't need to load A or B
    if iszero(alpha)
        global_a_layout = Layout.Zero{eltype(A)}
        global_b_layout = Layout.Zero{eltype(B)}
    else
        global_a_layout = a_layout_base{eltype(A)}
        global_b_layout = b_layout_base{eltype(B)}
    end
    ## if beta is zero, we don't need to load C
    global_c_layout = if iszero(beta)
        Layout.Zero{eltype(C)}
    else
        Layout.AlignedColMajor{eltype(C)}
    end
    global_d_layout = Layout.AlignedColMajor{eltype(C)}

    # determine shared memory layouts
    ## padded to avoid bank conflicts
    shared_a_layout = Layout.Padded{a_layout_base{eltype(A)}, 8}
    shared_b_layout = Layout.Padded{b_layout_base{eltype(B)}, 8}
    ## outputs are never transposed, and padding them doesn't seem worth it
    shared_c_layout = shared_d_layout = Layout.AlignedColMajor{eltype(C)}

    conf = GemmKernels.get_config(;
            gemm_shape = (M = m, N = n, K = k),
            operator = Operator.WMMAOp{16, 16, 16, eltype(C)},

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = !transpose_a,
            is_b_col_major = !transpose_b
                                )

    GemmKernels.matmul(convert_matrix(A), convert_matrix(B), convert_matrix(C), convert_matrix(C), conf;
                       transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                       kernel = kernel(global_a_layout, global_b_layout)
                      )
end

end
