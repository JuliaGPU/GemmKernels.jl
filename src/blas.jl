module BLAS

using CUDA
using GemmKernels
using LinearAlgebra

# Global layouts
global_layout(::Type{<:CuArray{T}}, ::Val{false}) where {T} = Layout.AlignedColMajor{T}
global_layout(::Type{<:CuArray{T}}, ::Val{true}) where {T} = Layout.AlignedRowMajor{T}
global_layout(::Type{<:Diagonal{T, <:CuArray{T}}}, transpose) where T = Layout.Diagonal{T}

# Shared layouts for A / B
shared_layout_ab(typ::Type{<:CuArray{T}}, transpose) where T = Layout.Padded{global_layout(typ, transpose), 8}
shared_layout_ab(::Type{<:Diagonal{T, <:CuArray{T, N}}}, transpose) where {N, P, T} = shared_layout_ab(CuArray{T, N}, transpose)

# Shared layouts for C / D
shared_layout_cd(typ::Type{<:CuArray{T}}, transpose) where {T} = global_layout(typ, transpose)

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

    a_layout = global_layout(typeof(A), Val(transpose_a))
    b_layout = global_layout(typeof(B), Val(transpose_b))

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            operator = Operator.WMMAOp{16, 16, 16, eltype(C)},

            global_a_layout = a_layout,
            global_b_layout = b_layout,
            global_c_layout = global_layout(typeof(C), Val(false)),
            global_d_layout = global_layout(typeof(C), Val(false)),

            shared_a_layout = shared_layout_ab(typeof(A), Val(transpose_a)),
            shared_b_layout = shared_layout_ab(typeof(B), Val(transpose_b)),
            shared_c_layout = shared_layout_cd(typeof(C), Val(false)),
            shared_d_layout = shared_layout_cd(typeof(C), Val(false)),

            is_a_col_major = !transpose_a,
            is_b_col_major = !transpose_b
                                )

    GemmKernels.matmul(convert_matrix(A), convert_matrix(B), convert_matrix(C), convert_matrix(C), conf;
                       transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                       kernel = kernel(a_layout, b_layout)
                      )
end

end
