module BLAS

using CUDA
using GemmKernels
using LinearAlgebra

# Global layouts
global_layout(::Type{CuArray{T, N}}, ::Val{false}) where {T, N} = Layout.AlignedColMajor{T}
global_layout(::Type{CuArray{T, N}}, ::Val{true}) where {T, N} = Layout.AlignedRowMajor{T}
global_layout(::Type{Diagonal{Float16, CuArray{Float16, N}}}, transpose) where {N} = Layout.Diagonal{Float16}

# Shared layouts for A / B
shared_layout_ab(typ::Type{CuArray{Float16, N}}, transpose) where {N} = Layout.Padded{global_layout(typ, transpose), 8}
shared_layout_ab(::Type{Diagonal{Float16, CuArray{Float16, N}}}, transpose) where {N, P} = shared_layout_ab(CuArray{Float16, N}, transpose)

# Shared layouts for C / D
shared_layout_cd(typ::Type{CuArray{T, N}}, transpose) where {T, N} = global_layout(typ, transpose)

# Convert matrix to type compatible with kernel
convert_matrix(mat) = mat
convert_matrix(mat::Diagonal{T, A}) where {T, A} = mat.diag

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

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            operator = Operator.WMMAOp{16, 16, 16},

            global_a_layout = global_layout(typeof(A), Val(transpose_a)),
            global_b_layout = global_layout(typeof(B), Val(transpose_b)),
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
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * (beta / alpha)),
                       transform_regs_to_shared_d = Transform.Elementwise(x -> x * alpha))
end

end
