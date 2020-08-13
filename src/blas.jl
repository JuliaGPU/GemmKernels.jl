module BLAS

using CUDA
using GemmKernels

# Based on https://github.com/JuliaGPU/CUDA.jl/blob/bd5a2a8800e91eb6a7df89eb5dd4bb8fc503541d/lib/cublas/wrappers.jl#L743-L769
function gemmEx!(transA::Char, transB::Char, alpha::Number, A::CuArray{Float16}, B::CuArray{Float16}, beta::Number, C::CuArray{Float32})
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)

    if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch("Dimensions do not match"))
    end

    computeType = Float32

    transpose_a = (transA == 'T')
    transpose_b = (transB == 'T')

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            operator = Operator.WMMAOp{16, 16, 16},

            global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},
            global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},
            global_c_layout = Layout.AlignedColMajor{Float32},
            global_d_layout = Layout.AlignedColMajor{Float32},

            is_a_col_major = !transpose_a,
            is_b_col_major = !transpose_b
                                )

    GemmKernels.matmul(A, B, C, C, conf;
                       transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta))
end

end
