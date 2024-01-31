# vim: fdm=marker

using GemmKernels
using GemmKernels: Config, LocalArray, @immutable, mma884_row_row, @staticdef, BitArrayIndex, @unrolled, @not_unrolled, constant, variadic, tid, bid_x, bid_y, warpid, vloada, vstorea!, Vec, b, Layout
using Test
using CUDA
using LLVMLoopInfo: @loopinfo
using Base.Cartesian: @ntuple
using Base
using LinearAlgebra
using GemmKernels.Operator: VoltaMmaSyncOp, mma, load_a, load_b, store_d
using GemmKernels.Layout: VoltaSwizzledOperandA, VoltaSwizzledOperandB
using GemmKernels.Tiling

# globals {{{
conf = GemmKernels.get_config(
    gemm_shape = (M = 2048, N = 2048, K = 2048),
    block_shape = (M = 128, N = 256, K = 32),
    warps_per_block = 8,

    compute_warp = (M = 64, N = 64, K = 4),

    global_a_layout = Layout.UnsafeAlignedRowMajor{Float16},
    global_b_layout = Layout.UnsafeAlignedRowMajor{Float16},
    global_c_layout = Layout.Zero{Float32},
    global_d_layout = Layout.UnsafeAlignedRowMajor{Float32},

    shared_a_layout = Layout.VoltaSwizzledOperandA{Float16},
    shared_b_layout = Layout.VoltaSwizzledOperandB{Float16},
    shared_c_layout = Layout.Zero{Float32},
    shared_d_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float32}, 2},

    operator = VoltaMmaSyncOp,

    is_a_col_major = false,
    is_b_col_major = false
   )

# The kernel calculates A * B = D (in row-major), as this is CUTLASS's
# convention.
# To calculate A * B = D in col-major, just flip the A and B operands
# and transpose: A * B = D <=> B^T * A^T = D^T.
A = CUDA.rand(Float16, (conf.matmul_shape.N, conf.matmul_shape.K))
B = CUDA.rand(Float16, (conf.matmul_shape.K, conf.matmul_shape.M))
D = CUDA.zeros(Float32, (conf.matmul_shape.N, conf.matmul_shape.M))
# }}}


# driver {{{
function test(; dump_code=false, debug=false)
    if debug
        @device_code_warntype interactive=true GemmKernels.matmul(conf, B, A, D, D, kernel = Kernel.volta_kernel)
        return
    end

    if dump_code
        @device_code dir="gemm-output" GemmKernels.matmul(conf, B, A, D, D, kernel = Kernel.volta_kernel)
    end

    D_ref = similar(D)

    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D_ref)

    GemmKernels.matmul(conf, B, A, D, D, kernel = Kernel.volta_kernel)

    compare(x, y) = isapprox(x, y; rtol=sqrt(eps(Float16)))

    @test isapprox(D_ref, D; rtol=sqrt(eps(Float16)))
    @test isapprox(D_ref, D; rtol=sqrt(eps(Float16)), norm=M -> LinearAlgebra.norm(M, Inf))
    @test all(compare.(D, D_ref))
end

function cublas()
    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D)
end

isinteractive() || test()
# }}}
