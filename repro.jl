using GemmKernels
using CUDA
using Pkg

include("configs/configs.jl")

# [ Info: Running benchmark WMMA GEMM Float16*Float16+Float32=Float32 (256×256) · (256×256) (TN) Block (256, 64, 64) Warps (2, 2) OP (16, 16, 16)...

function main()
    M = N = K = 256
    AB_type = Float16
    CD_type = Float32
    transpose_a = true
    transpose_b = false
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 64, 64
    WARPS_M, WARPS_N = 2, 2
    zero_c = false
    OP_M, OP_N, OP_K = 16, 16, 16
    kernel = Kernel.matmul_pipelined

    cf = @get_wmma_config

    c_h, a, b, c, d = generate_inputs(cf)
    run_gemm(cf, a, b, c, d)

    CUDA.device_synchronize()
end

isinteractive() || main()
