using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

################################################################################

@testset "Kernel cache" begin
    # plan_matmul caches compiled kernels on disk. The three kernels for the same
    # configuration generate different code, so each must get its own cache entry;
    # if the key ignores the kernel, all three load whichever was compiled first.
    @testcase "one entry per kernel" begin
        problem = WMMAMatrixMultiplication(; M = 256, N = 256, K = 256, AB_type = Float16, CD_type = Float32,
                                             transpose_a = false, transpose_b = false, zero_c = true)
        a, b, c, d = allocate_data(problem)
        kernels = [Kernel.matmul_singlestage, Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng]
        compiled = map(kernels) do kernel
            (; config) = prepare(problem; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 64, WARPS_M = 2, WARPS_N = 2,
                                          OP_M = 16, OP_N = 16, OP_K = 16, zero_c = true, kernel)
            GemmKernels.plan_matmul(config, a, b, c, d; kernel).compiled
        end
        @test allunique(compiled)

        # planning the same kernel again reuses its entry
        (; config) = prepare(problem; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 64, WARPS_M = 2, WARPS_N = 2,
                                      OP_M = 16, OP_N = 16, OP_K = 16, zero_c = true, kernel = kernels[1])
        @test GemmKernels.plan_matmul(config, a, b, c, d; kernel = kernels[1]).compiled == compiled[1]
    end
end
