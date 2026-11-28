using CUDA
using GemmKernels
using LinearAlgebra

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# Split-K (plan_matmul(...; split_k)): split_k thread blocks per output tile each compute part
# of the K dimension; the last one to finish adds the partial tiles (Epilogue.SplitK).

# A WMMA GEMM with the given kernel and split, planned once and run twice: the second run
# checks that the counters were reset, and must give exactly the same result (the partial
# tiles are always added in the same order).
function run_split_gemm(; K, kernel, split_k, transpose_a = false, transpose_b = false)
    problem = WMMAMatrixMultiplication(; M = 512, N = 256, K, AB_type = Float16, CD_type = Float32,
                                         transpose_a, transpose_b, zero_c = false)
    a, b, c, d = allocate_data(problem)
    initialize_data(problem, a, b, c, d)
    reference = Array(calculate_reference(problem, a, b, c, d))

    (; config) = prepare(problem; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                                  OP_M = 16, OP_N = 16, OP_K = 16, zero_c = false, kernel)
    alpha, beta = problem.alpha, problem.beta
    plan = GemmKernels.plan_matmul(config, a, b, c, d;
                                   transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                                   transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                                   kernel, split_k)
    results = map(1:2) do _
        initialize_data(problem, a, b, c, d)
        fill!(d, 0)
        GemmKernels.matmul(plan, a, b, c, d)
        Array(d)
    end
    return results[1], results[2], reference
end

################################################################################

@testset "Split-K" begin
    if CUDA.functional()
        kernels = Any[Kernel.matmul_singlestage]
        capability(device()) ≥ v"8.0" && push!(kernels, Kernel.matmul_pipelined_async(3))

        # K = 1024: 32 K-tiles; K = 160: 5 K-tiles, so with 4 parts (2, 2, 1, 0 K-tiles)
        # the last part is empty. C is non-zero, so it must be added exactly once.
        @testcase "GEMM, $(kernel), K = $(K), split_k = $(split_k)" for kernel in kernels, K in [1024, 160],
                                                                        split_k in [1, 2, 4]
            run1, run2, reference = run_split_gemm(; K, kernel, split_k)
            @test isapprox(run1, reference; rtol = sqrt(eps(Float16)))
            @test run1 == run2
        end

        @testcase "GEMM, transposed inputs, split_k = 3" begin
            run1, run2, reference = run_split_gemm(; K = 1024, kernel = Kernel.matmul_singlestage,
                                                      split_k = 3, transpose_a = true, transpose_b = true)
            @test isapprox(run1, reference; rtol = sqrt(eps(Float16)))
            @test run1 == run2
        end

        @testcase "Rejections" begin
            problem = WMMAMatrixMultiplication(; M = 512, N = 256, K = 160, AB_type = Float16, CD_type = Float32,
                                                 transpose_a = false, transpose_b = false, zero_c = false)
            a, b, c, d = allocate_data(problem)
            (; config) = prepare(problem; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                                          OP_M = 16, OP_N = 16, OP_K = 16, zero_c = false,
                                          kernel = Kernel.matmul_singlestage)
            # more parts than K-tiles (5), or none
            @test_throws GemmKernels.ConfigError GemmKernels.plan_matmul(config, a, b, c, d; split_k = 6)
            @test_throws GemmKernels.ConfigError GemmKernels.plan_matmul(config, a, b, c, d; split_k = 0)
            # kernels without split-K support
            for kernel in [Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng]
                @test_throws GemmKernels.ConfigError GemmKernels.plan_matmul(config, a, b, c, d; kernel, split_k = 2)
            end
            # a transform on the partial tiles would be applied to each part separately
            @test_throws GemmKernels.ConfigError GemmKernels.plan_matmul(
                config, a, b, c, d; split_k = 2, transform_regs_to_shared_d = Transform.Elementwise(x -> 2x))
            # split_k = 1 is the plain kernel
            @test GemmKernels.plan_matmul(config, a, b, c, d; split_k = 1) isa GemmKernels.MatmulPlan
        end

        # ab-acd-dbc (as TC13, smaller): a long K = 64 · 96 = 6144 for a small 64 × 64 output,
        # so with 32 × 32 tiles there are only 4 thread blocks without split-K.
        @testcase "Tensor contraction, split_k = $(split_k)" for split_k in [1, 4, 16]
            problem = WMMATensorContraction(; name = "1.2-1.3.4-4.2.3", extents = [64, 64, 64, 96],
                                              data_type = Float16, compute_type = Float16,
                                              accumulate_type = Float32, zero_c = true)
            data = allocate_data(problem)
            initialize_data(problem, data...)
            reference = Array(calculate_reference(problem, data...))

            params = (; BLOCK_M = 32, BLOCK_N = 32, BLOCK_K = 64, WARPS_M = 2, WARPS_N = 2,
                        OP_M = 16, OP_N = 16, OP_K = 16, kernel = Kernel.matmul_singlestage,
                        is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                        PERM_M = [1], PERM_N = [2], PERM_K = [3, 4], cta_swizzle = CTASwizzle.Identity)
            args = try
                prepare(problem, data...; params..., SPLIT_K = split_k)
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            initialize_data(problem, data...; params...)
            result = Array(execute(problem, data...; args...))
            @test verify(problem, reference, result)
        end
    end
end
