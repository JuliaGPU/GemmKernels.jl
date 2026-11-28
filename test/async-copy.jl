using CUDA
using GemmKernels
using LinearAlgebra

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

################################################################################

@testset "Asynchronous copies" begin
    @testcase "Layout support" begin
        F16 = Layout.UnsafeAlignedColMajor{Float16}
        F16_padded = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8}
        F16_row_padded = Layout.Padded{Layout.UnsafeAlignedRowMajor{Float16}, 8}

        @test Layout.can_copy_async(F16, F16_padded, (M = 8, K = 1))            # 16 bytes
        @test Layout.can_copy_async(F16, F16_padded, (M = 4, K = 1))            # 8 bytes
        @test Layout.can_copy_async(F16, F16_padded, (M = 2, K = 1))            # 4 bytes
        @test !Layout.can_copy_async(F16, F16_padded, (M = 1, K = 1))           # 2 bytes
        @test Layout.can_copy_async(Layout.UnsafeAlignedRowMajor{Float16}, F16_row_padded, (K = 1, N = 8))
        @test !Layout.can_copy_async(Layout.UnsafeAlignedColMajor{Float32}, F16_padded, (M = 4, K = 1))
        @test !Layout.can_copy_async(Layout.ColMajor{Float16}, F16_padded, (M = 8, K = 1))
    end

    # check_kernel runs when planning, on the host: the async kernels must reject GPUs
    # without cp.async, and configurations where neither A nor B can be copied
    # asynchronously; an operand with a global-to-shared transform is copied through
    # registers instead. Other kernels accept all of these.
    @testcase "Rejections" begin
        problem = WMMAMatrixMultiplication(; M = 512, N = 256, K = 1024, AB_type = Float16, CD_type = Float32,
                                             transpose_a = false, transpose_b = false, zero_c = false)
        (; config) = prepare(problem; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                                      OP_M = 16, OP_N = 16, OP_K = 16, zero_c = false,
                                      kernel = Kernel.matmul_pipelined_async(3))
        async = Kernel.matmul_pipelined_async(3)
        identity_transforms = (; global_to_shared_a = Transform.Elementwise(), global_to_shared_b = Transform.Elementwise())
        double_a = (; global_to_shared_a = Transform.Elementwise(x -> 2x), global_to_shared_b = Transform.Elementwise())
        double_b = (; global_to_shared_a = Transform.Elementwise(), global_to_shared_b = Transform.Elementwise(x -> 2x))
        double_both = (; global_to_shared_a = Transform.Elementwise(x -> 2x), global_to_shared_b = Transform.Elementwise(x -> 2x))

        @test GemmKernels.check_kernel(config, async, v"8.0", identity_transforms) === nothing
        @test GemmKernels.check_kernel(config, async, v"8.9", identity_transforms) === nothing
        @test_throws GemmKernels.ConfigError GemmKernels.check_kernel(config, async, v"7.5", identity_transforms)
        @test_throws GemmKernels.ConfigError GemmKernels.check_kernel(config, async, v"7.0", identity_transforms)
        @test GemmKernels.check_kernel(config, async, v"8.9", double_a) === nothing
        @test GemmKernels.check_kernel(config, async, v"8.9", double_b) === nothing
        @test_throws GemmKernels.ConfigError GemmKernels.check_kernel(config, async, v"8.9", double_both)
        @test GemmKernels.Kernel.copies_async(config.global_a_layout, config.shared_a_layout, config.mem_a_thread, double_a.global_to_shared_a) == false
        @test GemmKernels.Kernel.copies_async(config.global_b_layout, config.shared_b_layout, config.mem_b_thread, double_a.global_to_shared_b) == true
        for kernel in [Kernel.matmul_singlestage, Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng]
            @test GemmKernels.check_kernel(config, kernel, v"7.0", double_both) === nothing
        end
    end

    # plan_matmul must apply those checks, with the capability of the device it targets.
    if CUDA.functional()
        @testcase "Planning for a pre-Ampere GPU" begin
            problem = WMMAMatrixMultiplication(; M = 512, N = 256, K = 1024, AB_type = Float16, CD_type = Float32,
                                                 transpose_a = false, transpose_b = false, zero_c = false)
            a, b, c, d = allocate_data(problem)
            (; config) = prepare(problem; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                                          OP_M = 16, OP_N = 16, OP_K = 16, zero_c = false,
                                          kernel = Kernel.matmul_pipelined_async(3))
            old_cap = GemmKernels.OVERRIDE_cap[]
            try
                GemmKernels.OVERRIDE_cap[] = v"7.5"
                @test_throws GemmKernels.ConfigError GemmKernels.plan_matmul(config, a, b, c, d;
                                                                             kernel = Kernel.matmul_pipelined_async(3))
            finally
                GemmKernels.OVERRIDE_cap[] = old_cap
            end
        end
    end

    if CUDA.functional() && capability(device()) ≥ v"8.0"
        @testcase "GEMM $(transpose_a ? 'T' : 'N')$(transpose_b ? 'T' : 'N'), $(stages) stages, K = $(K)" for
                transpose_a in [false, true], transpose_b in [false, true], stages in [2, 3, 4], K in [32, 1024]
            # K = 32 is a single K-tile: fewer tiles than the prologue would prefetch.
            problem = WMMAMatrixMultiplication(; M = 512, N = 256, K, AB_type = Float16, CD_type = Float32,
                                                 transpose_a, transpose_b, zero_c = false)
            data = allocate_data(problem)
            initialize_data(problem, data...)
            reference = Array(calculate_reference(problem, data...))

            args = prepare(problem; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                                    OP_M = 16, OP_N = 16, OP_K = 16, zero_c = false,
                                    kernel = Kernel.matmul_pipelined_async(stages))
            initialize_data(problem, data...)
            result = Array(execute(problem, data...; args...))
            @test isapprox(result, reference; rtol = sqrt(eps(Float16)))
        end

        @testcase "Tensor contraction, $(stages) stages" for stages in [2, 3, 4]
            problem = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [312, 312, 24, 312],
                                              data_type = Float16, compute_type = Float16,
                                              accumulate_type = Float32, zero_c = true)
            data = allocate_data(problem)
            initialize_data(problem, data...)
            reference = Array(calculate_reference(problem, data...))

            params = (; BLOCK_M = 128, BLOCK_N = 16, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 1,
                        OP_M = 32, OP_N = 8, OP_K = 16, kernel = Kernel.matmul_pipelined_async(stages),
                        is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                        PERM_N = [3], PERM_K = [4], cta_swizzle = CTASwizzle.HorizontallyTiled{4})

            # PERM_M = [2, 1]: A's first dimension is accessed first, so loads are contiguous.
            args = try
                prepare(problem, data...; params..., PERM_M = [2, 1])
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            initialize_data(problem, data...; params...)
            result = Array(execute(problem, data...; args...))
            @test verify(problem, reference, result)

            # PERM_M = [1, 2]: strided loads of A, which cannot be copied asynchronously: A is
            # copied through registers, B still with cp.async.
            args = try
                prepare(problem, data...; params..., PERM_M = [1, 2])
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            initialize_data(problem, data...; params...)
            result = Array(execute(problem, data...; args...))
            @test verify(problem, reference, result)

            # ... and with B row-major, B's loads are strided too: nothing left to copy
            # asynchronously, so the configuration is rejected.
            try
                @test_throws GemmKernels.ConfigError prepare(problem, data...; params..., PERM_M = [1, 2],
                                                             is_B_col_major = false)
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
        end
    end
end
