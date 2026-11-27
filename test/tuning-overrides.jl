using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

################################################################################

@testset "Tuning overrides" begin
    # prepare(::TensorContraction) plans a contraction with the tuned layouts and
    # permutations by setting the global GemmKernels.Tensors.OVERRIDE_* variables. It must
    # switch them off again, also when planning fails, or every later contraction plan in
    # the same process silently uses the last tuned configuration.
    @testcase "reset after planning" begin
        problem = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [312, 312, 24, 312],
                                          data_type = Float16, compute_type = Float16,
                                          accumulate_type = Float32, zero_c = true)
        data = allocate_data(problem)
        params = (; BLOCK_N = 16, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 1,
                    OP_M = 32, OP_N = 8, OP_K = 16, kernel = Kernel.matmul_singlestage,
                    is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                    PERM_M = [2, 1], PERM_N = [3], PERM_K = [4], cta_swizzle = CTASwizzle.Identity)

        GemmKernels.Tensors.OVERRIDE_do_override = false

        # a valid configuration
        prepare(problem, data...; params..., BLOCK_M = 128)
        @test GemmKernels.Tensors.OVERRIDE_do_override == false

        # a configuration that is rejected while planning (BLOCK_M < 2 * OP_M)
        @test_throws GemmKernels.ConfigError prepare(problem, data...; params..., BLOCK_M = 32)
        @test GemmKernels.Tensors.OVERRIDE_do_override == false

        GemmKernels.Tensors.OVERRIDE_do_override = false
    end
end
