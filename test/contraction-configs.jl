using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# Run one tensor contraction with explicit tuning parameters through the same code path
# as the tuning script (tuning/tune.jl), and compare against cuTENSOR.
function run_contraction_config(problem; params...)
    data = allocate_data(problem)
    initialize_data(problem, data...)
    reference = Array(calculate_reference(problem, data...))

    # prepare() sets global overrides in GemmKernels.Tensors; don't leak them into other tests.
    args = try
        prepare(problem, data...; params...)
    finally
        GemmKernels.Tensors.OVERRIDE_do_override = false
    end
    initialize_data(problem, data...; params...)
    result = Array(execute(problem, data...; args...))

    return verify(problem, reference, result)
end

################################################################################

@testset "Tensor contraction configurations" begin
    # abc-bda-dc: N = 24 is padded to 32, i.e. 2 tile columns for BLOCK_N = 16,
    # and M = 312 * 312 padded to 780 tile rows for BLOCK_M = 128. Neither is a
    # multiple of the larger swizzle sizes, which exercises the partial strips.
    problem = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [312, 312, 24, 312],
                                      data_type = Float16, compute_type = Float16,
                                      accumulate_type = Float32, zero_c = true)

    @testcase "$(kernel), $(swizzle)" for kernel in [Kernel.matmul_singlestage, Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng],
                                          swizzle in [CTASwizzle.Identity,
                                                      CTASwizzle.HorizontallyTiled{2}, CTASwizzle.HorizontallyTiled{4},
                                                      CTASwizzle.VerticallyTiled{2}, CTASwizzle.VerticallyTiled{16}]
        @test run_contraction_config(problem;
                                     BLOCK_M = 128, BLOCK_N = 16, BLOCK_K = 32,
                                     WARPS_M = 2, WARPS_N = 1,
                                     OP_M = 32, OP_N = 8, OP_K = 16,
                                     kernel,
                                     is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                                     PERM_M = [1, 2], PERM_N = [3], PERM_K = [4],
                                     cta_swizzle = swizzle)
    end
end
