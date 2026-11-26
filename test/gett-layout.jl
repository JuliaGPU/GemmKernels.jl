using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# Run one tensor contraction with explicit tuning parameters through the same code path
# as the tuning script (tuning/tune.jl), and compare against cuTENSOR.
function run_gett_config(problem; params...)
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

@testset "GETT layouts" begin
    problem = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [312, 312, 24, 312],
                                      data_type = Float16, compute_type = Float16,
                                      accumulate_type = Float32, zero_c = true)

    # With a 16 x 32 A tile and 4 warps, every thread copies only
    # 16 * 32 ÷ (4 * 32) = 4 elements of A, i.e. 8 bytes instead of 16.
    #  - PERM_M = [2, 1]: the first accessed M mode is A's first dimension, so the loads
    #    of A are vectorised (and used to be misaligned 16-byte loads).
    #  - PERM_M = [1, 2]: the loads of A are strided (and used to read past the tile).
    @testcase "few elements per thread, $(kernel), PERM_M = $(perm_m)" for
            kernel in [Kernel.matmul_singlestage, Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng],
            perm_m in [[2, 1], [1, 2]]
        @test run_gett_config(problem;
                              BLOCK_M = 16, BLOCK_N = 128, BLOCK_K = 32,
                              WARPS_M = 1, WARPS_N = 4,
                              OP_M = 8, OP_N = 32, OP_K = 16,
                              kernel,
                              is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                              PERM_M = perm_m, PERM_N = [3], PERM_K = [4],
                              cta_swizzle = CTASwizzle.Identity)
    end
end
