using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# matmul_pipelined_ng alternates between two shared memory buffers for the K-tiles of A and
# B. It used to select the buffer for the next K-tile from the K-tile index, wrapped around
# to 0 in the last iteration. With an odd number of K-tiles (3 or more), that selected the
# buffer of the previous iteration, which other warps may still be reading: a race, giving
# wrong results that vary from run to run. These configurations come from a tuning run, in
# which they produced invalid results in (almost) every run.
function run_ng_stages_config(problem; params...)
    data = allocate_data(problem)
    initialize_data(problem, data...)
    reference = Array(calculate_reference(problem, data...))

    # prepare() sets global overrides in GemmKernels.Tensors; don't leak them into other tests.
    args = try
        prepare(problem, data...; params...)
    finally
        GemmKernels.Tensors.OVERRIDE_do_override = false
    end

    # the race does not hit every run (and less often with bounds checking, as under
    # Pkg.test): try several times
    valid = true
    for _ in 1:10
        initialize_data(problem, data...; params...)
        result = Array(execute(problem, data...; args...))
        valid &= verify(problem, reference, result)
    end
    return valid
end

################################################################################

@testset "pipelined_ng stage buffers" begin
    # abcd-ebad-ce: K = 72, padded to 80 (5 K-tiles of 16).
    problem_8warps = WMMATensorContraction(; name = "1.2.3.4-5.2.1.4-3.5", extents = [72, 72, 24, 72, 72],
                                             data_type = Float16, compute_type = Float16,
                                             accumulate_type = Float32, zero_c = true)

    # one K step per K-tile (BLOCK_K = OP_K), 5 K-tiles, 8 warps
    @testcase "256x16x16 block, 8x1 warps" begin
        @test run_ng_stages_config(problem_8warps;
                                   BLOCK_M = 256, BLOCK_N = 16, BLOCK_K = 16,
                                   WARPS_M = 8, WARPS_N = 1,
                                   OP_M = 32, OP_N = 8, OP_K = 16,
                                   kernel = Kernel.matmul_pipelined_ng,
                                   is_A_col_major = false, is_B_col_major = false, is_D_col_major = true,
                                   PERM_M = [2, 1, 4], PERM_N = [3], PERM_K = [5],
                                   cta_swizzle = CTASwizzle.Identity)
    end

    # abcd-ea-ebcd: K = 72, padded to 80 (5 K-tiles of 16) or 96 (3 K-tiles of 32).
    problem = WMMATensorContraction(; name = "1.2.3.4-5.1-5.2.3.4", extents = [72, 72, 72, 72, 72],
                                      data_type = Float16, compute_type = Float16,
                                      accumulate_type = Float32, zero_c = true)

    # one K step per K-tile (BLOCK_K = OP_K), 5 K-tiles
    @testcase "64x64x16 block, 2x1 warps" begin
        @test run_ng_stages_config(problem;
                                   BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 16,
                                   WARPS_M = 2, WARPS_N = 1,
                                   OP_M = 16, OP_N = 16, OP_K = 16,
                                   kernel = Kernel.matmul_pipelined_ng,
                                   is_A_col_major = true, is_B_col_major = false, is_D_col_major = true,
                                   PERM_M = [1], PERM_N = [3, 2, 4], PERM_K = [5],
                                   cta_swizzle = CTASwizzle.Identity)
    end

    # two K steps per K-tile (BLOCK_K = 2 * OP_K), 3 K-tiles
    @testcase "32x64x32 block, 2x1 warps" begin
        @test run_ng_stages_config(problem;
                                   BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 32,
                                   WARPS_M = 2, WARPS_N = 1,
                                   OP_M = 8, OP_N = 32, OP_K = 16,
                                   kernel = Kernel.matmul_pipelined_ng,
                                   is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                                   PERM_M = [1], PERM_N = [4, 2, 3], PERM_K = [5],
                                   cta_swizzle = CTASwizzle.Identity)
    end
end
