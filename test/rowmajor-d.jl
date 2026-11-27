using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# A GETT contraction with a row-major D (is_D_col_major = false) uses UnsafeAlignedRowMajor
# shared C and D arrays. Their shape must be (N, M), the order in which they are addressed.
function run_rowmajor_d_config(problem; params...)
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

@testset "Row-major shared D" begin
    @testcase "physical size" begin
        @test Layout.physical_size(Layout.UnsafeAlignedRowMajor{Float32}, (M = 128, N = 64)) == (64, 128)
        @test Layout.physical_size(Layout.UnsafeAlignedColMajor{Float32}, (M = 128, N = 64)) == (128, 64)
    end

    # abc-bda-dc, the contraction of the February 2026 sweep, in which 97% of the crashed
    # configurations had a row-major D and BLOCK_M > BLOCK_N.
    problem = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [312, 312, 24, 312],
                                      data_type = Float16, compute_type = Float16,
                                      accumulate_type = Float32, zero_c = true)

    # 128x64: the old shared D was addressed up to 4 (64 + 127 * 128) bytes into a
    # 32 KiB allocation (an illegal-address crash). 64x128: rows of the old shared D
    # overlapped (wrong results). 64x64: square, correct before and after the fix.
    @testcase "$(kernel), $(BLOCK_M)x$(BLOCK_N)" for kernel in [Kernel.matmul_singlestage, Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng],
                                                     (BLOCK_M, BLOCK_N) in [(128, 64), (64, 128), (64, 64)]
        @test run_rowmajor_d_config(problem;
                                    BLOCK_M, BLOCK_N, BLOCK_K = 32,
                                    WARPS_M = 2, WARPS_N = 2,
                                    OP_M = 16, OP_N = 16, OP_K = 16,
                                    kernel,
                                    is_A_col_major = true, is_B_col_major = true, is_D_col_major = false,
                                    PERM_M = [1, 2], PERM_N = [3], PERM_K = [4],
                                    cta_swizzle = CTASwizzle.Identity)
    end
end
