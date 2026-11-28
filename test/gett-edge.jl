using CUDA
using GemmKernels
using GemmKernels.Tensors.GETTLayout

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# Partial edge tiles (createGETTLayout's `bounds`, the configs' EDGE = "mask"): M and N need
# not be multiples of the block shape; the layouts skip the stores of the tiles beyond them,
# and clamp their loads to the last tile within them.

@testset "GETT partial edge tiles" begin
    @testcase "layout types" begin
        args = (Float16, [72, 72, 72, 72], ([1, 3], [4, 2]), true, false)
        @test GETTLayout.createGETTLayout(args...; bounds = (0, 0)) == GETTLayout.createGETTLayout(args...)
        masked = GETTLayout.createGETTLayout(args...; bounds = (5184, 0))
        @test masked <: GETTLayout.GETTLayoutMasked{Float16, GETTLayout.createGETTLayout(args...), 5184, 0}
        @test !GemmKernels.require_tile_sized_global(masked)
        @test GETTLayout.createGETTLayout(args...; indexing = :lean32, bounds = (0, 5184)) <:
              GETTLayout.GETTLayoutMasked{Float16, <:GETTLayout.GETTLayoutLeanColMajor}
        # loads beyond the bound are clamped to the last tile within it
        tile = GemmKernels.Tiling.Tile{(M = 8, K = 1), (:M, :K), Tuple{Int, Int}}((M = 5184, K = 3), (M = 8, K = 0))
        @test GETTLayout.clamp_tile(masked, tile).base.M + 8 == 5176
        @test GETTLayout.clamp_tile(masked, tile).base.K == 3
        tile = GemmKernels.Tiling.Tile{(M = 8, K = 1), (:M, :K), Tuple{Int, Int}}((M = 64, K = 3), (M = 8, K = 0))
        @test GETTLayout.clamp_tile(masked, tile) == tile
    end

    if CUDA.functional()
        kernels = Any[Kernel.matmul_singlestage, Kernel.matmul_pipelined, Kernel.matmul_pipelined_ng]
        capability(device()) ≥ v"8.0" && push!(kernels, Kernel.matmul_pipelined_async(3))

        # (name, extents, block shape, PERM_M, PERM_N, PERM_K, A col-major, B col-major):
        # a GEMM-like contraction with M = 200 and N = 136 (64-wide tiles: 3.1 and 2.1 tiles),
        # and TCCG TC20's pattern with M = N = 576 (128-wide tiles: 4.5). Every mode is a
        # multiple of 8, as the vectorised copies of the GETT layouts require.
        contractions = [("1.2-1.3-3.2", [200, 136, 128], (64, 64, 32), [1], [2], [3], true, true),
                        ("1.2.3.4-1.5.2.6-4.6.3.5", [24, 24, 24, 24, 16, 16], (128, 128, 32),
                         [1, 2], [4, 3], [6, 5], true, false)]

        run_contraction(problem, params) = begin
            data = allocate_data(problem)
            initialize_data(problem, data...)
            reference = Array(calculate_reference(problem, data...))
            args = try
                prepare(problem, data...; params...)
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            initialize_data(problem, data...; params...)
            result = Array(execute(problem, data...; args...))
            verify(problem, reference, result)
        end

        @testcase "$name, $kernel, $edge, indexing = $indexing" for (name, extents, (BM, BN, BK), PERM_M, PERM_N, PERM_K, a_col, b_col) in contractions,
                                                                   kernel in kernels, edge in ["pad", "mask"],
                                                                   indexing in ["default", "lean32"]
            problem = WMMATensorContraction(; name, extents, data_type = Float16, compute_type = Float16,
                                              accumulate_type = Float32, zero_c = true)
            params = (; BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, WARPS_M = 2, WARPS_N = 2,
                        OP_M = 16, OP_N = 16, OP_K = 16, kernel,
                        is_A_col_major = a_col, is_B_col_major = b_col, is_D_col_major = true,
                        PERM_M, PERM_N, PERM_K, cta_swizzle = CTASwizzle.Identity,
                        INDEXING = indexing, EDGE = edge)
            @test run_contraction(problem, params)
        end

        @testcase "split-K with partial edge tiles" begin
            problem = WMMATensorContraction(; name = "1.2-1.3-3.2", extents = [200, 136, 512],
                                              data_type = Float16, compute_type = Float16,
                                              accumulate_type = Float32, zero_c = true)
            for split in [2, 4]
                params = (; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                            OP_M = 16, OP_N = 16, OP_K = 16, kernel = Kernel.matmul_singlestage,
                            is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                            PERM_M = [1], PERM_N = [2], PERM_K = [3], cta_swizzle = CTASwizzle.Identity,
                            EDGE = "mask", SPLIT_K = split)
                @test run_contraction(problem, params)
            end
        end

        @testcase "dimensions that are not multiples of the copy tiles are rejected" begin
            # M = 100 is not a multiple of the 8-element copies of A
            problem = WMMATensorContraction(; name = "1.2-1.3-3.2", extents = [100, 136, 128],
                                              data_type = Float16, compute_type = Float16,
                                              accumulate_type = Float32, zero_c = true)
            params = (; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                        OP_M = 16, OP_N = 16, OP_K = 16, kernel = Kernel.matmul_singlestage,
                        is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                        PERM_M = [1], PERM_N = [2], PERM_K = [3], cta_swizzle = CTASwizzle.Identity,
                        EDGE = "mask")
            data = allocate_data(problem)
            @test_throws GemmKernels.ConfigError try
                prepare(problem, data...; params...)
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
        end
    end
end
