using GemmKernels

# Block shapes whose dimensions are not powers of two silently gave wrong results (e.g.
# BLOCK_M = 72 or 80 in tensor contractions), so get_config rejects them. Needs no GPU.

function wmma_config(block_shape; gemm_shape = (M = 1440, N = 1024, K = 1024))
    GemmKernels.get_config(;
        gemm_shape, block_shape,
        operator = Operator.WMMAOp{16, 16, 16, Float16, Float32},
        global_a_layout = Layout.UnsafeAlignedColMajor{Float16},
        global_b_layout = Layout.UnsafeAlignedColMajor{Float16},
        global_c_layout = Layout.UnsafeAlignedColMajor{Float32},
        global_d_layout = Layout.UnsafeAlignedColMajor{Float32},
        shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8},
        shared_b_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8},
        shared_c_layout = Layout.UnsafeAlignedColMajor{Float32},
        shared_d_layout = Layout.UnsafeAlignedColMajor{Float32},
        is_a_col_major = true, is_b_col_major = true)
end

@testset "Block shapes" begin
    @testcase "powers of two are accepted" begin
        for block_shape in [(M = 128, N = 128, K = 32), (M = 64, N = 32, K = 16), (M = 32, N = 256, K = 64)]
            @test wmma_config(block_shape; gemm_shape = (M = 1024, N = 1024, K = 1024)) isa GemmKernels.Config
        end
    end

    @testcase "other dimensions are rejected" begin
        # 1440 is a multiple of 72, 80, 96 and 160, so only the new check can reject these
        for block_shape in [(M = 72, N = 64, K = 32), (M = 80, N = 128, K = 32), (M = 96, N = 64, K = 32),
                            (M = 160, N = 64, K = 32), (M = 64, N = 64, K = 48)]
            gemm_shape = block_shape.K == 48 ? (M = 1024, N = 1024, K = 1440) : (M = 1440, N = 1024, K = 1024)
            err = try
                wmma_config(block_shape; gemm_shape)
                nothing
            catch err
                err
            end
            @test err isa GemmKernels.ConfigError
            @test err isa GemmKernels.ConfigError && occursin("powers of two", err.message)
        end
    end
end
