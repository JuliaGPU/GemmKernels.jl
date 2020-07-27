using CUDA
using GemmKernels

M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

function benchmark_matmul(a, b, c, d)
    CUDA.@sync begin
        conf = GemmKernels.get_config(
                gemm_shape = (M = M, N = N, K = K),
                operator = Operator.WMMAComplexOp{16, 16, 16},

                global_a_layout = Layout.InterleavedColMajor{Float16},
                global_b_layout = Layout.InterleavedColMajor{Float16},
                global_c_layout = Layout.InterleavedColMajor{Float32},
                global_d_layout = Layout.InterleavedColMajor{Float32},

                shared_a_layout = Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                shared_b_layout = Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                shared_c_layout = Layout.SplitColMajor{Float32},
                shared_d_layout = Layout.SplitColMajor{Float32},

                warps_per_block = 8,

                compute_warp = (M = 16, N = 32, K = 16),

                block_shape = (M = 64, N = 64, K = 32),

                mem_a_warp = (M = 64, K = 2),
                mem_b_warp = (K = 32, N = 4),
                mem_cd_warp = (M = 64, N = 1),

                mem_a_thread = (M = 4, K = 1),
                mem_b_thread = (K = 4, N = 1),
                mem_cd_thread = (M = 2, N = 1),

                is_a_col_major = true,
                is_b_col_major = true
            )

        GemmKernels.matmul(a, b, c, d, conf)
    end
end

a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K))
b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K))
c_h = rand(Complex{Float32}, (M, N))

a   = CuArray(a_h)
b   = CuArray(b_h)
c   = CuArray(c_h)
d   = similar(c)

# warmup
benchmark_matmul(a, b, c, d)

# profile
for i = 1 : 10
    CUDA.@profile benchmark_matmul(a, b, c, d)
end
