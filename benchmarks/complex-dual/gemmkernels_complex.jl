using CUDA
using GemmKernels

M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

transpose_a = (ARGS[4] == "n" ? false : ARGS[4] == "t" ? true : error("Invalid memory layout for A: $(ARGS[4])"))
transpose_b = (ARGS[5] == "n" ? false : ARGS[5] == "t" ? true : error("Invalid memory layout for B: $(ARGS[5])"))

function benchmark_matmul(a, b, c, d)
    CUDA.@sync begin
        conf = GemmKernels.get_config(
                gemm_shape = (M = M, N = N, K = K),
                operator = Operator.WMMAComplexOp{16, 16, 16},

                global_a_layout = transpose_a ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
                global_b_layout = transpose_b ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
                global_c_layout = Layout.InterleavedColMajor{Float32},
                global_d_layout = Layout.InterleavedColMajor{Float32},

                shared_a_layout = transpose_a ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                shared_b_layout = transpose_b ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                shared_c_layout = Layout.SplitColMajor{Float32},
                shared_d_layout = Layout.SplitColMajor{Float32},

                warps_per_block = 8,

                compute_warp = (M = 16, N = 32, K = 16),

                block_shape = (M = 64, N = 64, K = 32),

                mem_a_warp = transpose_a ? (M = 4, K = 32) : (M = 64, K = 2),
                mem_b_warp = transpose_b ? (K = 2, N = 64) : (K = 32, N = 4),
                mem_cd_warp = (M = 64, N = 1),

                mem_a_thread = transpose_a ? (M = 1, K = 4) : (M = 4, K = 1),
                mem_b_thread = transpose_b ? (K = 1, N = 4) : (K = 4, N = 1),
                mem_cd_thread = (M = 2, N = 1),

                is_a_col_major = !transpose_a,
                is_b_col_major = !transpose_b
            )

        GemmKernels.matmul(a, b, c, d, conf;
                           kernel = Kernel.matmul_pipelined)
    end
end

a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K))
b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K))
c_h = rand(Complex{Float32}, (M, N))

# Transpose input if necessary
a_h = transpose_a ? transpose(a_h) : a_h
b_h = transpose_b ? transpose(b_h) : b_h

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
