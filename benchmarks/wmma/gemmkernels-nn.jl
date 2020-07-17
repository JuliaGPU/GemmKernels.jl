using CUDA
using GemmKernels

M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

function benchmark_matmul(a, b, c, d)
    CUDA.@sync begin
        conf = GemmKernels.get_config(
            gemm_shape = (M = M, N = N, K = K),
            operator = Operator.WMMAOp{16, 16, 16},

            global_a_layout = Layout.AlignedColMajor{Float16},
            global_b_layout = Layout.AlignedColMajor{Float16},

            global_c_layout = Layout.AlignedColMajor{Float32},
            global_d_layout = Layout.AlignedColMajor{Float32},

            transform_shared_to_regs_a = Transform.Elementwise(x -> x * 2),
            transform_shared_to_regs_c = Transform.Elementwise(x -> x * 3),

            is_a_col_major = true,
            is_b_col_major = true,
                                )
        GemmKernels.matmul(a, b, c, d, conf)
    end
end

a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
c_h = rand(Float32, (M, N))

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
