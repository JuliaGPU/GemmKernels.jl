using CUDA
using GemmKernels

M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

transpose_a = (ARGS[4] == "n" ? false : ARGS[4] == "t" ? true : error("Invalid memory layout for A: $(ARGS[4])"))
transpose_b = (ARGS[5] == "n" ? false : ARGS[5] == "t" ? true : error("Invalid memory layout for B: $(ARGS[5])"))

function benchmark_matmul(a, b, c, d)
    alpha = 2
    beta  = 3

    CUDA.@sync begin
        conf = GemmKernels.get_config(
            gemm_shape = (M = M, N = N, K = K),
            operator = Operator.WMMAOp{16, 16, 16},
            global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},
            global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

            global_c_layout = Layout.AlignedColMajor{Float32},
            global_d_layout = Layout.AlignedColMajor{Float32},

            is_a_col_major = !transpose_a,
            is_b_col_major = !transpose_b,
                                )

        GemmKernels.matmul(a, b, c, d, conf;
                           transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                           transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta))
    end
end

a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
c_h = rand(Float32, (M, N))

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
