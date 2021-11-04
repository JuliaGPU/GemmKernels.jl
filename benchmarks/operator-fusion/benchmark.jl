using CUDA
using BenchmarkTools
using GemmKernels
using Statistics
using Printf

function summarise_bench(b)
    mu = mean(b.times)
    sigma = std(b.times)

    if mu < 1e3
        mu, sigma, unit = mu, sigma, "ns"
    elseif mu < 1e6
        mu, sigma, unit = mu / 1e3, sigma / 1e3, "μs"
    elseif mu < 1e9
        mu, sigma, unit = mu / 1e6, sigma / 1e6, "ms"
    else
        mu, sigma, unit = mu / 1e9, sigma / 1e9, "s"
    end

    println(string(@sprintf("(%.2f ± %.2f)", mu, sigma), " ", unit))
end

M = 4096
N = 4096
K = 4096

transpose_a = false
transpose_b = false

f(x) = max(x, 0)
g(x) = x + 1

a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
c_h = rand(Float32, (M, N))

a   = CuArray(a_h)
b   = CuArray(b_h)
c   = CuArray(c_h)

bias = CuArray(rand(Float32, (1, N)))

function bench_cublas(a, b, c, M, N, K, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : 100
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
        end
    end
end

function bench_cublas_relu(a, b, c, M, N, K, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : 100
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
            c = f.(c)
        end
    end
end

function bench_cublas_bias(a, b, c, bias, M, N, K, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : 100
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
            c = c .+ bias
        end
    end
end

function bench_cublas_biasrelu(a, b, c, bias, M, N, K, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : 100
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
            c = f.(c .+ bias)
        end
    end
end

function bench_cublas_biasrelutwice(a, b, c, bias, M, N, K, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : 100
            c = f.(c)
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
            c = f.(c .+ bias)
        end
    end
end

function bench_cublas_biasrelutwice_ab_elop(a, b, c, bias, M, N, K, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : 100
            a = g.(a)
            b = g.(b)
            c = f.(c)
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
            c = f.(c .+ bias)
        end
    end
end

function bench_gemmkernels(a, b, c, M, N, K, transpose_a, transpose_b)
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

        for i = 1 : 100
            GemmKernels.matmul(a, b, c, c, conf;
                               kernel = Kernel.matmul_pipelined)
        end
    end
end

function bench_gemmkernels_relu(a, b, c, M, N, K, transpose_a, transpose_b)
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

        for i = 1 : 100
            GemmKernels.matmul(a, b, c, c, conf;
                               transform_regs_to_shared_d = Transform.Elementwise(f),
                               kernel = Kernel.matmul_pipelined)
        end
    end
end

function bench_gemmkernels_bias(a, b, c, bias, M, N, K, transpose_a, transpose_b)
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

        for i = 1 : 100
            GemmKernels.matmul(a, b, c, c, conf;
                               epilogue = Epilogue.Bias(pointer(bias)),
                               kernel = Kernel.matmul_pipelined)
        end
    end
end

function bench_gemmkernels_biasrelu(a, b, c, bias, M, N, K, transpose_a, transpose_b)
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

        for i = 1 : 100
            GemmKernels.matmul(a, b, c, c, conf;
                               transform_regs_to_shared_d = Transform.Elementwise(f),
                               epilogue = Epilogue.Bias(pointer(bias)),
                               kernel = Kernel.matmul_pipelined)
        end
    end
end

function bench_gemmkernels_biasrelutwice(a, b, c, bias, M, N, K, transpose_a, transpose_b)
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

        for i = 1 : 100
            GemmKernels.matmul(a, b, c, c, conf;
                               transform_shared_to_regs_c = Transform.Elementwise(f),
                               transform_regs_to_shared_d = Transform.Elementwise(f),
                               epilogue = Epilogue.Bias(pointer(bias)),
                               kernel = Kernel.matmul_pipelined)
        end
    end
end

function bench_gemmkernels_biasrelutwice_ab_elop(a, b, c, bias, M, N, K, transpose_a, transpose_b)
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

        for i = 1 : 100
            GemmKernels.matmul(a, b, c, c, conf;
                               transform_shared_to_regs_a = Transform.Elementwise(g),
                               transform_shared_to_regs_b = Transform.Elementwise(g),
                               transform_shared_to_regs_c = Transform.Elementwise(f),
                               transform_regs_to_shared_d = Transform.Elementwise(f),
                               epilogue = Epilogue.Bias(pointer(bias)),
                               kernel = Kernel.matmul_pipelined)
        end
    end
end
