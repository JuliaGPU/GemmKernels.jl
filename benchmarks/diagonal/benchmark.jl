using CUDA
using BenchmarkTools
using GemmKernels
using Statistics
using Printf
using LinearAlgebra

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

    println(string(@sprintf("(%.3f ± %.3f)", mu, sigma), " ", unit))
end

M = 4096
N = 4096
K = 4096

transpose_a = false
transpose_b = false

a_h = rand(Float16, M);
b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
c_h = rand(Float32, (M, N))

# Transpose input if necessary
a_h = transpose_a ? transpose(a_h) : a_h
b_h = transpose_b ? transpose(b_h) : b_h

a_gk     = CuArray(a_h)
a_cublas = CuArray(Array(Diagonal(a_h)))
b        = CuArray(b_h)
c        = CuArray(c_h)

function bench_cublas(a, b, c, M, N, K, transpose_a, transpose_b, num_iterations = 100)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

    CUDA.@sync begin
        for i = 1 : num_iterations
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', Float32(1), a, b, Float32(1), c)
        end
    end
end

function bench_gemmkernels(a, b, c, M, N, K, transpose_a, transpose_b, num_iterations = 100)
    conf = GemmKernels.get_config(
                                  gemm_shape = (M = M, N = N, K = K),
                                  operator = Operator.WMMAOp{16, 16, 16},
                                  global_a_layout = Layout.Diagonal{Float16},
                                  global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

                                  global_c_layout = Layout.AlignedColMajor{Float32},
                                  global_d_layout = Layout.AlignedColMajor{Float32},

                                  shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},

                                  is_a_col_major = !transpose_a,
                                  is_b_col_major = !transpose_b,
                                 )
    CUDA.@sync begin
        for i = 1 : num_iterations
            GemmKernels.matmul(a, b, c, c, conf;)
        end
    end
end
