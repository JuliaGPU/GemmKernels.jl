using CUDA
using GemmKernels
using Test
using Statistics
using Printf

CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_DEFAULT_MATH)

function fpu_impl(transpose_a, transpose_b, alpha, a, b, beta, c, d, gemm_shape)
    conf = GemmKernels.get_config(
        gemm_shape = gemm_shape,
        block_shape = (M = 64, N = 64, K = 64),
        operator = Operator.FPUOp{8, 8, 1, Float32},
        global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float32} : Layout.AlignedColMajor{Float32},
        global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float32} : Layout.AlignedColMajor{Float32},

        global_c_layout = Layout.AlignedColMajor{Float32},
        global_d_layout = Layout.AlignedColMajor{Float32},

        is_a_col_major = !transpose_a,
        is_b_col_major = !transpose_b,
    )

    CUDA.@sync begin
        GemmKernels.matmul(
            a, b, c, d, conf;
            transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
            transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
            kernel = Kernel.matmul_singlestage
        )
    end
end

function main()
    # @printf("N,min,max,median,mean,std\n")

    for i = 7:7, transpose_a = [false], transpose_b = [false]
        M = 2 ^ i
        N = 2 ^ i
        K = 2 ^ i
        gemm_shape = (M = M, N = N, K = K)

        alpha = rand(Float32)
        beta = rand(Float32)

        a_h = rand(Float32, (M, K)) / sqrt(Float32(K))
        b_h = rand(Float32, (K, N)) / sqrt(Float32(K))
        c_h = rand(Float32, (M, N))

        # Transpose input if necessary
        a_h = transpose_a ? transpose(a_h) : a_h
        b_h = transpose_b ? transpose(b_h) : b_h

        a = CuArray(a_h)
        b = CuArray(b_h)
        c = CuArray(c_h)
        d = similar(c)

        # test_result = Float32(alpha) * Float32.(a_h) * Float32.(b_h) + beta * c_h

        fpu_impl(transpose_a, transpose_b, alpha, a, b, beta, c, d, gemm_shape)
        # @show @test all(isapprox.(test_result, Matrix(d); rtol = sqrt(eps(Float32))))
        fpu_impl(transpose_a, transpose_b, alpha, a, b, beta, c, d, gemm_shape)

        times = []
        for j = 1:10
            # time = CUDA.@elapsed fpu_impl(transpose_a, transpose_b, alpha, a, b, beta, c, d, gemm_shape)
            # push!(times, time)

            CUDA.@profile fpu_impl(transpose_a, transpose_b, alpha, a, b, beta, c, d, gemm_shape)
        end

        # times .= times .* 1e6

        # @printf(
        #     "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        #     N,
        #     minimum(times),
        #     maximum(times),
        #     median(times),
        #     mean(times),
        #     std(times),
        # )
    end
end

isinteractive() || main()