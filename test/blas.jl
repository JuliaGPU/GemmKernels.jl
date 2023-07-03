using CUDA
using GemmKernels
using LinearAlgebra

CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

@testset "BLAS API" begin
    @testset "WMMA GEMM $(AB_type)*$(AB_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true],
        (AB_type, CD_type, min_dimension) in [(Float16, Float16, 256), (Float16, Float32, 128)]

        @testcase "(M = $M, N = $N, K = $K)" for M in min_dimension .* [1, 2],
            N in min_dimension .* [1, 2],
            K in min_dimension .* [1, 2]

            alpha = rand(AB_type)
            beta  = rand(CD_type)

            a_h = rand(AB_type, (M, K)) / sqrt(AB_type(K))
            b_h = rand(AB_type, (K, N)) / sqrt(AB_type(K))
            c_h = rand(CD_type, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)

            c_gemmkernels = CuArray(c_h)
            GemmKernels.BLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_gemmkernels; wmma=true)

            c_cublas = CuArray(c_h)
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_cublas)

            @test Array(c_gemmkernels) ≈ Array(c_cublas) rtol=sqrt(eps(AB_type))
        end
    end

    @testset "FPU GEMM $(A_type)*$(B_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true],
        (A_type, B_type, CD_type, min_dimension) in [(Float32, Float32, Float32, 128)]

        @testcase "(M = $M, N = $N, K = $K)" for M in min_dimension .* [1, 2],
            N in min_dimension .* [1, 2],
            K in min_dimension .* [1, 2]

            compute_type = promote_type(A_type, B_type)
            alpha = rand(compute_type)
            beta  = rand(CD_type)

            a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
            b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            c_h = rand(CD_type, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)

            c_gemmkernels = CuArray(c_h)
            GemmKernels.BLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_gemmkernels; wmma=false)

            c_cublas = CuArray(c_h)
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_cublas)

            @test Array(c_gemmkernels) ≈ Array(c_cublas) rtol=sqrt(eps(compute_type))
        end
    end
end
