using CUDA
using GemmKernels
using LinearAlgebra

CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)

@test_if "blas" @testset "BLAS API" begin
    @testset "WMMA GEMM $(A_type)*$(B_type)+$(CD_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true],
        (A_type, B_type, CD_type, min_dimension) in [(Float16, Float16, Float16, 256), (Float16, Float16, Float32, 128)]

        @testset "(M = $M, N = $N, K = $K)" for M in min_dimension .* [1, 2],
            N in min_dimension .* [1, 2],
            K in min_dimension .* [1, 2]

            alpha = rand(A_type)
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
            GemmKernels.BLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_gemmkernels)

            c_cublas = CuArray(c_h)
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_cublas)

            @test all(isapprox.(Array(c_gemmkernels), Array(c_cublas); rtol=sqrt(eps(A_type))));
        end
    end

    @testset "WMMA GEMM (A = diagonal, B = $( !transpose_b ? 'N' : 'T' ))" for transpose_b = [false, true]
        @testset "(M = $M, N = $N, K = $K)" for M in [128, 256],
            N in [128, 256],
            K in [M]

            transpose_a = false

            alpha = rand(Float32)
            beta = rand(Float32)

            a_h = rand(Float16, M);
            b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
            c_h = rand(Float32, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = Diagonal(CuArray(a_h))
            b   = CuArray(b_h)

            c_gemmkernels = CuArray(c_h)
            GemmKernels.BLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c_gemmkernels)

            c_cublas = CuArray(c_h)
            CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, CuArray(Diagonal(a_h)), b, beta, c_cublas)

            @test all(isapprox.(Array(c_gemmkernels), Array(c_cublas); rtol=sqrt(eps(Float16))));
        end
    end
end
