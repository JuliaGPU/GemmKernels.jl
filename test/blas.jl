using CUDA
using GemmKernels
using LinearAlgebra

function transposed_rand(typ, dims, transposed, scale=true)
    values = rand(typ, transposed ? reverse(dims) : dims)
    values ./ scale
    transposed ? values' : values
end

@testset "BLAS API" begin
    @testset "WMMA GEMM $(AB_type)*$(AB_type)=$(C_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for
            transpose_a = [false, true],
            transpose_b = [false, true],
            (AB_type, C_type, min_dimension) in [(Float16, Float16, 256), (Float16, Float32, 128)]

        @testcase "(M = $M, N = $N, K = $K)" for M in [10; min_dimension; 1000],
                                                 N in [10; min_dimension; 1000],
                                                 K in [10; min_dimension; 1000]

            alpha = rand(AB_type)
            beta  = rand(C_type)

            a_h = transposed_rand(AB_type, (M, K), transpose_a, sqrt(AB_type(K)))
            b_h = transposed_rand(AB_type, (K, N), transpose_b, sqrt(AB_type(K)))
            c_h = rand(C_type, (M, N))

            a   = CuArray(parent(a_h))
            b   = CuArray(parent(b_h))
            c   = CuArray(c_h)

            GemmKernels.BLAS.matmatmul!(c, transpose_a ? 'T' : 'N', transpose_b ? 'T' : 'N',
                                        a, b, alpha, beta; wmma=true)
            mul!(c_h, a_h, b_h, alpha, beta)

            @test c_h ≈ Array(c) rtol=sqrt(eps(AB_type))
        end
    end

    @testset "FPU GEMM $(A_type)*$(B_type)=$(C_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for
            transpose_a = [false, true],
            transpose_b = [false, true],
            (A_type, B_type, C_type, min_dimension) in [(Float32, Float32, Float32, 128)]

        @testcase "(M = $M, N = $N, K = $K)" for M in [10; min_dimension; 1000],
                                                 N in [10; min_dimension; 1000],
                                                 K in [10; min_dimension; 1000]

            compute_type = promote_type(A_type, B_type)
            alpha = rand(compute_type)
            beta  = rand(C_type)

            a_h = transposed_rand(A_type, (M, K), transpose_a, sqrt(A_type(K)))
            b_h = transposed_rand(B_type, (K, N), transpose_b, sqrt(B_type(K)))
            c_h = rand(C_type, (M, N))

            a   = CuArray(parent(a_h))
            b   = CuArray(parent(b_h))
            c   = CuArray(c_h)

            GemmKernels.BLAS.matmatmul!(c, transpose_a ? 'T' : 'N', transpose_b ? 'T' : 'N',
                                        a, b, alpha, beta; wmma=false)
            mul!(c_h, a_h, b_h, alpha, beta)

            @test c_h ≈ Array(c) rtol=sqrt(eps(compute_type))
        end
    end
end
