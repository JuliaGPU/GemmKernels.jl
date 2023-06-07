using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Statistics
using Printf

CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_DEFAULT_MATH)

gemm_shape = eval(Meta.parse(ARGS[1]))
gemm_names = (:M, :N, :K)
gemm_shape = NamedTuple{gemm_names}(gemm_shape)

compute_type = eval(Meta.parse(ARGS[2]))
data_type = eval(Meta.parse(ARGS[3]))

function main()
    # ----
    # GEMM
    # ----

    alpha = rand(compute_type)

    a = CuArray(rand(compute_type, (gemm_shape.M, gemm_shape.K)) / sqrt(compute_type(gemm_shape.K)))
    b = CuArray(rand(compute_type, (gemm_shape.K, gemm_shape.N)) / sqrt(compute_type(gemm_shape.K)))

    beta = rand(data_type)
    c = CuArray(rand(data_type, (gemm_shape.M, gemm_shape.N)))
    d = similar(c)


    # ------
    # Matmul
    # ------

    CUDA.@profile begin
        for i = 1 : 10
            CUDA.@sync begin
                CUDA.CUBLAS.gemmEx!(
                    'N', 'N',
                    alpha,
                    a,
                    b,
                    beta,
                    c
                )
            end
        end
    end
end

isinteractive() || main()