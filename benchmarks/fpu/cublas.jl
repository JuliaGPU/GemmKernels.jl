using CUDA
using Test
using Statistics
using Printf

CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_DEFAULT_MATH)

function cublas_impl(transpose_a, transpose_b, alpha, a, b, beta, c)
    CUDA.CUBLAS.gemmEx!(
        !transpose_a ? 'N' : 'T',
        !transpose_b ? 'N' : 'T',
        alpha,
        a,
        b,
        beta,
        c
    )
    c
end

@printf("N,min,max,median,mean,std\n")

for i = 7:12, transpose_a = [false], transpose_b = [false]
    M = 2 ^ i
    N = 2 ^ i
    K = 2 ^ i

    alpha = rand(Float16)
    beta = rand(Float32)

    a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
    b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
    c_h = rand(Float32, (M, N))

    # Transpose input if necessary
    a_h = transpose_a ? transpose(a_h) : a_h
    b_h = transpose_b ? transpose(b_h) : b_h

    a = CuArray(a_h)
    b = CuArray(b_h)
    c = CuArray(c_h)
    d = copy(c)

    test_result = alpha * a_h * b_h + beta * c_h

    cublas_impl(transpose_a, transpose_b, alpha, a, b, beta, c)
    @test all(isapprox.(test_result, Matrix(c); rtol = sqrt(eps(Float32))))
    cublas_impl(transpose_a, transpose_b, alpha, a, b, beta, c)

    times = []
    for j = 1:100
        time = CUDA.@elapsed cublas_impl(transpose_a, transpose_b, alpha, a, b, beta, c)
        push!(times, time)

        c = copy(d)
    end
    times .= times .* 1e6

    @printf(
        "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        N,
        minimum(times),
        maximum(times),
        median(times),
        mean(times),
        std(times),
    )
end
