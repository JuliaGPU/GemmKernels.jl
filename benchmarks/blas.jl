group = addgroup!(SUITE, "BLAS")

function blas_benchmark(ab_type, cd_type, N, M=N, K=N; alpha=true, beta=false,
                        a_transpose=false, b_transpose=false)
    a_h = rand(ab_type, (M, K))
    b_h = rand(ab_type, (K, N))
    c_h = rand(cd_type, (M, N))

    a_h = a_transpose ? transpose(a_h) : a_h
    b_h = b_transpose ? transpose(b_h) : b_h

    a_layout = a_transpose ? 'T' : 'N'
    b_layout = b_transpose ? 'T' : 'N'

    # generate a name for the benchmark
    io = IOBuffer()
    print(io, ab_type)
    a_transpose && print(io, "'")
    print(io, "*")
    print(io, ab_type)
    b_transpose && print(io, "'")
    print(io, "=$cd_type ($N×$K×$N")
    alpha && print(io, ", alpha")
    beta && print(io, ", beta")
    print(io, ")")
    name = String(take!(io))

    group[name] = @benchmarkable(
        CUDA.@sync(GemmKernels.BLAS.gemmEx!($a_layout, $b_layout, $alpha, a, b, $beta, c)),
        setup=(a=CuArray($a_h); b=CuArray($b_h); c=CuArray($c_h)),
        teardown=(CUDA.unsafe_free!(a); CUDA.unsafe_free!(b); CUDA.unsafe_free!(c))
    )
end

for N in [256, 4096], (ab_type, cd_type) in [(Float16, Float16), (Float16, Float32)]
    # test the effect of alpha and beta
    for (alpha, beta) in [(true, true), (true, false), (false, true)]
        blas_benchmark(ab_type, cd_type, N; alpha, beta)
    end

    # test the effect of transposing
    for a_transpose in (true, false), b_transpose in (true, false)
        blas_benchmark(ab_type, cd_type, N; a_transpose, b_transpose)
    end
end
