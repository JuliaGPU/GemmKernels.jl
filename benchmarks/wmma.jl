group = addgroup!(SUITE, "wmma")

for N in [128, 16384], a_layout in ['N', 'T'], b_layout in ['N', 'T']
    M = N
    K = N

    a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
    b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
    c_h = rand(Float32, (M, N))

    # Transpose input if necessary
    a_h = a_layout == 'T' ? transpose(a_h) : a_h
    b_h = b_layout == 'T' ? transpose(b_h) : b_h

    alpha = rand(Float32)
    beta = rand(Float32)

    a = CuArray(a_h)
    b = CuArray(b_h)
    c = CuArray(c_h)

    group["Float16*Float16=Float32 (N=$N, A=$a_layout, B=$b_layout)"] =
        @benchmarkable(
            CUDA.@sync(GemmKernels.BLAS.gemmEx!($a_layout, $b_layout, $alpha, a, b, $beta, c)),
            setup=(a=CuArray($a_h); b=CuArray($b_h); c=CuArray($c_h)),
            teardown=(CUDA.unsafe_free!(a); CUDA.unsafe_free!(b); CUDA.unsafe_free!(c))
        )
end
