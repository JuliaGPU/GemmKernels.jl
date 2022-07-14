using BenchmarkTools
using GemmKernels, CUDA
using CUDA: unsafe_free!

using StableRNGs
rng = StableRNG(123)

@info "Loading benchmarks"
SUITE = BenchmarkGroup()

let group = addgroup!(SUITE, "wmma")
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
                teardown=(unsafe_free!(a); unsafe_free!(b); unsafe_free!(c))
            )
    end
end

@info "Warming-up benchmarks"
warmup(SUITE; verbose=false)

paramsfile = joinpath(@__DIR__, "params.json")
if !isfile(paramsfile)
    @info "Tuning benchmarks"
    tune!(SUITE)
    BenchmarkTools.save(paramsfile, params(SUITE))
else
    loadparams!(SUITE, BenchmarkTools.load(paramsfile)[1], :evals, :samples)
end

@info "Running benchmarks"
results = run(SUITE; verbose=true)
println(results)

# write out the results
BenchmarkTools.save(joinpath(@__DIR__, "results.json"), results)

# compare against previous results
# TODO: store these results so that we can compare when benchmarking PRs
reference_path = joinpath(@__DIR__, "reference.json")
if ispath(reference_path)
    reference = BenchmarkTools.load(reference_path)[1]
    comparison = judge(minimum(results), minimum(reference))

    println("Improvements:")
    println(improvements(comparison))

    println("Regressions:")
    println(regressions(comparison))
end
