using CUDA
using GemmKernels

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

################################################################################

@testset "cuTENSOR baseline" begin
    # The operator-fusion benchmarks compare against cuTENSOR with the elementwise
    # operations it cannot fuse applied separately: Leaky ReLU on A and B before the
    # contraction, and on D after it. D must therefore be leaky_relu(α leaky_relu(A) leaky_relu(B)).
    @testcase "Leaky ReLU applied to the output" begin
        problem = WMMATensorContraction(; name = "1.2-1.3-3.2", extents = [64, 48, 32],
                                          data_type = Float16, compute_type = Float16,
                                          accumulate_type = Float32, zero_c = true)
        a, b, c, d = allocate_data(problem)
        extents(t) = problem.extents[problem.modes[t]]
        # A ≥ 0 and B ≤ 0 make every element of D negative, so that each of the three
        # Leaky ReLUs changes the result (with random signs, D is almost entirely positive
        # and a missing Leaky ReLU on D goes unnoticed).
        A = Float16.(abs.(randn(extents(2)...)))
        B = Float16.(-abs.(randn(extents(3)...)))
        # the baseline uses the first prod(extents) elements of each buffer as the tensor
        fill_start!(buffer, x) = copyto!(buffer, vcat(vec(x), zeros(eltype(buffer), length(buffer) - length(x))))
        fill_start!(a, A)
        fill_start!(b, B)
        fill!(c, 0)

        baseline = prepare_baseline(problem, a, b, c, d)
        execute_baseline(problem, a, b, c, d; baseline..., elementwise_op = "unsupported")
        D = reshape(Float32.(Array(d)[1:prod(extents(1))]), extents(1)...)

        leaky_relu(x) = x > 0 ? x : 0.01f0 * x
        reference = leaky_relu.(Float32(problem.alpha) .* (leaky_relu.(Float32.(A)) * leaky_relu.(Float32.(B))))
        @test all(<(0), reference)
        @test isapprox(D, reference; rtol = 1e-2)
    end
end
