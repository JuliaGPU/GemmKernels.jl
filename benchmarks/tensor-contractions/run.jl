using BenchmarkTools
using GemmKernels
using GemmKernels.Tiling
using GemmKernels.Layout
using CUDA
using Test
using Statistics
using Printf

# TCCG benchmark 1: D_abc = A_bda * B_dc

# sizes for each dimension
SA = 64
SB = 32
SC = 2048
SD = 2048

A = CuArray(rand(Float16, (SB, SD, SA)))
B = CuArray(rand(Float16, (SD, SC)))

# layout for the A tensor
abstract type LayoutA{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    M = tile.base.M + tile.offset.M
    K = tile.base.K + tile.offset.K

    d = K

    a = M ÷ Base.size(workspace, 1)
    b = M % Base.size(workspace, 1)

    offset = 1 + b + d * Base.size(workspace, 1) + a * Base.size(workspace, 1) * Base.size(workspace, 2)

    Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
end

# layout for the B tensor
abstract type LayoutB{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    K = tile.base.K + tile.offset.K
    N = tile.base.N + tile.offset.N

    d = K
    c = N

    offset = 1 + d + c * Base.size(workspace, 1)

    Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
end

# layout for the C tensor
abstract type LayoutC{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    ntuple(Val(N)) do i
        VecElement{T}(zero(T))
    end
end

# layout for the D tensor
abstract type LayoutD{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.store!(::Type{LayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    for i = 1 : NUMEL
        M = tile.base.M + tile.offset.M
        N = tile.base.N + tile.offset.N

        c = N

        a = (M + i - 1) ÷ Base.size(workspace, 2)
        b = (M + i - 1) % Base.size(workspace, 2)

        @inbounds workspace[a + 1, b + 1, c + 1] = value[i].value
    end
end

# implementation using GemmKernels.jl
function gemmkernels_impl(;benchmark = false)
    D = CuArray(zeros(Float16, (SA, SB, SC)))

    M = SA * SB
    N = SC
    K = SD

    conf = GemmKernels.get_config(
                                  gemm_shape = (M = M, N = N, K = K),
                                  operator = Operator.WMMAOp{16, 16, 16, Float16},

                                  global_a_layout = LayoutA{Float16},
                                  global_b_layout = LayoutB{Float16},
                                  global_c_layout = LayoutC{Float16},
                                  global_d_layout = LayoutD{Float16},

                                  shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
                                  shared_b_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
                                  shared_c_layout = Layout.AlignedColMajor{Float16},
                                  shared_d_layout = Layout.AlignedColMajor{Float16},

                                  is_a_col_major = true,
                                  is_b_col_major = true,
                                 )

    if !benchmark
        GemmKernels.matmul(A, B, D, D, conf;
                        kernel = Kernel.matmul_pipelined
                        )
        D
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed GemmKernels.matmul(A, B, D, D, conf;
                            kernel = Kernel.matmul_pipelined
                        )
            push!(times, time)
        end

        times
    end
end

# cuTENSOR implementation
function cutensor_impl(;algo = CUDA.CUTENSOR.CUTENSOR_ALGO_DEFAULT, benchmark = false)
    D = CuArray(zeros(Float16, (SA, SB, SC)))

    plan = CUDA.CUTENSOR.plan_contraction(A, [ 'b', 'd', 'a' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          B, [ 'd', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          D, [ 'a', 'b', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          algo = algo,
                                          compute_type = Float32)



    if !benchmark
        CUDA.CUTENSOR.contraction!(1,
                                A, [ 'b', 'd', 'a' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                B, [ 'd', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                0,
                                D, [ 'a', 'b', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                compute_type = Float32,
                                plan = plan)
        D
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed CUDA.CUTENSOR.contraction!(1,
                                    A, [ 'b', 'd', 'a' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    B, [ 'd', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    0,
                                    D, [ 'a', 'b', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    compute_type = Float32,
                                    plan = plan)
            push!(times, time)
        end

        times
    end
end

# compare
function test()
    D_reference = cutensor_impl()
    D_gemmkernels = gemmkernels_impl()

    @test all(isapprox.(Array(D_reference), Array(D_gemmkernels); rtol = sqrt(eps(Float16))))
end

# Taken from BenchmarkTools.jl: src/trials.jl
function rmskew(t::Vector)
    st = sort(t)
    return st[1:(BenchmarkTools.skewcutoff(st) - 1)]
end

# Benchmark function
function bench()
    bench_cutensor = cutensor_impl(benchmark = true)

    bench_cutensor_gett = cutensor_impl(algo = CUDA.CUTENSOR.CUTENSOR_ALGO_GETT, benchmark = true)

    bench_gemmkernels = gemmkernels_impl(benchmark = true)

    for (title, bench) in [("cuTENSOR (let heuristic choose algorithm):", bench_cutensor),
                           ("cuTENSOR (force GETT):", bench_cutensor_gett),
                           ("GemmKernels:", bench_gemmkernels),
                           ("rmskew: cuTENSOR (let heuristic choose algorithm):", rmskew(bench_cutensor)),
                           ("rmskew: cuTENSOR (force GETT):", rmskew(bench_cutensor_gett)),
                           ("rmskew: GemmKernels:", rmskew(bench_gemmkernels))
                          ]
        println(title)
        bench .= bench .* 1e6 # convert seconds to us
        @printf("Samples:            %u\n", length(bench))
        @printf("Range (min .. max): %.2f us .. %.2f us\n", minimum(bench), maximum(bench))
        @printf("Median:             %.2f us\n", median(bench))
        @printf("Mean ± std:         %.2f us ± %.2f us\n", mean(bench), std(bench))
        println()
    end

    nothing
end

# Main entry point
function main()
    display(test())
    println()

    bench()
end

!isinteractive() && main()
