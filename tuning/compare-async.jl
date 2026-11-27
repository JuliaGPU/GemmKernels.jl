# Compare the asynchronous-copy prototype kernel (Kernel.matmul_pipelined_async) against
# the existing kernels, on a plain GEMM and on some TCCG tensor contractions.
#
# For every problem, we draw a random sample of "base" configurations (block shape, warps,
# operator shape, layouts, permutations, swizzle) and run *every* kernel variant on each
# of them, so that kernels are compared on the same configurations (a paired comparison),
# not on different random draws. Every result is verified against cuBLAS/cuTENSOR.
#
# Usage (from the repository root, on an sm_80+ GPU):
#
#     julia --project=tuning tuning/compare-async.jl [TC names...]
#
# Environment variables:
#     SAMPLES=60        base configurations per problem
#     RUNS=10           timed runs per configuration (minimum is reported)
#     SEED=0            random seed for the sample
#     GEMM=4096         size of the square FP16 GEMM (0 to skip)
#     OUT=compare-async.csv
#
# Note: a configuration that fails to compile is recorded and skipped, but one that crashes
# the GPU (e.g. an illegal memory access) aborts the whole script; re-run under
# `compute-sanitizer` to investigate. Make sure the kernel
# cache (~/.julia/scratchspaces/*/matmul_cache) does not contain kernels compiled by
# older versions of GemmKernels: older versions did not include the kernel in the cache
# key.

using CUDA, GemmKernels, JSON, Random, Printf, DataFrames
using Base.Iterators: product
using Combinatorics: permutations

# Total time of the kernels in a CUDA.@profile result. Older CUDA.jl versions return the
# device events as a DataFrame, newer ones as a NamedTuple of columns; DataFrame() takes both.
device_time(prof) = (events = DataFrame(prof.device); sum(events.stop .- events.start))

include(joinpath(@__DIR__, "..", "configs", "configs.jl"))

const SAMPLES = parse(Int, get(ENV, "SAMPLES", "60"))
const RUNS = parse(Int, get(ENV, "RUNS", "10"))
const SEED = parse(Int, get(ENV, "SEED", "0"))
const GEMM_SIZE = parse(Int, get(ENV, "GEMM", "4096"))
const OUT = get(ENV, "OUT", "compare-async.csv")

const KERNELS = [
    "singlestage" => Kernel.matmul_singlestage,
    "pipelined" => Kernel.matmul_pipelined,
    "pipelined_ng" => Kernel.matmul_pipelined_ng,
    "async_2" => Kernel.matmul_pipelined_async(2),
    "async_3" => Kernel.matmul_pipelined_async(3),
    "async_4" => Kernel.matmul_pipelined_async(4),
]

# TCs that the paper discusses individually (Table I numbering), plus TC1.
const DEFAULT_TCS = ["abc-bda-dc", "ab-acd-dbc", "abcd-aebf-dfce", "abcd-eafd-fbec", "abcdef-gdac-efgb"]

const BLOCK_SHAPES = [(64, 64, 32), (128, 64, 32), (64, 128, 32), (128, 128, 32),
                      (128, 128, 64), (256, 128, 32), (128, 256, 32), (64, 64, 64)]
const WARPS = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2), (2, 4)]
const OPS = [(16, 16, 16), (8, 32, 16), (32, 8, 16)]
const SWIZZLES = ["horizontal-1", "horizontal-4", "vertical-4"]

swizzle_type(str) = startswith(str, "horizontal-") ?
    CTASwizzle.HorizontallyTiled{parse(Int, chopprefix(str, "horizontal-"))} :
    CTASwizzle.VerticallyTiled{parse(Int, chopprefix(str, "vertical-"))}

# time the kernels launched by `f`, using the CUDA profiler like tuning/tune.jl does
function measure(f)
    f(); synchronize()  # warm-up
    times = map(1:RUNS) do _
        prof = CUDA.@profile concurrent=false f()
        device_time(prof)
    end
    minimum(times)
end

function run_config(problem, data, reference, params, is_tc)
    # Planning and compiling happen on the host, so a failure there doesn't affect the GPU:
    # record it (like tuning/tune.jl does) and continue with the next configuration.
    args = try
        is_tc ? prepare(problem, data...; params...) : prepare(problem; params...)
    catch err
        isa(err, GemmKernels.ConfigError) && return "config_error", NaN
        isa(err, CUDA.InvalidIRError) && return "compilation_error", NaN
        isa(err, InterruptException) && rethrow()
        msg = first(split(sprint(showerror, err), '\n'))
        @warn "Unexpected error while compiling $(params.kernel) for $(repr(params)): $msg"
        return "unknown_error_during_prepare", NaN
    finally
        GemmKernels.Tensors.OVERRIDE_do_override = false
    end

    # An error while running a kernel (e.g. an illegal address) can leave the GPU unusable,
    # so that still aborts the script, after printing the configuration.
    is_tc && initialize_data(problem, data...; params...)
    result = try
        Array(execute(problem, data...; args...))
    catch err
        isa(err, GemmKernels.ConfigError) && return "config_error", NaN
        @error "Error while running $(params.kernel) for $(repr(params))"
        rethrow()
    end
    ok = is_tc ? verify(problem, reference, result) :
                 isapprox(result, reference; rtol = sqrt(eps(Float16)))
    ok || return "invalid_result", NaN

    return "success", measure(() -> execute(problem, data...; args...))
end

function sample_tc_configs(problem, rng)
    D, A, B = problem.modes
    perm_m = collect(permutations(intersect(D, A)))
    perm_n = collect(permutations(intersect(D, B)))
    perm_k = collect(permutations(intersect(A, B)))
    space = collect(product(BLOCK_SHAPES, WARPS, OPS, [false, true], [false, true], [false, true],
                            perm_m, perm_n, perm_k, SWIZZLES))
    map(rand(rng, space, SAMPLES)) do ((BM, BN, BK), (WM, WN), (OM, ON, OK), ca, cb, cd, pm, pn, pk, sw)
        (; BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, WARPS_M = WM, WARPS_N = WN,
           OP_M = OM, OP_N = ON, OP_K = OK,
           is_A_col_major = ca, is_B_col_major = cb, is_D_col_major = cd,
           PERM_M = pm, PERM_N = pn, PERM_K = pk, cta_swizzle = swizzle_type(sw))
    end
end

function sample_gemm_configs(rng)
    space = collect(product(BLOCK_SHAPES, WARPS, OPS))
    map(rand(rng, space, SAMPLES)) do ((BM, BN, BK), (WM, WN), (OM, ON, OK))
        (; BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, WARPS_M = WM, WARPS_N = WN,
           OP_M = OM, OP_N = ON, OP_K = OK, zero_c = true)
    end
end

function main(tc_names)
    @info "Comparing kernels on $(name(device())) (sm_$(CUDA.capability(device()).major)$(CUDA.capability(device()).minor)), $SAMPLES configurations per problem"
    CUDA.capability(device()) ≥ v"8.0" || @warn "This GPU does not support asynchronous copies; the async kernels will be skipped as config errors."

    suite = JSON.parsefile(joinpath(@__DIR__, "..", "test", "benchmark-suite.json"))
    problems = Any[]
    GEMM_SIZE > 0 && push!(problems, ("GEMM $(GEMM_SIZE)³ NN",
        WMMAMatrixMultiplication(; M = GEMM_SIZE, N = GEMM_SIZE, K = GEMM_SIZE, AB_type = Float16,
                                   CD_type = Float32, transpose_a = false, transpose_b = false, zero_c = true), false))
    for tc_name in tc_names
        el = only(filter(el -> el["name"] == tc_name, suite))
        push!(problems, (tc_name, WMMATensorContraction(; name = el["parseableName"], extents = el["extents"],
                                                         data_type = Float16, compute_type = Float16,
                                                         accumulate_type = Float32, zero_c = true), true))
    end

    io = open(OUT, "w")
    println(io, "problem,sample,kernel,status,time_s,config")
    summary = []
    for (label, problem, is_tc) in problems
        rng = MersenneTwister(SEED)
        data = allocate_data(problem)
        initialize_data(problem, data...)
        reference = Array(calculate_reference(problem, data...))
        configs = is_tc ? sample_tc_configs(problem, rng) : sample_gemm_configs(rng)

        best = Dict{String, Float64}()
        statuses = Dict{String, Dict{String, Int}}()
        for (i, config) in enumerate(configs), (kname, kernel) in KERNELS
            status, t = run_config(problem, data, reference, (; config..., kernel), is_tc)
            println(io, "\"$label\",$i,$kname,$status,$t,\"$(repr(config))\"")
            flush(io)
            st = get!(statuses, kname, Dict{String, Int}())
            st[status] = get(st, status, 0) + 1
            if status == "success"
                best[kname] = min(get(best, kname, Inf), t)
            end
        end

        println("\n== $label")
        for (kname, _) in KERNELS
            st = join(["$k=$v" for (k, v) in sort(collect(get(statuses, kname, Dict())))], " ")
            @printf("  %-13s best %10s   (%s)\n", kname,
                    haskey(best, kname) ? @sprintf("%.1f µs", 1e6 * best[kname]) : "-", st)
        end
        sync_best = minimum(get(best, k, Inf) for k in ("singlestage", "pipelined", "pipelined_ng"))
        async_best = minimum(get(best, k, Inf) for k in ("async_2", "async_3", "async_4"))
        if isfinite(sync_best) && isfinite(async_best)
            @printf("  best async vs best existing kernel: %.2fx\n", sync_best / async_best)
        end
        push!(summary, (label, sync_best, async_best))
    end
    close(io)
    println("\nPer-configuration results written to $OUT")
end

isinteractive() || main(isempty(ARGS) ? DEFAULT_TCS : ARGS)
