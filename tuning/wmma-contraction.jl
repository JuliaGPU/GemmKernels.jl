## constants

using JSON
using Base.Iterators: product
using Combinatorics: permutations
using StatsBase
using LaTeXStrings

const data_type = Float16
const compute_type = Float16
const accumulate_type = Float32

const zero_c = true

config_path = joinpath(@__DIR__, "../test/benchmark-suite.json")
fp = open(config_path, "r")
const jsonData = JSON.parse(read(fp, String))


## configurations

include("../configs/configs.jl")

# GK_PROBLEMS selects contractions by their number in the TCCG suite (1-48, the order of
# test/benchmark-suite.json), e.g. GK_PROBLEMS=13,14,20; unset or empty means all of them.
function generate_problems()
    selection = strip(get(ENV, "GK_PROBLEMS", ""))
    selected = isempty(selection) ? collect(eachindex(jsonData)) :
                                    [parse(Int, strip(i)) for i in split(selection, ',')]
    all(in(eachindex(jsonData)), selected) ||
        error("GK_PROBLEMS=$selection: contraction numbers must be between 1 and $(length(jsonData))")
    problems = []
    for el in jsonData[selected]
        push!(problems, WMMATensorContraction(; name=el["parseableName"], extents=el["extents"],
                                                data_type, compute_type, accumulate_type, zero_c))
    end
    problems
end

function create_configs()
    DataFrame(
        # problem
        name=String[],
        extents=Vector{Int}[],

        # config
        BLOCK_M=Int[],
        BLOCK_N=Int[],
        BLOCK_K=Int[],
        WARPS_M=Int[],
        WARPS_N=Int[],
        OP_M=Int[],
        OP_N=Int[],
        OP_K=Int[],
        kernel_str=String[],
        ## layout
        is_A_col_major=Bool[],
        is_B_col_major=Bool[],
        is_D_col_major=Bool[],
        PERM_M=Vector{Int}[],
        PERM_N=Vector{Int}[],
        PERM_K=Vector{Int}[],
        ## CTA swizzling
        cta_swizzle_str=String[]
    )
end

# All kernels the GPU supports: the asynchronous-copy kernels need sm_80 or later. Exclude
# them with GEMMKERNELS_TUNE_ASYNC=0 (e.g. to reproduce sweeps that didn't have them).
const TUNE_ASYNC = get(ENV, "GEMMKERNELS_TUNE_ASYNC", "1") != "0" &&
                   CUDA.functional() && CUDA.capability(device()) >= v"8.0"
const KERNEL_STRINGS = vcat(["singlestage", "pipelined", "pipelined_ng"],
                            TUNE_ASYNC ? ["pipelined_async_2", "pipelined_async_3", "pipelined_async_4"] : String[])

function config_iterator(problem)
    param_product = shuffle_product(
        2 .^ (3:9),
        2 .^ (3:9),
        2 .^ (3:7),
        2 .^ (0:3),
        2 .^ (0:3),
        [(16, 16, 16), (8, 32, 16), (32, 8, 16)],
        KERNEL_STRINGS,

        [false, true],
        [false, true],
        [false, true],
        permutations(intersect(problem.modes[1], problem.modes[2])),
        permutations(intersect(problem.modes[1], problem.modes[3])),
        permutations(intersect(problem.modes[2], problem.modes[3])),

        ["horizontal-1", "horizontal-2", "horizontal-4", "horizontal-8", "horizontal-16",
         "vertical-1", "vertical-2", "vertical-4", "vertical-8", "vertical-16"]
    )

    return ((;
        problem.name,
        extents = [problem.extents...],

        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        WARPS_M,
        WARPS_N,
        OP_M, OP_N, OP_K,
        kernel_str,

        is_A_col_major,
        is_B_col_major,
        is_D_col_major,
        PERM_M,
        PERM_N,
        PERM_K,

        cta_swizzle_str
    ) for (
        BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N,
        (OP_M, OP_N, OP_K), kernel_str, is_A_col_major,
        is_B_col_major, is_D_col_major, PERM_M, PERM_N,
        PERM_K, cta_swizzle_str
    ) in param_product)
end

# Reject configurations that GemmKernels is certain to reject, without sending them to a
# compilation worker, where each costs over a second (setting up types and a plan). Every rule
# implies an error that the worker would raise: the ConfigErrors of `get_config` (block shape,
# operator shape, elements per thread, warp tile) and of `plan_matmul` (pipelined stages,
# shared memory, with a lower bound of the kernels' `shmem_size`), for the parameters that
# `prepare(::TensorContraction)` passes. Configurations with more than 32 accumulator fragments
# per warp fail to compile (an `InvalidIRError`: updating a tuple of more than 32 elements
# allocates). Returns the status to record, or `nothing` to compile the configuration.
# Checked against the outcomes in four sweeps (November 2026 trial, February 2026,
# December 2025): no configuration that compiled would have been rejected.
function precheck_config(config; max_shmem)
    (; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, kernel_str) = config
    warps = WARPS_M * WARPS_N
    rejected = "config_error_during_prepare"

    # get_config: the block shape is too small
    (BLOCK_M * BLOCK_K < 128 || BLOCK_K * BLOCK_N < 128 || BLOCK_K < 8) && return rejected
    # get_config: mismatch between the block shape and the operator shape
    (BLOCK_M < 2 * OP_M || BLOCK_N < 2 * OP_N || BLOCK_K < OP_K) && return rejected
    # get_config: at least one element per thread for copying A, B and C
    (BLOCK_M * BLOCK_K < 32 * warps || BLOCK_K * BLOCK_N < 32 * warps ||
     BLOCK_M * BLOCK_N < 32 * warps) && return rejected
    # get_config: the warp tile (BLOCK ÷ WARPS) must be a multiple of the operator shape
    warp_m, warp_n = BLOCK_M ÷ WARPS_M, BLOCK_N ÷ WARPS_N
    (warp_m == 0 || warp_n == 0 || warp_m % OP_M != 0 || warp_n % OP_N != 0) && return rejected
    # plan_matmul: the pipelined kernel needs at least two stages
    (kernel_str == "pipelined" && BLOCK_K < 2 * OP_K) && return rejected
    # plan_matmul: shared memory. A and B are Float16 (padding only adds), D is Float32.
    stages = kernel_str == "pipelined_ng" ? 2 :
             startswith(kernel_str, "pipelined_async_") ? parse(Int, chopprefix(kernel_str, "pipelined_async_")) : 1
    shmem_lower_bound = max(4 * BLOCK_M * BLOCK_N, stages * 2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N))
    shmem_lower_bound > max_shmem && return rejected

    # more than 32 accumulator fragments per warp do not compile
    (warp_m ÷ OP_M) * (warp_n ÷ OP_N) > 32 && return "compilation_error"

    return nothing
end

function select_configs(configs, problem)
    filter([:name, :extents] => (name, extents) -> name == problem.name && (length(extents) == length(problem.extents)) && all(extents .== problem.extents), configs; view=true)
end

function repr_row(row)
    io = IOBuffer()

    # tccg benchmark parseable name
    print(io, "$(row.name)")

    # details
    print(io, " ($(row.BLOCK_M)×$(row.BLOCK_N)×$(row.BLOCK_K) block")
    print(io, ", $(row.WARPS_M)×$(row.WARPS_N) warp")
    print(io, ", $(row.OP_M)×$(row.OP_N)×$(row.OP_K) operator")
    print(io, ", $(row.PERM_M)×$(row.PERM_N)×$(row.PERM_K) layout")
    if row.is_A_col_major || row.is_B_col_major || row.is_D_col_major
        print(io, ", ")
        row.is_A_col_major && print(io, "A ")
        row.is_B_col_major && print(io, "B ")
        row.is_D_col_major && print(io, "D ")
        print(io, "col-major")
    end
    print(io, ", $(row.cta_swizzle_str) swizzle")
    print(io, ", $(row.kernel_str) kernel)")

    return String(take!(io))
end

function create_params(row)
    BLOCK_M = row.BLOCK_M
    BLOCK_N = row.BLOCK_N
    BLOCK_K = row.BLOCK_K
    WARPS_M = row.WARPS_M
    WARPS_N = row.WARPS_N
    OP_M = row.OP_M
    OP_N = row.OP_N
    OP_K = row.OP_K
    kernel = kernel_string_to_function(row.kernel_str)
    is_A_col_major = row.is_A_col_major
    is_B_col_major = row.is_B_col_major
    is_D_col_major = row.is_D_col_major
    PERM_M = row.PERM_M
    PERM_N = row.PERM_N
    PERM_K = row.PERM_K
    cta_swizzle = cta_swizzle_string_to_function(row.cta_swizzle_str)

    (; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, kernel,
       is_A_col_major, is_B_col_major, is_D_col_major, PERM_M, PERM_N, PERM_K, cta_swizzle)
end

function kernel_string_to_function(str)
    if str == "singlestage"
        return Kernel.matmul_singlestage
    elseif str == "pipelined"
        return Kernel.matmul_pipelined
    elseif str == "pipelined_ng"
        return Kernel.matmul_pipelined_ng
    elseif startswith(str, "pipelined_async_")
        return Kernel.matmul_pipelined_async(parse(Int, chopprefix(str, "pipelined_async_")))
    else
        error("Unknown kernel string: $str")
    end
end

function cta_swizzle_string_to_function(str)
    if startswith(str, "horizontal-")
        N = parse(Int, chopprefix(str, "horizontal-"))
        return CTASwizzle.HorizontallyTiled{N}
    elseif startswith(str, "vertical-")
        N = parse(Int, chopprefix(str, "vertical-"))
        return CTASwizzle.VerticallyTiled{N}
    else
        error("Unknown CTA swizzle string: $str")
    end
end


## output

function plot_best_configs(all_configs, best_configs)
    markershapes = Dict(
        "NN" => :circle,
        "NT" => :dtriangle,
        "TN" => :diamond,
        "TT" => :cross
    )

    p = plot()
    plot!(guidefontsize=8)
    ylabel!("Performance relative to cuTENSOR [%]")

    problems = generate_problems()
    idx = 1:length(problems)

    labels = []
    ratios = []
    ratios_lo = []
    ratios_hi = []
    coverage = []

    for (i, problem) in enumerate(problems)
        name_idx = findfirst(el -> el["parseableName"] == problem.name, jsonData)
        name_idx == nothing && error("Unknown parseable name: $(problem.name)")
        push!(labels, "$i")

        best_config = select_configs(best_configs, problem)
        if isempty(best_config)
            push!(coverage, 0)
            push!(ratios, 0)
            push!(ratios_lo, 0)
            push!(ratios_hi, 0)
            continue
        end
        best_config = best_config[1, :]
        configs = select_configs(all_configs, problem)

        push!(ratios, 100 * perf_ratio(best_config.gemmkernels_times, best_config.baseline_times))
        push!(ratios_lo, 100 * perf_ratio_lo(best_config.gemmkernels_times, best_config.baseline_times))
        push!(ratios_hi, 100 * perf_ratio_hi(best_config.gemmkernels_times, best_config.baseline_times))
        push!(coverage, 100 * size(configs, 1) / length(config_iterator(problem)))
    end

    # determine colors based on a gradient: 100 is white, over 100 becomes green, below 100 becomes red
    colors = map(ratios) do ratio
        if ratio > 100
            # turn green: 25% faster is full grean
            alpha = clamp((ratio - 100) / 25, 0, 1)
            return RGBA(0, 1, 0, alpha)
        else
            # turn red towards 25% performance
            alpha = clamp((100 - ratio) / 75, 0, 1)
            return RGBA(1, 0, 0, alpha)
        end
    end

    mean_speedup = geomean(filter(>=(1e-6), ratios))
    mean_speedup_combine = geomean(max.(ratios, 100))

    bar!(p, idx, legend=false,
         xticks=(idx, labels), xrotation=90, xtickfont=font(5),
         ratios, err=(ratios .- ratios_lo, ratios_hi .- ratios),
         color=colors, ylims=(0, max(mean_speedup + 10, mean_speedup_combine + 10, 150)),
        #  series_annotations=text.(annotations, :top, 6, rotation = 90),
         # xxx: work around title getting cut off
         left_margin=1Plots.cm, bottom_margin=1Plots.cm)

    # # put the coverage percentage and relative performance in the bar
    # annotations = map(coverage, ratios) do cov_pct, perf_pct
    #     "   $(round(Int, cov_pct))% | $(round(Int, perf_pct))%"
    # end
    # annotate!(p, idx, 0, text.(annotations, 4, rotation=90, :left))

    # put the coverage percentage and relative performance in the bar
    annotations = map(coverage, ratios) do cov_pct, perf_pct
        "   $(round(Int, perf_pct))%"
    end
    annotate!(p, idx, 0, text.(annotations, 4, rotation=90, :left))

    # draw geometric mean of speedup factors
    text_margin = 0.25
    hline!([mean_speedup], color=:black, linestyle=:dash)
    annotate!(p, 9, mean_speedup + text_margin, text(L"\mathrm{GM}_1: %$(round(Int, mean_speedup))\%", 7, rotation=0, :left, :bottom, color=:black))

    hline!([mean_speedup_combine], color=:red, linestyle=:dash)
    annotate!(p, 9, mean_speedup_combine + text_margin, text(L"\mathrm{GM}_2: %$(round(Int, mean_speedup_combine))\%", 7, rotation=0, :left, :bottom, color=:red))

    plot!(size=(600, 325)) # default (600, 400)
    savefig(p, joinpath(@__DIR__, "$(name(device())).pdf"))
end
