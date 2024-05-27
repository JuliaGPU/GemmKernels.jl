## constants

using JSON
using Base.Iterators: product
using Combinatorics: permutations

const data_type = Float16
const compute_type = Float16
const accumulate_type = Float32

const zero_c = true

config_path = joinpath(@__DIR__, "../test/benchmark-suite.json")
fp = open(config_path, "r")
const jsonData = JSON.parse(read(fp, String))


## configurations

include("../configs/configs.jl")

function generate_problems()
    problems = []
    for el in jsonData
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

function config_iterator(problem)
    param_product = shuffle_product(
        2 .^ (6:9),
        2 .^ (6:9),
        2 .^ (5:7),
        2 .^ (0:3),
        2 .^ (0:3),
        [(16, 16, 16), (8, 32, 16), (32, 8, 16)],
        ["singlestage", "pipelined", "pipelined_ng"],

        [false, true],
        [false, true],
        [true], # only true for is_D_col_major because of illegal memory accesses
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

function select_configs(configs, problem)
    filter(configs) do config
        config.name == problem.name && config.extents == [problem.extents...]
    end
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
    title!("TTCG on $(name(device()))")
    xlabel!("Tensor contraction")
    ylabel!("Performance relative to cuTENSOR [%]")

    problems = generate_problems()
    idx = 1:length(problems)

    labels = []
    ratios = []
    ratios_lo = []
    ratios_hi = []
    coverage = []

    for problem in problems
        name_idx = findfirst(el -> el["parseableName"] == problem.name, jsonData)
        name_idx == nothing && error("Unknown parseable name: $(problem.name)")
        push!(labels, jsonData[name_idx]["name"])

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

    bar!(p, idx, legend=false,
         xticks=(idx, labels), xrotation=45, xtickfont=font(5),
         ratios, err=(ratios .- ratios_lo, ratios_hi .- ratios),
         color=colors, ylims=(0,150),
        #  series_annotations=text.(annotations, :top, 6, rotation = 90),
         # xxx: work around title getting cut off
         left_margin=1Plots.cm, bottom_margin=1.5Plots.cm)

    # put the coverage percentage in the bar
    annotations = map(coverage) do pct
        "   $(round(Int, pct))%"
    end
    annotate!(p, idx, 0, text.(annotations, 5, rotation=90, :left))

    savefig(p, joinpath(@__DIR__, "$(name(device())).pdf"))
end
