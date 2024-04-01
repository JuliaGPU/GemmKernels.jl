## constants

using JSON
using Combinatorics
using ProgressMeter

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

function count_configs(problem)
    modes = problem.modes
    length(6:9) *                                           # BLOCK_M
    length(6:9) *                                           # BLOCK_N
    length(5:7) *                                           # BLOCK_K
    length(0:3) *                                           # WARPS_M
    length(0:3) *                                           # WARPS_N
    3 *                                                     # (OP_M, OP_N, OP_K)
    2 *                                                     # kernel_str
    2 *                                                     # is_A_col_major
    2 *                                                     # is_B_col_major
    1 *                                                     # is_D_col_major
    length(permutations(intersect(modes[1], modes[2]))) *   # PERM_M
    length(permutations(intersect(modes[1], modes[3]))) *   # PERM_N
    length(permutations(intersect(modes[2], modes[3])))     # PERM_K
end

function generate_configs(problem)
    configs = DataFrame(
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
        PERM_K=Vector{Int}[]
    )

    for BLOCK_M in 2 .^ (6:9),
        BLOCK_N in 2 .^ (6:9),
        BLOCK_K in 2 .^ (5:7),
        WARPS_M in 2 .^ (0:3),
        WARPS_N in 2 .^ (0:3),
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        kernel_str in ["singlestage", "pipelined"],
        is_A_col_major in [false, true],
        is_B_col_major in [false, true],
        is_D_col_major in [#=false,=# true], # XXX: causes illegal memory accesses
        PERM_M in permutations(intersect(problem.modes[1], problem.modes[2])),
        PERM_N in permutations(intersect(problem.modes[1], problem.modes[3])),
        PERM_K in permutations(intersect(problem.modes[2], problem.modes[3]))

        push!(configs, (;
            :name => problem.name,
            :extents => [problem.extents...],

            :BLOCK_M => BLOCK_M,
            :BLOCK_N => BLOCK_N,
            :BLOCK_K => BLOCK_K,
            :WARPS_M => WARPS_M,
            :WARPS_N => WARPS_N,
            :OP_M => OP_M,
            :OP_N => OP_N,
            :OP_K => OP_K,
            :kernel_str => kernel_str,

            :is_A_col_major => is_A_col_major,
            :is_B_col_major => is_B_col_major,
            :is_D_col_major => is_D_col_major,
            :PERM_M => PERM_M,
            :PERM_N => PERM_N,
            :PERM_K => PERM_K
        ))
    end

    @assert nrow(configs) == count_configs(problem)
    configs
end

function select_configs(configs, problem)
    # use groupby to return a mutable handle
    for group in groupby(configs, [:name, :extents])
        config = first(group)
        if config.name == problem.name && config.extents == [problem.extents...]
            return group
        end
    end
    return nothing
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

    (; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, kernel,
       is_A_col_major, is_B_col_major, is_D_col_major, PERM_M, PERM_N, PERM_K)
end

function kernel_string_to_function(str)
    if str == "singlestage"
        return Kernel.matmul_singlestage
    elseif str == "pipelined"
        return Kernel.matmul_pipelined
    else
        error("Unknown kernel string: $str")
    end
end


## output

function plot_best_configs(df)
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

    idx = 1:nrow(df)
    problems = generate_problems()
    labels = map(eachrow(df)) do row
        name_idx = findfirst(el -> el["parseableName"] == row.name, jsonData)
        if name_idx == nothing
            error("Unknown parseable name: $(row.name)")
        end
        jsonData[name_idx]["name"]
    end
    ratios = @. 100 * perf_ratio(df.gemmkernels_times, df.baseline_times)
    ratios_lo = @. 100 * perf_ratio_lo(df.gemmkernels_times, df.baseline_times)
    ratios_hi = @. 100 * perf_ratio_hi(df.gemmkernels_times, df.baseline_times)

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
    annotations = map(eachrow(df)) do row
        "   $(round(Int, 100*row.coverage))%"
    end
    annotate!(p, idx, 0, text.(annotations, 5, rotation=90, :left))

    savefig(p, joinpath(@__DIR__, "$(name(device())).pdf"))
end
