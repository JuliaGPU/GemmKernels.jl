## constants

using JSON

const data_type = Float16
const compute_type = Float16
const accumulate_type = Float32

const zero_c = true

config_path = joinpath(@__DIR__, "../test/benchmark-suite.json")
fp = open(config_path, "r")
const jsonData = JSON.parse(read(fp, String))


## configurations

include("../configs/configs.jl")

function generate_configs()
    configs = DataFrame(
        # problem
        name=String[],
        extents=Vector{Int}[],

        # params
        BLOCK_M=Int[],
        BLOCK_N=Int[],
        BLOCK_K=Int[],
        WARPS_M=Int[],
        WARPS_N=Int[],
        OP_M=Int[],
        OP_N=Int[],
        OP_K=Int[],
        kernel_str=String[],
    )

    for (name, extents) in [(el["parseableName"], el["extents"]) for el in jsonData],
        BLOCK_M in 2 .^ (6:9),
        BLOCK_N in 2 .^ (6:9),
        BLOCK_K in 2 .^ (5:7),
        WARPS_M in 2 .^ (0:3),
        WARPS_N in 2 .^ (0:3),
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        kernel_str in ["singlestage", "pipelined"]

        push!(configs, Dict(
            :name => name,
            :extents => extents,
            :BLOCK_M => BLOCK_M,
            :BLOCK_N => BLOCK_N,
            :BLOCK_K => BLOCK_K,
            :WARPS_M => WARPS_M,
            :WARPS_N => WARPS_N,
            :OP_M => OP_M,
            :OP_N => OP_N,
            :OP_K => OP_K,
            :kernel_str => kernel_str,
        ))
    end

    configs
end

function repr_row(row)
    io = IOBuffer()

    # tccg benchmark parseable name
    print(io, "$(row.name)")

    # details
    print(io, " ($(row.BLOCK_M)×$(row.BLOCK_N)×$(row.BLOCK_K) block")
    print(io, ", $(row.WARPS_M)×$(row.WARPS_N) warp")
    print(io, ", $(row.OP_M)×$(row.OP_N)×$(row.OP_K) operator")
    print(io, ", $(row.kernel_str) kernel)")

    return String(take!(io))
end

function get_problem(row)
    name = row.name
    extents = row.extents

    WMMATensorContraction(; name, extents, data_type, compute_type, accumulate_type, zero_c)
end

function get_params(row)
    BLOCK_M = row.BLOCK_M
    BLOCK_N = row.BLOCK_N
    BLOCK_K = row.BLOCK_K
    WARPS_M = row.WARPS_M
    WARPS_N = row.WARPS_N
    OP_M = row.OP_M
    OP_N = row.OP_N
    OP_K = row.OP_K
    kernel = kernel_string_to_function(row.kernel_str)

    (; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, kernel)
end

group_configs(configs) = groupby(configs, [:name, :extents])

function kernel_string_to_function(str)
    if str == "singlestage"
        return Kernel.matmul_singlestage
    elseif str == "pipelined"
        return Kernel.matmul_pipelined
    else
        error("Unknown kernel string: $str")
    end
end

function select_best(configs)
    best_configs = similar(configs, 0)

    for (parseable_name, extents) in [(el["parseableName"], el["extents"]) for el in jsonData]
        relevant_configs = configs[(@. (configs[!, "parseable_name"] == parseable_name)), :]

        _, best_config_index = findmin(relevant_configs[!, "time"])
        best_config = relevant_configs[best_config_index, :]

        push!(best_configs, Dict(
            :parseable_name => parseable_name,
            :extents => extents,
            :BLOCK_M => best_config["BLOCK_M"],
            :BLOCK_N => best_config["BLOCK_N"],
            :BLOCK_K => best_config["BLOCK_K"],
            :WARPS_M => best_config["WARPS_M"],
            :WARPS_N => best_config["WARPS_N"],
            :OP_M => best_config["OP_M"],
            :OP_N => best_config["OP_N"],
            :OP_K => best_config["OP_K"],
            :kernel_str => best_config["kernel_str"],
            :time => best_config["time"],
        ))
    end

    return best_configs
end


## output

function plot_results(df)
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
    names = df.parseable_name
    names = map(df.parseable_name) do parseable_name
        for el in jsonData
            if el["parseableName"] == parseable_name
                return el["name"]
            end
        end
        error("Unknown parseable name: $parseable_name")
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
         xticks=(idx, names), xrotation=45, xtickfont=font(5),
         ratios, err=(ratios .- ratios_lo, ratios_hi .- ratios),
         color=colors, ylims=(0,150))

    savefig(p, joinpath(@__DIR__, "$(name(device())).pdf"))
end

get_label(transpose_a, transpose_b) = "$(transpose_a ? "T" : "N")$(transpose_b ? "T" : "N")"
