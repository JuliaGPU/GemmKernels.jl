## constants

const N_vals = 2 .^ (7:14)

const AB_type = Float16
const CD_type = Float32

const zero_c = true

const MEMORY_USAGE = maximum(N_vals)^2 * 2 * sizeof(AB_type) +
                     maximum(N_vals)^2 * 2 * sizeof(CD_type)


## configs

include("../configs/configs.jl")

function generate_configs()
    all_configs = DataFrame(
        transpose_a=Bool[],
        transpose_b=Bool[],
        N=Int[],
        BLOCK_M=Int[],
        BLOCK_N=Int[],
        BLOCK_K=Int[],
        WARPS_M=Int[],
        WARPS_N=Int[],
        OP_M=Int[],
        OP_N=Int[],
        OP_K=Int[],
        kernel_str=String[]
    )

    for transpose_a in [false, true],
        transpose_b in [false, true],
        N in N_vals,
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

        push!(all_configs, Dict(
            :transpose_a => transpose_a,
            :transpose_b => transpose_b,
            :N => N,
            :BLOCK_M => BLOCK_M,
            :BLOCK_N => BLOCK_N,
            :BLOCK_K => BLOCK_K,
            :WARPS_M => WARPS_M,
            :WARPS_N => WARPS_N,
            :OP_M => OP_M,
            :OP_N => OP_N,
            :OP_K => OP_K,
            :kernel_str => kernel_str
        ))
    end

    all_configs
end

function repr_row(row)
    io = IOBuffer()

    # gemm shape
    print(io, "$(row.N)×$(row.N)")
    row.transpose_a && print(io, "'")
    print(io, "*$(row.N)×$(row.N)")
    row.transpose_b && print(io, "'")
    print(io, "=$(row.N)×$(row.N)")

    # details
    print(io, " ($(row.BLOCK_M)×$(row.BLOCK_N)×$(row.BLOCK_K) block")
    print(io, ", $(row.WARPS_M)×$(row.WARPS_N) warp")
    print(io, ", $(row.OP_M)×$(row.OP_N)×$(row.OP_K) operator")
    print(io, ", $(row.kernel_str) kernel)")

    return String(take!(io))
end

function get_config(row)
    transpose_a = row.transpose_a
    transpose_b = row.transpose_b
    M = N = K = row.N
    BLOCK_M = row.BLOCK_M
    BLOCK_N = row.BLOCK_N
    BLOCK_K = row.BLOCK_K
    WARPS_M = row.WARPS_M
    WARPS_N = row.WARPS_N
    OP_M = row.OP_M
    OP_N = row.OP_N
    OP_K = row.OP_K
    kernel = kernel_string_to_function(row.kernel_str)

    @get_wmma_config
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

function select_best(configs)
    best_configs = similar(configs, 0)

    for transpose_a = [false, true],
        transpose_b = [false, true],
        N = N_vals

        relevant_configs = configs[(@. (configs[!, "transpose_a"] == transpose_a) & (configs[!, "transpose_b"] == transpose_b) & (configs[!, "N"] == N)), :]
        _, best_config_index = findmin(relevant_configs[!, "time"])
        best_config = relevant_configs[best_config_index, :]

        push!(best_configs, Dict(
            :transpose_a => transpose_a,
            :transpose_b => transpose_b,
            :N => N,
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


## operations

# generate_inputs: directly from configs.jl

function initialize_inputs(cf, reference_mul!, a, b, c, d)
    rand!(a)
    rand!(b)
    rand!(c)
    d .= 0
end

function execute(cf, reference_mul!, a, b, c, d)
    run_gemm(cf, a, b, c, d)
    return d
end

function execute_baseline(cf, reference_mul!, a, b, c, d)
    run_baseline(cf, a, b, c, d)
    return c
end

function execute_reference(cf, reference_mul!, a, b, c, d)
    reference_mul!(c, a, b)
    return c
end

# verify: directly from configs.jl


## output

function plot_results(best_configs)
    markershapes = Dict(
        "NN" => :circle,
        "NT" => :dtriangle,
        "TN" => :diamond,
        "TT" => :cross
    )

    p = plot()
    title!("$AB_type x $AB_type = $CD_type ($(name(device())))")
    xlabel!("Matrix size [-]")
    ylabel!("Performance relative to cuBLAS [%]")

    for transpose_a in [false, true],
        transpose_b in [false, true]

        label = get_label(transpose_a, transpose_b)

        relevant_configs = best_configs[(@. (best_configs[!, "transpose_a"] == transpose_a) & (best_configs[!, "transpose_b"] == transpose_b)), :]

        ratios = @. 100 * perf_ratio(relevant_configs.gemmkernels_times, relevant_configs.baseline_times)
        ratios_lo = @. 100 * perf_ratio_lo(relevant_configs.gemmkernels_times, relevant_configs.baseline_times)
        ratios_hi = @. 100 * perf_ratio_hi(relevant_configs.gemmkernels_times, relevant_configs.baseline_times)

        plot!(p, relevant_configs.N, ratios, ribbon=(ratios .- ratios_lo, ratios_hi .- ratios), label=label, markershape=markershapes[label], xscale=:log2)
    end

    savefig(p, joinpath(@__DIR__, "$(name(device())).pdf"))
end

get_label(transpose_a, transpose_b) = "$(transpose_a ? "T" : "N")$(transpose_b ? "T" : "N")"
