using GemmKernels, CUDA
using DataFrames, Serialization

include("configs/configs.jl")

const data_type = Float16
const compute_type = Float16
const accumulate_type = Float32
const zero_c = true

function get_extents(tc)
    df = open("2024-03-28-best-configs-ripper.bin") do io
        deserialize(io)
    end

    first(filter(row -> row.name == tc, df)).extents
end

function get_optimal_params(tc, machine)
    file = Dict("ripper" => "2024-03-28-best-configs-ripper.bin",
                "jupiter" => "2024-03-05-best-configs-jupiter.bin")[machine]

    df = open(file) do io
        deserialize(io)
    end

    if in("name", names(df))
        row = first(filter(row -> row.name == tc, df))
    else
        row = first(filter(row -> row.parseable_name == tc, df))
    end

    has_layout_sweep = in("is_A_col_major", names(df))

    Dict(
        "BLOCK_M" => row.BLOCK_M,
        "BLOCK_N" => row.BLOCK_N,
        "BLOCK_K" => row.BLOCK_K,
        "WARPS_M" => row.WARPS_M,
        "WARPS_N" => row.WARPS_N,
        "OP_M" => row.OP_M,
        "OP_N" => row.OP_N,
        "OP_K" => row.OP_K,
        "kernel" => row.kernel_str == "pipelined" ? Kernel.matmul_pipelined : Kernel.matmul_singlestage,
        "is_A_col_major" => has_layout_sweep ? row.is_A_col_major : nothing,
        "is_B_col_major" => has_layout_sweep ? row.is_B_col_major : nothing,
        "is_D_col_major" => has_layout_sweep ? row.is_D_col_major : nothing,
        "PERM_M" => has_layout_sweep ? row.PERM_M : nothing,
        "PERM_N" => has_layout_sweep ? row.PERM_N : nothing,
        "PERM_K" => has_layout_sweep ? row.PERM_K : nothing)
end

function run_tc_gemmkernels(name, extents, optimal_params)
    problem = WMMATensorContraction(; name, extents, data_type, compute_type, accumulate_type, zero_c)
    data = allocate_data(problem)

    # tuned parameters

    BLOCK_M, BLOCK_N, BLOCK_K = (optimal_params["BLOCK_M"], optimal_params["BLOCK_N"], optimal_params["BLOCK_K"])
    WARPS_M, WARPS_N = (optimal_params["WARPS_M"], optimal_params["WARPS_N"])
    OP_M, OP_N, OP_K = (optimal_params["OP_M"], optimal_params["OP_N"], optimal_params["OP_K"])
    kernel = optimal_params["kernel"]
    PERM_M = optimal_params["PERM_M"]
    PERM_N = optimal_params["PERM_N"]
    PERM_K = optimal_params["PERM_K"]
    is_A_col_major = optimal_params["is_A_col_major"]
    is_B_col_major = optimal_params["is_B_col_major"]
    is_D_col_major = optimal_params["is_D_col_major"]

    params = (; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, kernel,
        is_A_col_major, is_B_col_major, is_D_col_major,
        PERM_M, PERM_N, PERM_K)

    args = prepare(problem, data...; params...)

    # main warm up
    initialize_data(problem, data...; params...)
    execute(problem, data...; args...)

    CUDA.@profile external=true begin
        for i = 1 : 10
            execute(problem, data...; args...)
        end
    end
end

function run_tc_cutensor(name, extents)
    problem = WMMATensorContraction(; name, extents, data_type, compute_type, accumulate_type, zero_c)
    data = allocate_data(problem)

    # reference warm up
    reference_args = prepare_baseline(problem, data...)
    execute_baseline(problem, data...; reference_args...)

    CUDA.@profile external=true begin
        for i = 1 : 10
            execute_baseline(problem, data...; reference_args...)
        end
    end
end

function main()
    @assert length(ARGS) == 2

    TC = ARGS[1]
    implementation = ARGS[2]

    if implementation == "cutensor"
        run_tc_cutensor(TC, get_extents(TC))
    elseif implementation == "gemmkernels-optimal-for-jupiter"
        run_tc_gemmkernels(TC, get_extents(TC), get_optimal_params(TC, "jupiter"))
    elseif implementation == "gemmkernels-optimal-for-ripper"
        run_tc_gemmkernels(TC, get_extents(TC), get_optimal_params(TC, "ripper"))
    else
        @assert false
    end
end

isinteractive() || main()
