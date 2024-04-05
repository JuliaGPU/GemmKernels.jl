using GemmKernels, CUDA

include("configs/configs.jl")

const data_type = Float16
const compute_type = Float16
const accumulate_type = Float32
const zero_c = true

const EXTENTS = Dict(
    "1.2-1.3-3.2" => [5136, 5120, 5136], # TC 12
    "1.2.3.4-5.3-1.2.5.4" => [72, 72, 72, 72, 72], # TC 11
)

# key: (tensor_contraction, machine).
const OPTIMAL_PARAMS = Dict(
    # Optimal parameters of TC 12 from sweep 2024-03-28 on ripper.
    ("1.2-1.3-3.2", "ripper") => Dict(
        "BLOCK_M"        => 64,
        "BLOCK_N"        => 256,
        "BLOCK_K"        => 32,
        "WARPS_M"        => 2,
        "WARPS_N"        => 4,
        "OP_M"           => 32,
        "OP_N"           => 8,
        "OP_K"           => 16,
        "kernel"         => Kernel.matmul_pipelined,
        "is_A_col_major" => false,
        "is_B_col_major" => true,
        "is_D_col_major" => true,
        "PERM_M"         => [1],
        "PERM_N"         => [2],
        "PERM_K"         => [3],
    ),

    # Optimal parameters of TC 12 from sweep 2024-03-05 on jupiter.
    ("1.2-1.3-3.2", "jupiter") => Dict(
        "BLOCK_M"        => 64,
        "BLOCK_N"        => 64,
        "BLOCK_K"        => 32,
        "WARPS_M"        => 1,
        "WARPS_N"        => 2,
        "OP_M"           => 8,
        "OP_N"           => 32,
        "OP_K"           => 16,
        "kernel"         => Kernel.matmul_pipelined,
        "is_A_col_major" => nothing,
        "is_B_col_major" => nothing,
        "is_D_col_major" => nothing,
        "PERM_M"         => nothing,
        "PERM_N"         => nothing,
        "PERM_K"         => nothing,
    ),

    # Optimal parameters of TC 11 from sweep 2024-03-28 on ripper.
    ("1.2.3.4-5.3-1.2.5.4", "ripper") => Dict(
        "BLOCK_M"        => 64,
        "BLOCK_N"        => 64,
        "BLOCK_K"        => 64,
        "WARPS_M"        => 8,
        "WARPS_N"        => 1,
        "OP_M"           => 8,
        "OP_N"           => 32,
        "OP_K"           => 16,
        "kernel"         => Kernel.matmul_pipelined,
        "is_A_col_major" => false,
        "is_B_col_major" => false,
        "is_D_col_major" => true,
        "PERM_M"         => [3],
        "PERM_N"         => [1, 2, 4],
        "PERM_K"         => [5],
    ),

    ("1.2.3.4-5.3-1.2.5.4", "jupiter") => Dict(
        "BLOCK_M"        => 64,
        "BLOCK_N"        => 64,
        "BLOCK_K"        => 64,
        "WARPS_M"        => 4,
        "WARPS_N"        => 2,
        "OP_M"           => 16,
        "OP_N"           => 16,
        "OP_K"           => 16,
        "kernel"         => Kernel.matmul_pipelined,
        "is_A_col_major" => nothing,
        "is_B_col_major" => nothing,
        "is_D_col_major" => nothing,
        "PERM_M"         => nothing,
        "PERM_N"         => nothing,
        "PERM_K"         => nothing,
    )
)

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

    @assert TC ∈ keys(EXTENTS)

    if implementation == "cutensor"
        run_tc_cutensor(TC, EXTENTS[TC])
    elseif implementation == "gemmkernels-optimal-for-jupiter"
        run_tc_gemmkernels(TC, EXTENTS[TC], OPTIMAL_PARAMS[(TC, "jupiter")])
    elseif implementation == "gemmkernels-optimal-for-ripper"
        run_tc_gemmkernels(TC, EXTENTS[TC], OPTIMAL_PARAMS[(TC, "ripper")])
    else
        @assert false
    end
end

isinteractive() || main()
