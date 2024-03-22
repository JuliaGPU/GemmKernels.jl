using GemmKernels, CUDA, Random, NVTX, Combinatorics, ProgressMeter, DataFrames, Serialization

include("configs/configs.jl")

const data_type = Float16
const compute_type = Float16
const accumulate_type = Float32

const zero_c = true

const NUM_SAMPLES = 10

struct SweepConfig
    is_A_col_major::Bool
    is_B_col_major::Bool
    is_D_col_major::Bool
    perm_M::Vector{Int}
    perm_N::Vector{Int}
    perm_K::Vector{Int}
end

function get_time_for_profile(prof, nvtx_name)
    starts = filter(row -> row.type == :start && row.name == nvtx_name, prof.nvtx)
    ends = filter(row -> row.type == :end, prof.nvtx)

    joined = rename(innerjoin(starts, ends, on=:id, makeunique=true), :start_1 => :end)

    @assert nrow(joined) == NUM_SAMPLES

    times = joined[!, :end] - joined[!, :start]

    minimum(times)
end

function sweep_tc_config(row)
    sweep_configs = []

    # TODO: Why does is_D_col_major == false result in incorrect results?

    for is_A_col_major = [false, true],
        is_B_col_major = [false, true],
        is_D_col_major = [false, true],
        perm_M = permutations([1, 2, 3, 5]),
        perm_N = permutations([4]),
        perm_K = permutations([6])

        push!(sweep_configs, SweepConfig(is_A_col_major, is_B_col_major, is_D_col_major, perm_M, perm_N, perm_K))
    end

    best_config, best_time = nothing, nothing
    num_incorrect = 0
    incorrect_configs = []
    baseline_time = nothing

    @showprogress dt=1 for sweep_config in sweep_configs
        parseable_name = row.parseable_name
        extents = row.extents

        BLOCK_M, BLOCK_N, BLOCK_K = (64, 64, 64)
        WARPS_M, WARPS_N = (4, 4)
        OP_M, OP_N, OP_K = (16, 16, 16)
        kernel = Kernel.matmul_singlestage

        GemmKernels.Tensors.OVERRIDE_do_override = true
        GemmKernels.Tensors.OVERRIDE_is_A_col_major = sweep_config.is_A_col_major
        GemmKernels.Tensors.OVERRIDE_is_B_col_major = sweep_config.is_B_col_major
        GemmKernels.Tensors.OVERRIDE_is_D_col_major = sweep_config.is_D_col_major
        GemmKernels.Tensors.OVERRIDE_perm_M = sweep_config.perm_M
        GemmKernels.Tensors.OVERRIDE_perm_N = sweep_config.perm_N
        GemmKernels.Tensors.OVERRIDE_perm_K = sweep_config.perm_K

        cf = @get_tc_wmma_config

        run_reference!, a, b, c, d = generate_inputs(cf)
        rand!(a)
        rand!(b)
        rand!(c)
        c .= 0

        d .= 0
        run_tc(cf, a, b, c, d)
        d_ours = Array(d)

        run_reference!(c, a, b)
        d_theirs = Array(c)

        if !all(isapprox.(d_ours, d_theirs))
            @warn "Results do not match"
            num_incorrect += 1
            push!(incorrect_configs, sweep_config)
            continue
        end

        prof = CUDA.@profile begin
            for i = 1 : NUM_SAMPLES
                NVTX.@range "GemmKernels" CUDA.@sync run_tc(cf, a, b, c, d)
            end
        end

        time = get_time_for_profile(prof, "GemmKernels")

        if isnothing(baseline_time)
            baseline_prof = CUDA.@profile begin
                for i = 1 : NUM_SAMPLES
                    NVTX.@range "cuTENSOR" CUDA.@sync run_baseline(cf, a, b, c, d)
                end
            end

            baseline_time = get_time_for_profile(baseline_prof, "cuTENSOR")
        end


        if (best_time === nothing) || (time < best_time)
            best_time = time
            best_config = sweep_config
        end
    end

    @show best_time
    @show baseline_time
    @show round(100 * baseline_time / best_time; digits=2)
    @show best_config
    @show num_incorrect

    incorrect_configs
end

function main()
    df = open("tuning/best-configs.bin") do io
        deserialize(io)
    end

    sweep_tc_config(df[7, :])
end
