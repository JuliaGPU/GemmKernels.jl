using Arrow
using CUDA, GemmKernels
using Serialization
using DataFrames
using Printf
using Statistics

const BENCHMARK_SAMPLES = 5

isinteractive() || include("wmma-contraction.jl")

function wait_if_throttling(dev=NVML.Device(parent_uuid(device())))
    # make sure we're reading accurate data
    # (for when this function is called in a loop)
    sleep(0.01)

    cer = NVML.clock_event_reasons(dev)

    while cer.hw_power_brake || cer.sw_power_cap || cer.hw_slow || cer.sw_thermal || cer.hw_thermal
        sleep(0.1)
        cer = NVML.clock_event_reasons(dev)
    end
end

function benchmark_cutensor(problem; elementwise_op)
    CUDA.reclaim()

    times = []

    data = allocate_data(problem)
    args = prepare_baseline(problem, data...)
    execute_baseline(problem, data...; args..., elementwise_op)
    wait_if_throttling()

    for i in 1:BENCHMARK_SAMPLES
        prof = CUDA.@profile concurrent=false execute_baseline(problem, data...; args..., elementwise_op)
        cur_time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
        push!(times, cur_time)
    end

    return times
end

function benchmark_gemmkernels(problem, config; elementwise_op)
    CUDA.reclaim()

    times = []

    data = allocate_data(problem)
    params = create_params(config)
    args = prepare(problem, data...; params..., elementwise_op)
    execute(problem, data...; args...)
    wait_if_throttling()

    for i in 1:BENCHMARK_SAMPLES
        prof = CUDA.@profile concurrent=false execute(problem, data...; args...)
        cur_time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
        push!(times, cur_time)
    end

    return times
end

function generate_data()
    # Load best configurations.
    best_configs_path = joinpath(@__DIR__, "best-configs.bin")
    @assert isfile(best_configs_path)

    best_configs = deserialize(best_configs_path)

    # Benchmark 6 configurions:

    # 1) base (no elementwise operation)
    #   1.1) cuTENSOR
    #   1.2) GemmKernels
    #
    # 2) supported elementwise operation
    #   2.1) cuTENSOR (fused)
    #   2.2) GemmKernels (fused)
    #
    # 3) non-supported elementwise operations
    #   3.1) cuTENSOR (not fused)
    #   3.2) GemmKernels (fused)

    data = DataFrame(
        name = String[],

        cuTENSOR_base_times = Vector{Float64}[],
        GemmKernels_base_times = Vector{Float64}[],

        cuTENSOR_supported_times = Vector{Float64}[],
        GemmKernels_supported_times = Vector{Float64}[],

        cuTENSOR_unsupported_times = Vector{Float64}[],
        GemmKernels_unsupported_times = Vector{Float64}[],
    )

    problems = generate_problems()

    for (i, problem) in enumerate(problems)
        name_idx = findfirst(el -> el["parseableName"] == problem.name, jsonData)
        name_idx == nothing && error("Unknown parseable name: $(problem.name)")
        name = jsonData[name_idx]["name"]

        @info "Running TC $i/$(length(problems)): $name"

        configs = select_configs(best_configs, problem)
        @assert size(configs, 1) == 1
        config = first(configs)

        # (1.1)
        cut_base = benchmark_cutensor(problem; elementwise_op = "none")

        # (1.2)
        gk_base = benchmark_gemmkernels(problem, config; elementwise_op = "none")

        # (2.1)
        cut_supported = benchmark_cutensor(problem; elementwise_op = "supported")

        # (2.2)
        gk_supported = benchmark_gemmkernels(problem, config; elementwise_op = "supported")

        # (3.1)
        cut_unsupported = benchmark_cutensor(problem; elementwise_op = "unsupported")

        # (3.2)
        gk_unsupported = benchmark_gemmkernels(problem, config; elementwise_op = "unsupported")

        pretty_percent(result, base) = @sprintf "%+.2f%%" 100*(minimum(result)/minimum(base)-1)

        @info "Compared to best-configs cuTENSOR:    $(pretty_percent(cut_base, config["baseline_times"]))"
        @info "Compared to best-configs GemmKernels: $(pretty_percent(gk_base, config["gemmkernels_times"]))"

        push!(data, Dict(
            :name => "$name",
            :cuTENSOR_base_times => cut_base,
            :GemmKernels_base_times => gk_base,
            :cuTENSOR_supported_times => cut_supported,
            :GemmKernels_supported_times => gk_supported,
            :cuTENSOR_unsupported_times => cut_unsupported,
            :GemmKernels_unsupported_times => gk_unsupported,
        ))
    end

    data
end

function pretty_print_data(data)
    open("operator-fusion.tex", "w") do io
        for (i, row) in enumerate(eachrow(data))
            name = row["name"]
            cut_base = row["cuTENSOR_base_times"]
            gk_base = row["GemmKernels_base_times"]
            cut_supported = row["cuTENSOR_supported_times"]
            gk_supported = row["GemmKernels_supported_times"]
            cut_unsupported = row["cuTENSOR_unsupported_times"]
            gk_unsupported = row["GemmKernels_unsupported_times"]

            res(actual) = @sprintf "%.2f" 1e6*minimum(actual)
            rat(actual, base) = @sprintf "%.2f (%+3.1f%%)" 1e6*minimum(actual) 100*(minimum(actual)/minimum(base)-1)

            println(io, "$i & $name & $(res(cut_base)) & $(res(gk_base)) & $(rat(cut_supported, cut_base)) & $(rat(gk_supported, gk_base)) & $(rat(cut_unsupported, cut_base)) & $(rat(gk_unsupported, gk_base)) \\\\")
        end
    end
end

function main()
    data_path = joinpath(@__DIR__, "operator-fusion.bin")

    if !isfile(data_path)
        data = generate_data()
        serialize(data_path, data)
    else
        @info "Loading previous results from disk."
    end

    data = deserialize(data_path)

    pretty_print_data(data)
end

if !isinteractive()
    main()
end
