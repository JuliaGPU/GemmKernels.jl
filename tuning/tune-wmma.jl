using CUDA, GemmKernels
using DataFrames
using DataStructures
using Dates
using Distributed
using FileWatching.Pidfile
using Logging
using LoggingExtras
using Octavian
using ProgressMeter
using Serialization
using Statistics
using StatsBase
using Random

# if myid() == 1
#     using Plots
#     pythonplot()
# end

#######

const N_vals = 2 .^ (7:14)

# Stop sampling when normalised 95p CI is smaller than this...
const BENCH_NORM_CI_THRESHOLD = 0.01

# ... or we have exceeded the time limit...
const BENCH_MAX_NUM_SECONDS = 5

# ... but have at least 10 samples.
const BENCH_MIN_NUM_SAMPLES = 10

const BENCH_MEMORY_USAGE = 5*2^30

#####

# Stop gathering samples for plot if we have spent this much time...
# 60 seconds/configuration * 32 configurations = 32 minutes for plot.
const PLOT_NUM_SECONDS = 60

# ... but have at least 100 samples.
const PLOT_MIN_NUM_SAMPLES = 100

# Group samples in batches of 10 samples each.
const PLOT_BATCH_SIZE = 10

const AB_type = Float16
const CD_type = Float32

const zero_c = true

#######

include("../configs/configs.jl")

# Write logging messages to file for persistence.
timestamp_logger(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) $(log.message)"))
end
function log_filename()
    path = joinpath(@__DIR__, "tuning.log")
    if myid() != 1
        path = "$(path).$(myid())"
    end
    path
end
FileLogger(log_filename(); append=true) |> timestamp_logger |> (x -> MinLevelLogger(x, Logging.Info)) |> global_logger

function kernel_string_to_function(str)
    Dict(
        "singlestage" => Kernel.matmul_singlestage,
        "pipelined" => Kernel.matmul_pipelined
       )[str]
end

get_label(transpose_a, transpose_b) = "$(transpose_a ? "T" : "N")$(transpose_b ? "T" : "N")"

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
        kernel_str=String[],
        category=String[],
        times=Vector{Any}[]
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
            :kernel_str => kernel_str,
            :category => "pending",
            :times => [],
        ))
    end

    all_configs
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

function get_inputs_for_plot(input_dict, row)
    if row.N ∉ keys(input_dict)
        cf = get_config(row)
        _, a, b, c, d = generate_inputs(cf)
        input_dict[row.N] = (a, b, c, d)
    end

    return input_dict[row.N]
end

function measure_config(row)
    @info "Measuring configuration $(repr_row(row))..."
    cf = get_config(row)

    pidfile = joinpath(@__DIR__, "tuning.pid")

    get_ref, a, b, c = generate_inputs(cf)
    d = similar(c)
    d .= 0

    try
        run_gemm(cf, a, b, c, d)
    catch err
        bt = catch_backtrace()
        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

        if isa(err, GemmKernels.ConfigError)
            @info "Skipping configuration $(repr_row(row))\n" * log
            return [Inf], "unsupported_config_post_run"
        end

        if isa(err, CuError)
            @error "Configuration failed: $(repr_row(row))\n" * log
            rethrow()
        end

        @info "Skipping configuration: $(repr_row(row))\n" * log
        return [Inf], "error"
    end

    c_ref = mkpidlock(pidfile) do
        get_ref(a, b, c)
    end
    if !verify(cf, c_ref, d)
        @warn "Configuration produced invalid result: $(repr_row(row))"

        return [Inf], "invalid_result"
    end

    times = Float64[]

    # Use CUDA.@elapsed instead of CUDA.@profile, because the latter is slower.
    device_synchronize()
    GC.gc(true)

    mkpidlock(pidfile) do
        start_time = Dates.now()

        while true
            synchronize(stream())
            time = CUDA.@elapsed run_gemm(cf, a, b, c, d)
            push!(times, time)

            if length(times) >= BENCH_MIN_NUM_SAMPLES
                (Dates.now() - start_time > Second(BENCH_MAX_NUM_SECONDS)) && break
                (confidence_interval_95(times) / median(times) < BENCH_NORM_CI_THRESHOLD) && break
            end
        end
    end

    return times, "success"
end

confidence_interval_95(times) = 1.58 * iqr(times) / sqrt(length(times))

function prettytime(times)
    times == [Inf] && return "no samples"

    min, q1, med, q3, max = nquantile(times, 4)
    ci_95 = confidence_interval_95(times)

    # timescale
    scale, unit = if med < 1e3
        1, "ns"
    elseif med < 1e6
        1e3, "μs"
    elseif med < 1e9
        1e6, "ms"
    else
        1e9, "s"
    end

    rnd_min, rnd_q1, rnd_med, rnd_q3, rnd_max, rnd_ci_95 = round.([min, q1, med, q3, max, ci_95] ./ scale; sigdigits=3)
    rnd_rel_ci_95 = round(100 * ci_95 / med; sigdigits=3)

    return "$rnd_med $unit ± $rnd_ci_95 $unit ($rnd_rel_ci_95%) (length: $(length(times)), 5-num summary: $rnd_min, $rnd_q1, $rnd_med, $rnd_q3, $rnd_max $unit)"
end

perf_ratio(gemmkernels, baseline) = percentile(baseline, 0) / percentile(gemmkernels, 0)
perf_ratio_lo(gemmkernels, baseline) = percentile(baseline, 0) / percentile(gemmkernels, 75)
perf_ratio_hi(gemmkernels, baseline) = percentile(baseline, 75) / percentile(gemmkernels, 0)

function get_uncertainty(gk, bl)
    lo, mid, hi = (perf_ratio_lo(gk, bl), perf_ratio(gk, bl), perf_ratio_hi(gk, bl))

    hi_uncertainty = abs(hi - mid) / mid
    lo_uncertainty = abs(lo - mid) / mid
    uncertainty = max(hi_uncertainty, lo_uncertainty)

    uncertainty, lo_uncertainty, hi_uncertainty
end

function got_enough_samples(row)
    gk, bl = row["gemmkernels_times"], row["baseline_times"]

    (length(gk) < PLOT_MIN_NUM_SAMPLES) && return false
    (length(bl) < PLOT_MIN_NUM_SAMPLES) && return false

    row["time_spent"] >= PLOT_NUM_SECONDS
end

function get_nvml_data(dev)
    Dict(
         :clock_info => NVML.clock_info(dev),
         :max_clock_info => NVML.max_clock_info(dev),
         :clock_event_reasons => NVML.clock_event_reasons(dev),
         :power_usage => NVML.power_usage(dev),
         :energy_consumption => NVML.energy_consumption(dev),
         :temperature => NVML.temperature(dev),
         :memory_info => NVML.memory_info(dev),
         :utilization_rates => NVML.utilization_rates(dev),
    )
end

function wait_if_throttling(dev)
    cer = NVML.clock_event_reasons(dev)

    while cer.hw_power_brake || cer.sw_power_cap || cer.hw_slow || cer.sw_thermal || cer.hw_thermal
        @info "Throttling detected. Sleeping for one second..."
        sleep(1)
    end
end

function benchmark_best_configs(configs)
    best_configs = DataFrame(
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
        kernel_str=String[],
        category=String[],
        time_spent=Float64[],
        gemmkernels_times=Vector{Any}[],
        baseline_times=Vector{Any}[],
        gemmkernels_nvml=Vector{Any}[],
        baseline_nvml=Vector{Any}[]
    )

    dev = NVML.Device(parent_uuid(device()))

    for transpose_a = [false, true],
        transpose_b = [false, true],
        N = N_vals

        relevant_configs = configs[(@. (configs[!, "transpose_a"] == transpose_a) & (configs[!, "transpose_b"] == transpose_b) & (configs[!, "N"] == N)), :]
        _, best_config_index = findmin(minimum.(relevant_configs[!, "times"], init=Inf))
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
            :category => "todo",
            :time_spent => 0.0,
            :gemmkernels_times => [],
            :baseline_times => [],
            :gemmkernels_nvml => [],
            :baseline_nvml => [],
        ))
    end

    # We will reuse matrix inputs across iterations. This takes about 4 GB of GPU memory for e.g. all matrix sizes for NN.
    # Group runs of the same transposition together, so we don't have to keep 4 * 4 GB of inputs in memory.
    for transpose_a in [false, true],
        transpose_b in [false, true]

        input_dict = Dict()

        p = ProgressUnknown(desc="Benchmarking", dt=1.0)

        # Spread the samples of one configuration over time, to reduce the effect
        # of time-related noise. Note that this means that the progress bar may
        # make big jumps.
        while true
            (sum(@. (best_configs[!, "category"] == "todo") & (best_configs[!, "transpose_a"] == transpose_a) & (best_configs[!, "transpose_b"] == transpose_b)) == 0) && break

            for config_row in eachrow(best_configs)
                if (config_row.category, config_row.transpose_a, config_row.transpose_b) != ("todo", transpose_a, transpose_b)
                    continue
                end

                a, b, c, d = get_inputs_for_plot(input_dict, config_row)
                cf = get_config(config_row)

                @info "Profiling configuration $(repr_row(config_row))..."

                for run_baseline in [false, true]
                    for i in 1:PLOT_BATCH_SIZE
                        wait_if_throttling()

                        start_time = Dates.now()

                        push!(config_row[if run_baseline "baseline_nvml" else "gemmkernels_nvml" end], get_nvml_data(dev))

                        if run_baseline
                            prof = CUDA.@profile concurrent=false run_baseline(cf, a, b, c, d)
                        else
                            prof = CUDA.@profile concurrent=false run_gemm(cf, a, b, c, d)
                        end

                        push!(config_row[if run_baseline "baseline_times" else "gemmkernels_times" end], sum(prof.device[!, "stop"] - prof.device[!, "start"]))

                        config_row["time_spent"] += (Dates.now() - start_time) / Second(1)
                    end
                end

                if got_enough_samples(config_row)
                    config_row["category"] = "done"
                end

                # Update progress bar.
                next!(p; showvalues = [
                    (:transpose_a, transpose_a),
                    (:transpose_b, transpose_b),
                    (:N, config_row["N"]),
                    (:num_samples, length(config_row["gemmkernels_times"])),
                    (:time_spent_in_config, config_row["time_spent"]),
                    (:remaining_N, best_configs[(@. (best_configs[!, "category"] == "todo") & (best_configs[!, "transpose_a"] == transpose_a) & (best_configs[!, "transpose_b"] == transpose_b)), :].N),
                    (:remaining_configurations, sum(best_configs[!, "category"] .== "todo"))
                ])
            end
        end
    end

    best_configs
end

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

function addworkers(X)
    env = [
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => string(BENCH_MEMORY_USAGE),
    ]
    exeflags = [
        "--project=$(Base.active_project())",
        "--heap-size-hint=$BENCH_MEMORY_USAGE"
    ]

    procs = addprocs(X; exeflags, env)
    @everywhere procs include($(joinpath(@__DIR__, "tune-wmma.jl")))
    procs
end

function main()
    # (0) Load configurations from disk, or generate them.
    config_path = joinpath(@__DIR__, "configs.bin")
    configs = nothing
    if isfile(config_path)
        @info "Loading configurations from disk..."
        try
            configs = open(config_path, "r") do io
                deserialize(io)
            end
            @info "Loaded $(size(configs, 1)) configurations."
        catch err
            @error "Error while loading configurations from disk: $(sprint(Base.showerror, err)))"
            mv(config_path, "$(config_path).broken")
        end
    end
    if configs === nothing
        # (1) Generate configurations.
        @info "Generating configurations..."
        configs = generate_configs()
        @info "Generated $(size(configs, 1)) configurations."

        # (2) Filter configurations where we can determine upfront that they are unsupported.
        @info "Filtering configurations that we know are unsupported a-priori..."

        for config_row in eachrow(configs)
            try
                cf = get_config(config_row)
            catch err
                if isa(err, GemmKernels.ConfigError)
                    config_row["category"] = "unsupported_config_pre_run"
                else
                    rethrow()
                end
            end
        end

        @info "Filtered $(counter(configs[!, "category"])["unsupported_config_pre_run"]) configurations."

        open(config_path, "w") do io
            serialize(io, configs)
        end
    end

    # (3) Measure performance of configurations.
    num_pending = counter(configs[!, "category"])["pending"]
    p = Progress(num_pending; desc="Parameter sweep", dt=1.0, showspeed=true)

    @info "Need to perform parameter sweep over $(num_pending) configurations."

    pending = filter(1:size(configs, 1)) do i
        configs[i, :category] == "pending"
    end
    shuffle!(pending)
    results = Channel(Inf)
    @sync begin
        # measure configurations on workers
        for p in workers()
            errormonitor(@async begin
                while length(pending) > 0
                    i = popfirst!(pending)
                    config_row = configs[i, :]
                    try
                        start_time = Dates.now()
                        config_row.times, config_row.category =
                            remotecall_fetch(measure_config, p, NamedTuple(config_row))
                        end_time = Dates.now()

                        push!(results, (p, i, start_time, end_time))
                    catch err
                        config_row.category = "crashed"

                        bt = catch_backtrace()
                        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                        @error "Error while measuring configurations: $log"

                        # recycle our worker
                        rmprocs(p; waitfor=30)
                        p = addworkers(1)[1]
                    end
                end
            end)
        end

        # process the results
        errormonitor(@async begin
            while !isempty(pending) || !isempty(results)
                worker, i, start_time, end_time = take!(results)

                # Update configuration
                config_row = configs[i, :]
                @info "Result from worker $worker for $(repr_row(config_row)): $(config_row.category) -- $(prettytime(config_row.times .* 1e9))"

                # Save results in case the process crashes.
                open(config_path, "w") do io
                    serialize(io, configs)
                end

                # Update progress bar
                counter_dict_abs = Dict(counter(configs[!, "category"]))
                counter_dict_rel = Dict(k => "$(round(100 * v / sum(values(counter_dict_abs)); sigdigits=3))%" for (k, v) in counter_dict_abs)
                next!(p; showvalues=[
                    (:N, config_row.N),
                    (:transpose, get_label(config_row.transpose_a, config_row.transpose_b)),
                    (:block_shape, (config_row.BLOCK_M, config_row.BLOCK_N, config_row.BLOCK_K)),
                    (:num_warps, (config_row.WARPS_M, config_row.WARPS_N)),
                    (:op_shape, (config_row.OP_M, config_row.OP_N, config_row.OP_K)),
                    (:kernel, config_row.kernel_str),
                    (:counters, counter_dict_abs),
                    (:counters_relative, counter_dict_rel),
                    (:last_result, "$(config_row.category) -- $(prettytime(config_row.times .* 1e9))"),
                    (:last_iteration_time, end_time - start_time)
                ])
            end
        end)
    end

    # Save data for final iteration.
    open(config_path, "w") do io
        serialize(io, configs)
    end

    # And load again, for good measure.
    configs = open(config_path, "r") do io
        deserialize(io)
    end

    # (4) Select best configurations, and benchmark.
    best_configs_path = joinpath(@__DIR__, "best-configs.bin")
    best_configs = nothing
    if isfile(best_configs_path)
        try
            @info "Loading best configurations from disk..."
            best_configs = open(best_configs_path, "r") do io
                deserialize(io)
            end
        catch err
            @error "Error while loading best configurations from disk: $(sprint(Base.showerror, err)))"
            mv(best_configs_path, "$(best_configs_path).broken")
        end
    end
    if best_configs === nothing
        @info "Benchmarking configurations for plot..."
        best_configs = benchmark_best_configs(configs)

        open(best_configs_path, "w") do io
            serialize(io, best_configs)
        end
    end


    # (5) Plotting results
    @info "Plotting results..."
    plot_results(best_configs)
end

if !isinteractive() && myid() == 1
    # Spawn workers
    cpu_memory = Sys.free_memory()
    gpu_memory = CUDA.available_memory()
    let
        addworkers(min(
            floor(Int, cpu_memory / BENCH_MEMORY_USAGE),
            floor(Int, gpu_memory / BENCH_MEMORY_USAGE),
            Sys.CPU_THREADS
        ))
    end
    @info "Starting WMMA tuning script for device $(name(device())) using $(nworkers()) workers..."

    main()
end
