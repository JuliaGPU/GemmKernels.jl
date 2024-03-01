using CUDA, GemmKernels
using DataFrames
using DataStructures: counter
using Dates
using Distributed
using FileWatching.Pidfile
using Logging
using LoggingExtras
using ProgressMeter: Progress, ProgressUnknown, next!, update!, finish!, @showprogress
using Serialization
using Statistics
using StatsBase: percentile
using Random

if myid() == 1
    using Plots
end

# interface for tuning modules:
#
# configurations
# - generate_configs(): return Dataframe with possible configurations
# - repr_row(row): pretty-print a row of the Dataframe
# - select_best(configs): given a Dataframe of configurations, with an added :time column,
#   return a Dataframe with the best configuration for each combination of parameters
# - get_config(row): create a Configuration object `cf` from a Dataframe row
#
# operations
# - generate_inputs(cf): allocate inputs
# - initialize_inputs(cf, inputs...): initialize inputs appropriately
# - execute(cf, inputs...): execute the configuration, and return the results
# - execute_reference(cf, inputs...): execute the reference implementation (for validation)
# - execute_baseline(cf, inputs...): execute the baseline implementation (for benchmarking)
# - verify(cf, reference, results): verify the results
#
# output
# - plot_results(best_configs): plot the results
include("wmma-contraction.jl")

#######

# Stop sampling when a configuration is really slow
const BENCH_MAX_SAMPLE_SECONDS = 1

# After a certain time, a measurement is considered stale (i.e., because of a crash).
# The time here determines when to ignore a lock. Note that it is automatically
# multiplied by 5 when the process is still alive, so it shouldn't be too large.
const BENCH_STALE_AGE = 60

const PIDFILE = joinpath(@__DIR__, "tuning.pid")

# Retry configurations with these categories.
const RETRY_CATEGORIES = ["oom", "crashed"]

#####

# Stop gathering samples for plot if we have spent this much time...
# 60 seconds/configuration * 32 configurations = 32 minutes for plot.
const PLOT_NUM_SECONDS = 60

# ... but have at least 100 samples.
const PLOT_MIN_NUM_SAMPLES = 100

# Group samples in batches of 10 samples each.
const PLOT_BATCH_SIZE = 10

#######

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

function measure_config(row)
    @info "Measuring configuration $(repr_row(row))..."
    cf = get_config(row)

    # allocate inputs
    inputs = try
        # this is the only place where we allocate device memory.
        # other allocation failures will be reported as crashes.
        generate_inputs(cf)
    catch err
        bt = catch_backtrace()
        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

        if isa(err, OutOfGPUMemoryError)
            @info "Not enough memory for configuration $(repr_row(row))\n" * log
            return Inf, "oom"
        else
            rethrow()
        end
    end

    # compile and warm-up
    try
        execute(cf, inputs...)
    catch err
        bt = catch_backtrace()
        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

        # determine the cause of the error
        if isa(err, GemmKernels.ConfigError)
            @info "Skipping unsupported configuration $(repr_row(row))\n" * log
            return Inf, "config_error"
        end
        if isa(err, CUDA.InvalidIRError)
            @info "Failed to compile $(repr_row(row))\n" * log
            return Inf, "compilation_error"
        end

        @info "Unknown error processing $(repr_row(row))\n" * log
        return Inf, "unknown_error"
    end

    # initialize inputs and calculate the reference
    c_h = mkpidlock(PIDFILE; stale_age=BENCH_STALE_AGE) do
        initialize_inputs(cf, inputs...)
    end

    # settle down
    device_synchronize()
    GC.gc(true)

    # benchmark, but be quick about it (using `CUDA.@elapsed` instead of `CUDA.@profile`,
    # only taking the minimum time, etc)
    time = Inf
    reference_host, results_host = mkpidlock(PIDFILE; stale_age=BENCH_STALE_AGE) do
        # make sure we're starting with an idle device
        wait_if_throttling()

        # keep benchmarking until the time isn't improving anymore
        results = nothing
        while true
            new_time = CUDA.@elapsed begin
                results = execute(cf, inputs...)
            end
            if new_time > time
                break
            end
            time = new_time

            # if this configuration is really slow, don't even bother running a second time
            if time > BENCH_MAX_SAMPLE_SECONDS
                break
            end
        end

        # compute the reference (may mutate inputs, so we need to do this last)
        reference = execute_reference(cf, inputs...)

        # copy results to host, so that we can verify without additional GPU allocations
        Array(reference), Array(results)
    end

    # verify the results
    if !verify(cf, reference_host, results_host)
        @warn "Configuration produced invalid result: $(repr_row(row))"
        return time, "invalid_result"
    end

    return time, "success"
end

function prettytime(time)
    time == Inf && return "Inf"

    # timescale
    time *= 1e9
    scale, unit = if time < 1e3
        1, "ns"
    elseif time < 1e6
        1e3, "μs"
    elseif time < 1e9
        1e6, "ms"
    else
        1e9, "s"
    end

    rnd_time = round(time / scale; sigdigits=3)
    return "$rnd_time $unit"
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

function get_nvml_data(dev=NVML.Device(parent_uuid(device())))
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

function benchmark_best_configs(configs)
    configs = select(configs, Not([:category]))
    best_configs = select_best(configs)
    best_configs.category .= "pending"
    best_configs.time_spent .= 0.0
    best_configs.gemmkernels_times = [[] for _ in 1:size(best_configs, 1)]
    best_configs.baseline_times = [[] for _ in 1:size(best_configs, 1)]
    best_configs.gemmkernels_nvml = [[] for _ in 1:size(best_configs, 1)]
    best_configs.baseline_nvml = [[] for _ in 1:size(best_configs, 1)]

    p = Progress(size(best_configs, 1); desc="Benchmarking", showspeed=true)
    for config_row in eachrow(best_configs)
        cf = get_config(config_row)
        inputs = generate_inputs(cf)

        @info "Profiling configuration $(repr_row(config_row))..."

        for baseline in [false, true]
            for i in 1:PLOT_BATCH_SIZE
                wait_if_throttling()

                start_time = time()

                push!(config_row[if baseline "baseline_nvml" else "gemmkernels_nvml" end], get_nvml_data())

                if baseline
                    prof = CUDA.@profile concurrent=false execute_baseline(cf, inputs...)
                else
                    prof = CUDA.@profile concurrent=false execute(cf, inputs...)
                end

                push!(config_row[if baseline "baseline_times" else "gemmkernels_times" end], sum(prof.device[!, "stop"] - prof.device[!, "start"]))

                config_row["time_spent"] += (time() - start_time)
            end
        end

        if got_enough_samples(config_row)
            config_row["category"] = "done"
        end

        # Update progress bar.
        next!(p)
    end

    best_configs
end

function addworkers(X; memory)
    env = [
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => string(memory),
    ]
    exeflags = [
        "--project=$(Base.active_project())",
        "--heap-size-hint=$memory"
    ]

    procs = addprocs(X; exeflags, env)
    @everywhere procs include($(joinpath(@__DIR__, "tune.jl")))
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
        configs.category .= "pending"
        configs.time .= Inf
        @info "Generated $(size(configs, 1)) configurations."

        # (2) Filter configurations where we can determine upfront that they are unsupported.
        @info "Finding configurations that we know are unsupported a-priori..."
        @showprogress desc="Filtering configurations..." for config_row in eachrow(configs)
            try
                cf = get_config(config_row)
            catch err
                if isa(err, GemmKernels.ConfigError)
                    config_row["category"] = "skipped"
                else
                    rethrow()
                end
            end
        end

        @info "Skipping $(counter(configs[!, "category"])["skipped"]) configurations."

        open(config_path, "w") do io
            serialize(io, configs)
        end
    end

    # (3) Measure performance of configurations.

    jobs = filter(1:size(configs, 1)) do i
        configs[i, :category] in ["pending"; RETRY_CATEGORIES]
    end
    shuffle!(jobs)

    njobs = length(jobs)
    p = Progress(njobs; desc="Parameter sweep", showspeed=true)
    @info "Need to perform parameter sweep over $(njobs) configurations."

    max_memory_usage = 0
    @showprogress desc="Determining memory usage..." for config_row in eachrow(configs)
        config_row.category in ["pending"; RETRY_CATEGORIES] || continue
        cf = get_config(config_row)
        max_memory_usage = max(max_memory_usage, sizeof(cf))
    end
    max_memory_usage += 128*2^20                    # there's always some overhead
    worker_memory_usage = max_memory_usage + 2^30   # includes CUDA context size, etc

    if njobs > 0
        # Spawn workers
        cpu_memory = Sys.free_memory()
        gpu_memory = CUDA.available_memory()
        max_workers = min(
            floor(Int, cpu_memory / worker_memory_usage),
            floor(Int, gpu_memory / worker_memory_usage),
            Sys.CPU_THREADS,
            njobs+1
        )
        addworkers(max(1, max_workers-1); memory=max_memory_usage)
        @info "Starting tuning script for device $(name(device())) using $(nworkers()) workers..."
    end

    results = Channel(Inf)
    @sync begin
        # measure configurations on workers
        worker_jobs = Dict()
        for worker in workers()
            errormonitor(@async begin
                while length(jobs) > 0
                    i = popfirst!(jobs)
                    config_row = configs[i, :]
                    worker_jobs[worker] = (; start_time=time(), i)
                    try
                        config_row.time, config_row.category =
                            remotecall_fetch(measure_config, worker, NamedTuple(config_row))

                        # keep memory usage under control
                        remotecall(worker) do
                            CUDA.reclaim()
                        end
                    catch err
                        config_row.category = "crashed"

                        bt = catch_backtrace()
                        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                        @error "Unexpected exception on worker $worker: $log"

                        # recycle our worker
                        rmprocs(worker; waitfor=30)
                        worker = addworkers(1)[1]
                    finally
                        delete!(worker_jobs, worker)
                        push!(results, (worker, i))

                        # Consider retrying failed configurations
                        if config_row.category in RETRY_CATEGORIES
                            push!(jobs, i)
                        end
                    end
                end
            end)
        end

        # monitor progress
        errormonitor(@async begin
            while !isempty(jobs) || !isempty(results)
                # process results
                if isready(results)
                    while isready(results)
                        worker, i = take!(results)

                        # Update configuration
                        config_row = configs[i, :]
                        @info "Result from worker $worker for $(repr_row(config_row)): $(config_row.category) -- $(prettytime(config_row.time))"
                    end

                    # Save results in case the process crashes.
                    open(config_path, "w") do io
                        serialize(io, configs)
                    end
                end

                # update the progress bar
                function showvalues()
                    vals = []

                    # worker stats
                    for worker in workers()
                        job = get(worker_jobs, worker, nothing)
                        if job === nothing
                            push!(vals, ("worker $worker", "idle"))
                        else
                            config_row = configs[job.i, :]
                            elapsed = time() - job.start_time
                            push!(vals, ("worker $worker", "$(prettytime(elapsed)) @ $(repr_row(config_row))"))
                        end
                    end

                    push!(vals, ("", ""))

                    # job state
                    category_counters = Dict(counter(configs[!, "category"]))
                    for k in sort(collect(keys(category_counters)); by=k->category_counters[k])
                        abs = category_counters[k]
                        rel = round(100 * abs / sum(values(category_counters)); sigdigits=3)
                        push!(vals, (k, "$(abs) ($(rel)%)"))
                    end

                    push!(vals, ("", ""))

                    # gpu stats
                    dev = NVML.Device(parent_uuid(device()))
                    push!(vals, ("power usage", "$(NVML.power_usage(dev)) W"))
                    push!(vals, ("temperature", "$(NVML.temperature(dev)) °C"))
                    meminfo = NVML.memory_info(dev)
                    push!(vals, ("memory usage", "$(Base.format_bytes(meminfo.used)) / $(Base.format_bytes(meminfo.total))"))
                    utilization = NVML.utilization_rates(dev)
                    push!(vals, ("utilization", "$(round(100*utilization.compute; sigdigits=3))% compute, $(round(100*utilization.memory; sigdigits=3))% memory"))

                    vals
                end
                nfinished = njobs - length(jobs)
                update!(p, nfinished; showvalues)

                sleep(5)
            end
            finish!(p)
        end)
    end

    # Save data for final iteration.
    open(config_path, "w") do io
        serialize(io, configs)
    end

    # Kill workers
    if njobs > 0
        rmprocs(workers()...; waitfor=30)
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
    isfile(PIDFILE) && error("Another tuning process is already running. If this is not the case, please remove the file $(PIDFILE).")
    main()
end
