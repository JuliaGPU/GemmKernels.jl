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
# The interface uses 3 kinds of objects:
# - a problem, representing the details of the operation we're performing
#   (e.g., a 2048x2048x2048 Float32xFloat32xFloat32 WMMA GEMM)
# - the (input and output) data used when executing the problem
# - the parameters used to customize the execution (e.g. the operator, block shape, etc)
#
# This interface is mostly defined in `configs/configs.jl`. The tuning script layers
# some additional abstraction on top of this for the purpose of iteration & serialization:
# - generate_problems(): return a list of input problems
# - generate_configs(problem): for this problem, generate configurations to sweep over.
#   each configuration should also identify the problem it belongs to.
# - select_configs(configs, problem): from a pre-existing list of configurations, list
#   those matching a specific problem.
# - repr_row(config): pretty-print a configuration
# - create_params(row): create configuration parameters for this row
# - select_best(configs): given a list of measured configurations (i.e. with an added
#   :time column), return the best configuration for each problem
#
# output
# - plot_results(best_configs): plot the results
include("wmma-contraction.jl")

#######

# Stop sampling when a configuration is really slow
# If unset, we will stop measuring if we're 2x slower than the baseline,
#const BENCH_MAX_TIME = 1

# After a certain time, a measurement is considered stale (i.e., because of a crash).
# The time here determines when to ignore a lock. Note that it is automatically
# multiplied by 5 when the process is still alive, so it shouldn't be too large.
const BENCH_STALE_AGE = 60

const PIDFILE = joinpath(@__DIR__, "tuning.pid")

# Whether we stop after beating the baseline, or continue until we've tested every config.
const EXHAUSTIVE = false

# Retry configurations with these categories.
const RETRY_STATUSSES = ["oom", "crashed"]

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

function measure_config(problem, config, max_time, reference_result)
    @info "Measuring configuration $(repr_row(config))..."

    # allocate data
    data = try
        # this is the only place where we allocate device memory.
        # other allocation failures will be reported as crashes.
        allocate_data(problem)
    catch err
        bt = catch_backtrace()
        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

        if isa(err, OutOfGPUMemoryError)
            @info "Not enough memory for configuration $(repr_row(config))\n" * log
            return (;), "oom"
        else
            rethrow()
        end
    end

    try
        # compile and warm-up
        params = create_params(config)
        args = nothing
        try
            args = prepare(problem; params...)
            execute(problem, data...; args...)
        catch err
            bt = catch_backtrace()
            log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

            # determine the cause of the error
            if isa(err, GemmKernels.ConfigError)
                @info "Skipping unsupported configuration $(repr_row(config))\n" * log
                return (;), "config_error"
            end
            if isa(err, CUDA.InvalidIRError)
                @info "Failed to compile $(repr_row(config))\n" * log
                return (;), "compilation_error"
            end
            if isa(err, OutOfGPUMemoryError)
                # XXX: this shouldn't happen here as we already allocated all memory,
                #      however it can occur during kernel launches or other CUDA calls
                #      when very close to the memory limit.
                @info "Not enough memory for configuration $(repr_row(config))\n" * log
                return (;), "oom"
            end

            @info "Unknown error processing $(repr_row(config))\n" * log
            return (;), "unknown_error"
        end

        # initialize data
        initializing = mkpidlock(PIDFILE; stale_age=BENCH_STALE_AGE) do
            @elapsed initialize_data(problem, data...)
        end

        # settle down
        device_synchronize()
        GC.gc(true)

        # benchmark, but be quick about it (using `CUDA.@elapsed` instead of `CUDA.@profile`,
        # only taking the minimum time, etc)
        measurements = Float64[]
        result, exclusives = mkpidlock(PIDFILE; stale_age=BENCH_STALE_AGE) do
            # make sure we're starting with an idle device
            settling = @elapsed wait_if_throttling()

            # perform time measurements
            result = nothing
            measuring = @elapsed while true
                best_time = minimum(measurements; init=Inf)
                time = CUDA.@elapsed begin
                    result = execute(problem, data...; args...)
                end
                push!(measurements, time)

                # keep benchmarking until the time isn't improving anymore
                if time > best_time
                    break
                end

                # if this configuration is really slow, bail out
                if time > max_time
                    result = nothing
                    break
                end
            end

            # copy results to host, so that we can verify without additional GPU allocations
            copying = @elapsed begin
                if result !== nothing
                    result = Array(result)
                end
            end

            return result, (; initializing, settling, measuring, copying)
        end

        # verify the results
        if result !== nothing && !verify(problem, reference_result, result)
            @warn "Configuration produced invalid result: $(repr_row(config))"
            return (; measurements, exclusives), "invalid_result"
        end

        if minimum(measurements) > max_time
            return (; measurements, exclusives), "slow"
        end
        return (; measurements, exclusives), "success"
    finally
        # clean-up
        for arr in data
            CUDA.unsafe_free!(arr)
        end
        CUDA.reclaim()
    end
end

function timescale(time)
    if time < 1e-6
        1e9, "ns"
    elseif time < 1e-3
        1e6, "μs"
    elseif time < 1
        1e3, "ms"
    else
        1, "s"
    end
end

function prettytime(time)
    time == Inf && return "Inf"

    # timescale
    scale, unit = timescale(time)

    rnd_time = round(time * scale; sigdigits=3)
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

function got_enough_samples(config)
    gk, bl = config["gemmkernels_times"], config["baseline_times"]

    (length(gk) < PLOT_MIN_NUM_SAMPLES) && return false
    (length(bl) < PLOT_MIN_NUM_SAMPLES) && return false

    config["time_spent"] >= PLOT_NUM_SECONDS
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
    configs = select(configs, Not([:status]))
    best_configs = select_best(configs)
    best_configs.status .= "pending"
    best_configs.time_spent .= 0.0
    best_configs.gemmkernels_times = [Float64[] for _ in 1:size(best_configs, 1)]
    best_configs.baseline_times = [Float64[] for _ in 1:size(best_configs, 1)]

    p = Progress(size(best_configs, 1); desc="Benchmarking", showspeed=true)
    for config in eachrow(best_configs)
        problem = create_problem(config)
        data = allocate_data(problem)

        @info "Profiling configuration $(repr_row(config))..."

        params = create_params(config)
        for baseline in [false, true]
            for i in 1:PLOT_BATCH_SIZE
                wait_if_throttling()

                start_time = time()

                if baseline
                    prof = CUDA.@profile concurrent=false execute_baseline(problem, data...)
                else
                    prof = CUDA.@profile concurrent=false execute(problem, data...; params...)
                end

                push!(config[if baseline "baseline_times" else "gemmkernels_times" end], sum(prof.device[!, "stop"] - prof.device[!, "start"]))

                config["time_spent"] += (time() - start_time)
            end
        end

        if got_enough_samples(config)
            config.status = "done"
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

function merge_configs(all_configs, configs; on)
    if all_configs === nothing
        return configs
    end

    # merge in results from previous runs
    configs = leftjoin(configs, all_configs; on, makeunique=true)
    configs.time = coalesce.(configs.time_1, configs.time)
    configs.status = coalesce.(configs.status_1, configs.status)
    configs = select(configs, Not([:time_1, :status_1]))

    # find the configs that are new, and merge them back
    other_configs = antijoin(all_configs, configs; on)
    all_configs = vcat(configs, other_configs)

    all_configs
end

function main()
    # get rid of workers from a previous run
    if length(workers()) > 1
        rmprocs(workers()...; waitfor=30)
    end

    # (0) Load previous results from disk
    config_path = joinpath(@__DIR__, "configs.bin")
    all_configs = nothing
    if isfile(config_path)
        @info "Loading configurations from disk..."
        try
            all_configs = deserialize(config_path)
            @info "Loaded $(size(all_configs, 1)) configurations."
        catch err
            @error "Error while loading configurations from disk: $(sprint(Base.showerror, err)))"
            mv(config_path, "$(config_path).broken")
        end
    end

    # (1) Process each unique problem.
    problems = generate_problems()
    for (problem_idx, problem) in enumerate(problems)
        # generate this problem's configurations
        configs = generate_configs(problem)
        config_keys = names(configs)
        configs.status .= "new"
        configs.time .= Inf
        all_configs = merge_configs(all_configs, configs; on=config_keys)
        configs = select_configs(all_configs, problem)

        # Filter configurations where we can determine upfront that they are unsupported.
        @showprogress desc="Filtering configurations..." for config in eachrow(configs)
            if config.status != "new"
                continue
            end

            try
                params = create_params(config)
                prepare(problem; params...)
                config.status = "pending"
            catch err
                if isa(err, GemmKernels.ConfigError)
                    config.status = "skipped"
                else
                    rethrow()
                end
            end
        end
        serialize(config_path, all_configs)

        # Measure baseline performance of problem.
        target_time, reference_result = let
            data = allocate_data(problem)
            initialize_data(problem, data...)

            # warm-up
            args = prepare_baseline(problem, data...)
            execute_baseline(problem, data...; args...)

            # measure baseline
            wait_if_throttling()
            time = CUDA.@elapsed(execute_baseline(problem, data...; args...))

            # calculate reference
            result = Array(calculate_reference(problem, data...))

            for arr in data
                CUDA.unsafe_free!(arr)
            end
            CUDA.reclaim()

            time, result
        end

        # Find current best time
        best_time = minimum(configs.time)
        if !EXHAUSTIVE && best_time < target_time
            continue
        end

        # Determine jobs we can still run
        jobs = filter(1:size(configs, 1)) do i
            configs[i, :status] in ["pending"; RETRY_STATUSSES]
        end
        if isempty(jobs)
            continue
        end
        shuffle!(jobs)
        njobs = length(jobs)

        # Determine memory usage
        max_memory_usage = sizeof(problem) + 128*2^20       # there's always some overhead
        worker_memory_usage = max_memory_usage + 1500*2^20  # CUDA context, etc

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

        # determine how long each measurement can take
        max_time = if @isdefined(BENCH_MAX_TIME)
            BENCH_MAX_TIME
        else
            2 * target_time
        end

        # keep track of time spent in exclusive sections
        exclusive_times = Dict{Symbol, Float64}()

        @info "Need to perform parameter sweep over $(njobs) configurations."
        p = Progress(njobs; desc="$(problem) [$problem_idx/$(length(problems))]", showspeed=true)
        results = Channel(Inf)
        sweep_start = time()
        @sync begin
            # measure configurations on workers
            worker_jobs = Dict()
            for worker in workers()
                errormonitor(@async begin
                    while !isempty(jobs)
                        i = popfirst!(jobs)
                        config = configs[i, :]
                        worker_jobs[worker] = (; start_time=time(), i)
                        try
                            times, config.status =
                                remotecall_fetch(measure_config, worker,
                                                 problem, NamedTuple(config),
                                                 max_time, reference_result)

                            # save the best measurement
                            if haskey(times, :measurements)
                                config.time = minimum(times.measurements)
                            end

                            # keep track of time spend in exclusive sections
                            if haskey(times, :exclusives)
                                for k in keys(times.exclusives)
                                    v = times.exclusives[k]
                                    exclusive_times[k] = get(exclusive_times, k, 0.0) + v
                                end
                            end
                        catch err
                            config.status = "crashed"

                            bt = catch_backtrace()
                            log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                            @error "Unexpected exception on worker $worker: $log"

                            # recycle our worker
                            rmprocs(worker; waitfor=30)
                            worker = addworkers(1; memory=max_memory_usage)[1]
                        finally
                            delete!(worker_jobs, worker)
                            push!(results, (worker, i))

                            # Consider retrying failed configurations
                            if config.status in RETRY_STATUSSES
                                push!(jobs, i)
                            end
                        end
                    end
                end)
            end

            # monitor progress
            errormonitor(@async begin
                times = []
                while !isempty(jobs) || !isempty(worker_jobs) || !isempty(results)
                    # process results
                    if isready(results)
                        while isready(results)
                            worker, i = take!(results)

                            # Update configuration
                            config = configs[i, :]
                            best_time = min(best_time, config.time)
                            @info "Result from worker $worker for $(repr_row(config)): $(config.status) -- $(prettytime(config.time))"
                        end

                        # Save results in case the process crashes.
                        serialize(config_path, all_configs)
                    end
                    if !isinf(best_time)
                        push!(times, best_time)
                    end

                    # update the progress bar
                    function showvalues()
                        vals = []

                        # configuration times
                        push!(vals, ("target time", prettytime(target_time)))
                        if !isinf(best_time)
                            push!(vals, ("best time", prettytime(best_time)))
                        end

                        push!(vals, ("", ""))

                        # how much time we spent in exclusive sections
                        if !isempty(exclusive_times)
                            push!(vals, ("total", prettytime(time() - sweep_start)))

                            for (k, v) in exclusive_times
                                push!(vals, (k, prettytime(v)))
                            end

                            push!(vals, ("", ""))
                        end

                        # job state
                        category_counters = Dict(counter(configs[!, "status"]))
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
                    update!(p, nfinished; showvalues, valuecolor=:normal)

                    # Check if we're done
                    if !EXHAUSTIVE && best_time < target_time
                        break
                    end

                    sleep(5)
                end

                empty!(jobs)

                # Kill workers
                rmprocs(workers()...; waitfor=30)
            end)
        end
        return
    end

    # And load again, for good measure.
    all_configs = deserialize(config_path)

    # (4) Select best configurations, and benchmark.
    best_configs_path = joinpath(@__DIR__, "best-configs.bin")
    best_configs = nothing
    if isfile(best_configs_path)
        try
            @info "Loading best configurations from disk..."
            best_configs = deserialize(best_configs_path)
        catch err
            @error "Error while loading best configurations from disk: $(sprint(Base.showerror, err)))"
            mv(best_configs_path, "$(best_configs_path).broken")
        end
    end
    if best_configs === nothing
        @info "Benchmarking configurations for plot..."
        best_configs = benchmark_best_configs(all_configs)

        serialize(best_configs_path, best_configs)
    end


    # (5) Plotting results
    @info "Plotting results..."
    plot_results(best_configs)
end

if !isinteractive() && myid() == 1
    isfile(PIDFILE) && error("Another tuning process is already running. If this is not the case, please remove the file $(PIDFILE).")
    main()
end
