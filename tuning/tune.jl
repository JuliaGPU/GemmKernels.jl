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
using Profile
using Adapt

if myid() == 1
    using Plots
end


############################################################################################

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
#
# output
# - plot_best_configs(best_configs): plot the results
isinteractive() || include("wmma-contraction.jl")

############################################################################################

const PIDFILE = joinpath(@__DIR__, "tuning.pid")

# After a certain time, a measurement is considered stale (i.e., because of a crash).
# The time here determines when to ignore a lock. Note that it is automatically
# multiplied by 5 when the process is still alive, so it shouldn't be too large.
const PIDFILE_STALE_AGE = 60

# Whether we stop after beating the baseline, or continue until we've tested every config.
const EXHAUSTIVE = false

# The time limit for measuring configurations, in seconds. This will be used to determine
# a per-problem time limit.
const TIME_LIMIT = 24*3600

# Retry configurations with these categories.
const RETRY_STATUSSES = ["oom", "crashed"]

# When benchmarking the best configurations, how many candidates to consider.
const BENCHMARK_CANDIDATES = 3

# When benchmarking the best configurations, how many samples to take.
const BENCHMARK_SAMPLES = 5

############################################################################################

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

function addworkers(X; gpu_mem_target, cpu_mem_target)
    env = [
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => string(gpu_mem_target),
    ]
    exeflags = [
        "--project=$(Base.active_project())",
        "--heap-size-hint=$cpu_mem_target"
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

############################################################################################

function measure_config(problem, config, best_time, reference_result)
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
            args = prepare(problem, data...; params...)
            execute(problem, data...; args...)
            synchronize()
            # XXX: prevent this from actually executing? it may influence the measurements below
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
        initializing = mkpidlock(PIDFILE; stale_age=PIDFILE_STALE_AGE) do
            @elapsed CUDA.@sync initialize_data(problem, data...; params...)
        end

        # settle down
        device_synchronize()
        GC.gc(true)

        # take time measurements
        max_time = 2 * best_time
        measurements = Float64[]
        result, exclusives = mkpidlock(PIDFILE; stale_age=PIDFILE_STALE_AGE) do
            # make sure we're starting with an idle device
            settling = @elapsed wait_if_throttling()

            # first measurement: check if the configuration is even worth measuring
            measuring = @elapsed begin
                time = CUDA.@elapsed blocking=true execute(problem, data...; args...)
            end
            push!(measurements, time)
            if time > max_time
                return nothing, (; settling, measuring)
            end

            # second measurement: fetch results to verify (outside of the pidlock)
            initializing = @elapsed CUDA.@sync initialize_data(problem, data...; params...)
            measuring += @elapsed begin
                time = CUDA.@elapsed blocking=true begin
                    result = execute(problem, data...; args...)
                end
                push!(measurements, time)
            end
            copying = @elapsed begin
                # NOTE: we adapt, since the result may be a non-contiguous view,
                #       and copying those allocates GPU memory.
                result = adapt(Array, result)
            end

            # subsequent measurements: keep going until time doesn't improve
            measuring += @elapsed begin
                while need_more_measurements(measurements)
                    time = CUDA.@elapsed blocking=true execute(problem, data...; args...)
                    push!(measurements, time)
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

function benchmark_configs(all_configs)
    # we only care about successful configurations
    all_configs = all_configs[all_configs[!, "status"] .== "success", :]
    select!(all_configs, Not(:status))

    problems = generate_problems()

    # gather the best configurations for each problem
    candidate_configs = similar(all_configs, 0)
    for problem in problems
        configs = select_configs(all_configs, problem)
        append!(candidate_configs, first(sort(configs, :time), BENCHMARK_CANDIDATES))
    end

    # get rid of the old time measurements
    select!(all_configs, Not(:time))
    select!(candidate_configs, Not(:time))

    # benchmark
    nbenchmarks = size(candidate_configs, 1) + size(problems, 1)
    p = Progress(nbenchmarks * BENCHMARK_SAMPLES; desc="Benchmarking", showspeed=true)
    best_configs = similar(all_configs, 0)
    best_configs.gemmkernels_times = Vector{Float64}[]
    best_configs.baseline_times = Vector{Float64}[]
    for problem in problems
        data = allocate_data(problem)

        # measure baseline
        baseline_times = []
        let
            # warm-up
            args = prepare_baseline(problem, data...)
            execute_baseline(problem, data...; args...)
            wait_if_throttling()

            for i in 1:BENCHMARK_SAMPLES
                prof = CUDA.@profile concurrent=false execute_baseline(problem, data...; args...)
                time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
                push!(baseline_times, time)
                next!(p)
            end
        end

        # measure gemmkernels
        best_config = nothing
        for config in eachrow(select_configs(candidate_configs, problem))
            # warm-up
            params = create_params(config)
            args = prepare(problem, data...; params...)
            execute(problem, data...; args...)
            wait_if_throttling()

            times = []
            for i in 1:BENCHMARK_SAMPLES
                prof = CUDA.@profile concurrent=false execute(problem, data...; args...)
                time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
                push!(times, time)
                next!(p)
            end

            if best_config === nothing || minimum(times) < minimum(best_config.gemmkernels_times)
                best_config = (; gemmkernels_times = times, copy(config)...)
            end
        end

        best_config = (; baseline_times, best_config...)
        push!(best_configs, best_config)
    end

    return best_configs
end

function need_more_measurements(times)
    if length(times) < 3
        return true
    end

    # check that the last two measurements didn't improve the minimum
    best_time = minimum(times)
    return times[end-1] < best_time || times[end] < best_time
end

function main()
    #
    # Phase 1: Gather baseline performance and results
    #

    baseline_performances = []
    reference_results = mktempdir()

    # XXX: do this on a worker
    problems = generate_problems()
    @showprogress desc="Measuring baselines..." for (problem_idx, problem) in enumerate(problems)
        data = allocate_data(problem)
        initialize_data(problem, data...)

        # calculate
        result = Array(calculate_reference(problem, data...))

        # warm-up
        args = prepare_baseline(problem, data...)
        execute_baseline(problem, data...; args...)
        wait_if_throttling()

        # measure
        measurements = []
        while need_more_measurements(measurements)
            prof = CUDA.@profile concurrent=false execute_baseline(problem, data...; args...)
            time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
            push!(measurements, time)
        end

        for arr in data
            CUDA.unsafe_free!(arr)
        end
        push!(baseline_performances, minimum(measurements))
        serialize(joinpath(reference_results, "$(problem_idx).bin"), result)
    end

    # reclaim memory
    GC.gc(true)
    CUDA.reclaim()


    #
    # Phase 2: Sweep parameters
    #

    # Load previous results from disk
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
    function checkpoint()
        relevant_configs = filter(all_configs) do row
            # skip configurations we didn't process
            !in(row.status, ["pending", "skipped"])
        end
        temp_path = "$(config_path).$(getpid)"
        serialize(temp_path, relevant_configs)
        mv(temp_path, config_path; force=true)
    end

    for (problem_idx, problem) in enumerate(problems)
        println("Processing $problem [$problem_idx/$(length(problems))]...")

        data = allocate_data(problem)
        target_time = baseline_performances[problem_idx]
        reference_result = deserialize(joinpath(reference_results, "$(problem_idx).bin"))
        println(" - target time: $(prettytime(target_time))")

        # generate this problem's configurations
        configs = generate_configs(problem)
        config_keys = names(configs)
        configs.status .= "new"
        configs.time .= Inf
        all_configs = merge_configs(all_configs, configs; on=config_keys)
        configs = select_configs(all_configs, problem)

        # See if there's anything we need to do
        best_time = minimum(filter(x->x.status == "success", configs).time; init=Inf)
        println(" - best time so far: $(prettytime(best_time))")
        if !EXHAUSTIVE && best_time < target_time
            println("... fast enough already")
        end

        # Filter configurations where we can determine upfront that they are unsupported.
        # XXX: do this on a worker
        @showprogress desc="Filtering configurations..." for config in eachrow(configs)
            if config.status != "new"
                continue
            end

            try
                params = create_params(config)
                prepare(problem, data...; params...)
                # XXX: lots of failures only happen during `execute()`
                config.status = "pending"
            catch err
                if isa(err, GemmKernels.ConfigError)
                    config.status = "skipped"
                else
                    bt = catch_backtrace()
                    log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                    @error "Unexpected error preparing $problem: $log" config

                    rethrow()
                end
            end
        end
        checkpoint()

        # Get rid of the data
        for arr in data
            CUDA.unsafe_free!(arr)
        end
        GC.gc(true)
        CUDA.reclaim()
        data = nothing

        jobs = filter(1:size(configs, 1)) do i
            configs[i, :status] in ["pending"; RETRY_STATUSSES]
        end
        if !isempty(jobs)
            shuffle!(jobs)
            njobs = length(jobs)

            # Determine memory usage
            println(" - problem memory requirement: $(Base.format_bytes(sizeof(problem)))")
            gpu_mem_target = sizeof(problem) + 32*2^20      # allow minimal unaccounted allocations
            gpu_mem_max = gpu_mem_target + 2000*2^20        # overhead from CUDA context, etc
            # TODO: how come the context grows so large?
            cpu_mem_target = sizeof(problem)
            cpu_mem_max = sizeof(problem) + 2500*2^20       # compilation headroom
            println(" - allowed worker memory use: $(Base.format_bytes(gpu_mem_max)) GPU memory, $(Base.format_bytes(cpu_mem_max)) CPU memory")

            # Spawn workers
            cpu_memory = Sys.free_memory()
            gpu_memory = CUDA.available_memory()
            max_workers = min(
                floor(Int, cpu_memory / cpu_mem_max),
                floor(Int, gpu_memory / gpu_mem_max),
                Sys.CPU_THREADS,
                njobs+1
            )
            total_workers = max(1, max_workers-nworkers())
            addworkers(total_workers; cpu_mem_target, gpu_mem_target)
            @assert nworkers() == total_workers

            # Process jobs!
            p = Progress(njobs; desc="Measuring configurations", showspeed=true)
            exclusive_times = Dict{Symbol, Float64}()
            results = Channel(Inf)
            sweep_start = time()
            @sync begin
                # Measuring tasks
                worker_jobs = Dict()
                for worker in workers()
                    errormonitor(@async begin
                        function recycle_worker()
                            rmprocs(worker)
                            worker = addworkers(1; cpu_mem_target, gpu_mem_target)[1]
                        end

                        while !isempty(jobs)
                            i = popfirst!(jobs)
                            config = configs[i, :]
                            worker_jobs[worker] = (; start_time=time(), i)
                            try
                                times, config.status =
                                    remotecall_fetch(measure_config, worker,
                                                     problem, NamedTuple(config),
                                                     best_time, reference_result)

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
                                recycle_worker()
                            finally
                                delete!(worker_jobs, worker)
                                push!(results, (worker, i))

                                # Consider retrying failed configurations
                                if config.status in RETRY_STATUSSES
                                    push!(jobs, i)
                                end
                            end

                            # make sure our worker doesn't overshoot its memory budget.
                            # XXX: why does this happen? does the CUDA context grow
                            #      because of compiling so many kernels?
                            worker_pid, worker_rss = try
                                remotecall_fetch(worker) do
                                    getpid(), Sys.maxrss()
                                end
                            catch
                                # this can happen when the worker is exiting
                                nothing, nothing
                            end
                            if worker_pid !== nothing
                                try
                                    dev = NVML.Device(parent_uuid(device()))
                                    procs = NVML.compute_processes(dev)
                                    if haskey(procs, worker_pid)
                                        info = procs[worker_pid]
                                        if info.used_gpu_memory !== missing &&
                                           info.used_gpu_memory > gpu_mem_max
                                            @warn "Worker $worker exceeded GPU memory budget: $(Base.format_bytes(info.used_gpu_memory))"
                                            recycle_worker()
                                        end
                                    end
                                catch err
                                    isa(err, NVML.NVMLError) || rethrow()
                                    err.code in [NVML.ERROR_NOT_SUPPORTED,
                                                 NVML.ERROR_NO_PERMISSION] || rethrow()
                                end
                            end
                            if worker_rss !== nothing
                                # XXX: instead of checking periodically, run under a memory limiter?
                                if worker_rss > cpu_mem_max
                                    @warn "Worker $worker exceeded CPU memory budget: $(Base.format_bytes(worker_rss))"
                                    recycle_worker()
                                end
                            end
                        end
                    end)
                end

                # Result processing task
                errormonitor(@async begin
                    while !isempty(jobs) || !isempty(worker_jobs) || !isempty(results)
                        # process results
                        if isready(results)
                            while isready(results)
                                worker, i = take!(results)

                                # Update configuration
                                config = configs[i, :]
                                if config.status == "success"
                                    best_time = min(best_time, config.time)
                                end
                                @info "Result from worker $worker for $(repr_row(config)): $(config.status) -- $(prettytime(config.time))"
                            end
                            checkpoint()
                        end

                        # update the progress bar
                        function showvalues()
                            vals = []

                            push!(vals, ("workers", "$(length(workers())) / $(total_workers)"))

                            push!(vals, ("", ""))

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
                            print(" - stopping after beating target time\n")
                            break
                        end
                        if (time() - sweep_start) > (TIME_LIMIT / length(problems))
                            print(" - stopping after hitting time limit\n")
                            break
                        end

                        sleep(5)
                    end

                    empty!(jobs)
                end)
            end

            println(" - final time: $(prettytime(best_time))")

            # kill workers
            rmprocs(workers()...)
            @assert workers() == [myid()]
        end
        checkpoint()
    end


    #
    # Phase 3: Process results
    #

    # Select best configurations, and benchmark.
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
        best_configs = benchmark_configs(all_configs)

        serialize(best_configs_path, best_configs)
    end


    # Plotting results
    @info "Plotting results..."
    plot_best_configs(best_configs)
end

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

if !isinteractive() && myid() == 1
    isfile(PIDFILE) && error("Another tuning process is already running. If this is not the case, please remove the file $(PIDFILE).")
    main()
end
