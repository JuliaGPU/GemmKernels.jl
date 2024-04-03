using CUDA, GemmKernels
using DataFrames
using DataStructures: counter
using Dates
using Distributed
using FileWatching.Pidfile
using Logging
using LoggingExtras
using ProgressMeter: Progress, next!, update!, finish!, cancel, @showprogress
using Serialization
using Statistics
using StatsBase: percentile
using Random
using Profile
using Adapt
using Scratch

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
# - count_configs(problem): for this problem, count the number of configurations.
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
                cur_time = CUDA.@elapsed blocking=true execute(problem, data...; args...)
            end
            push!(measurements, cur_time)
            if cur_time > max_time
                return nothing, (; settling, measuring)
            end

            # second measurement: fetch results to verify (outside of the pidlock)
            initializing = @elapsed CUDA.@sync initialize_data(problem, data...; params...)
            settling += @elapsed wait_if_throttling()
            measuring += @elapsed begin
                cur_time = CUDA.@elapsed blocking=true begin
                    result = execute(problem, data...; args...)
                end
                push!(measurements, cur_time)
            end
            copying = @elapsed begin
                # NOTE: we adapt, since the result may be a non-contiguous view,
                #       and copying those allocates GPU memory.
                result = adapt(Array, result)
            end

            # subsequent measurements: keep going until time doesn't improve
            measuring += @elapsed begin
                while need_more_measurements(measurements)
                    cur_time = CUDA.@elapsed blocking=true execute(problem, data...; args...)
                    push!(measurements, cur_time)
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
        GC.gc(true)
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
        configs ===  nothing && continue
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
        select_configs(candidate_configs, problem) === nothing && continue

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
                cur_time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
                push!(baseline_times, cur_time)
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
                cur_time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
                push!(times, cur_time)
                next!(p)
            end

            if best_config === nothing || minimum(times) < minimum(best_config.gemmkernels_times)
                best_config = (; gemmkernels_times = times, copy(config)...)
            end
        end

        if best_config !== nothing
            best_config = (; baseline_times, best_config...)
            push!(best_configs, best_config)
        end
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
    # Phase 1: Prepare
    #

    problems = generate_problems()

    # Determine per-problem time limits
    nconfigs = sum(count_configs, problems)
    time_limits = []
    for problem in problems
        time_limits = push!(time_limits, TIME_LIMIT * count_configs(problem) / nconfigs)
    end

    # Gather baseline performance and results
    baseline_performances = []
    reference_results = get_scratch!(GemmKernels, "reference_results")
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
            cur_time = sum(prof.device[!, "stop"] - prof.device[!, "start"])
            push!(measurements, cur_time)
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

            # transitionary measure: remove status=new rows
            all_configs = all_configs[all_configs[!, "status"] .!= "new", :]

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
        println("\nProcessing $problem [$problem_idx/$(length(problems))]...")
        configs = select_configs(all_configs, problem)

        # See if there's anything we need to do
        best_time = Inf
        if configs !== nothing
            best_time = minimum(filter(x->x.status == "success", configs).time; init=Inf)
        end
        target_time = baseline_performances[problem_idx]
        println(" - target time: $(prettytime(target_time))")
        if best_time != Inf
            println(" - best time so far: $(prettytime(best_time))")
            if !EXHAUSTIVE && best_time < target_time
                println("   fast enough already")
                continue
            end
        end

        # augment the loaded configurations with new ones
        configs = generate_configs(problem)
        config_keys = names(configs)
        configs.status .= "new"
        configs.time .= Inf
        all_configs = merge_configs(all_configs, configs; on=config_keys)
        configs = select_configs(all_configs, problem)

        # Give configurations that ran into an unknown error another change
        # (note that we don't retry them within a run)
        configs[configs.status .== "unknown_error", :status] .= "new"

        # Filter new configurations where we can determine upfront that they are unsupported.
        let data = allocate_data(problem)
            # XXX: do this on a worker
            @showprogress desc="Filtering new configurations..." for config in eachrow(configs)
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
            for arr in data
                CUDA.unsafe_free!(arr)
            end
            GC.gc(true)
            CUDA.reclaim()
        end
        checkpoint()
        initial_count = count(configs.status .!= "pending")
        total_count = size(configs, 1)
        println(" - have processed $(round(100*initial_count/total_count; digits=2))% ($initial_count/$total_count) of configurations already")

        jobs = filter(1:size(configs, 1)) do i
            configs[i, :status] in ["pending"; RETRY_STATUSSES]
        end
        if !isempty(jobs)
            shuffle!(jobs)
            njobs = length(jobs)

            # Determine memory usage
            println(" - problem memory requirement: $(Base.format_bytes(sizeof(problem)))")
            gpu_mem_target = sizeof(problem) + 32*2^20      # allow minimal unaccounted allocations
            gpu_mem_max = gpu_mem_target + 1500*2^20        # overhead from CUDA context, etc
            # TODO: how come the context grows so large?
            cpu_mem_target = 3*sizeof(problem)              # 2 for the data, 1 for the comparison
            cpu_mem_max = sizeof(problem) + 2000*2^20       # compilation headroom

            # Spawn workers
            memory_margin = 0.9
            initial_cpu_memory = Sys.free_memory()
            initial_gpu_memory = CUDA.available_memory()
            max_workers = min(
                floor(Int, initial_cpu_memory * memory_margin / cpu_mem_max),
                floor(Int, initial_gpu_memory * memory_margin / gpu_mem_max),
                Sys.CPU_THREADS,
                njobs+1
            )
            println(" - using $max_workers workers with $(Base.format_bytes(cpu_mem_max))/$(Base.format_bytes(initial_cpu_memory)) CPU memory, $(Base.format_bytes(gpu_mem_max))/$(Base.format_bytes(initial_gpu_memory)) GPU memory")
            total_workers = max(1, max_workers-nworkers())
            addworkers(total_workers; cpu_mem_target, gpu_mem_target)
            @assert nworkers() == total_workers

            # Process jobs!
            println(" - time limit: $(prettytime(time_limits[problem_idx]))")
            reference_result = deserialize(joinpath(reference_results, "$(problem_idx).bin"))
            p = Progress(njobs; desc="Measuring configurations", showspeed=true)
            exclusive_times = Dict{Symbol, Float64}()
            results = Channel(Inf)
            sweep_start = time()
            @sync begin
                # Measuring tasks
                # NOTE: the job done by this task should be kept minimal, as it determines
                #       how quickly work can be submtited. we could switch to threads, but
                #       Distributed isn't threadsafe.
                worker_jobs = Dict()
                worker_memory_usage = Dict()
                for worker in workers()
                    errormonitor(@async begin
                        while !isempty(jobs)
                            # ensure we still have a worker
                            if worker === nothing
                                try
                                    worker = addworkers(1; cpu_mem_target, gpu_mem_target)[1]
                                catch err
                                    # give up for this problem
                                    @error "Failed to add worker: $(sprint(Base.showerror, err))"
                                    break
                                end
                            end

                            # get a job
                            isempty(jobs) && break
                            i = popfirst!(jobs)
                            config = configs[i, :]
                            worker_jobs[worker] = (; start_time=time(), i)
                            kill_worker = false
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

                                # store memory usage statistics
                                worker_pid, worker_cpu_mem = remotecall_fetch(worker) do
                                    getpid(), Sys.maxrss()
                                end
                                worker_gpu_mem = try
                                    dev = NVML.Device(parent_uuid(device()))
                                    procs = NVML.compute_processes(dev)
                                    if haskey(procs, worker_pid)
                                        info = procs[worker_pid]
                                        coalesce(info.used_gpu_memory, 0)
                                    else
                                        0
                                    end
                                catch err
                                    isa(err, NVML.NVMLError) || rethrow()
                                    err.code in [NVML.ERROR_NOT_SUPPORTED,
                                                 NVML.ERROR_NO_PERMISSION] || rethrow()
                                end
                                worker_memory_usage[worker] = (; cpu=worker_cpu_mem,
                                                                 gpu=worker_gpu_mem)

                                # check if we're running close to OOM
                                current_cpu_memory = Sys.free_memory()
                                current_gpu_memory = CUDA.available_memory()
                                cpu_hungry_worker = sort(collect(keys(worker_memory_usage));
                                                         by=worker->-worker_memory_usage[worker].cpu)[end]
                                gpu_hungry_worker = sort(collect(keys(worker_memory_usage));
                                                         by=worker->-worker_memory_usage[worker].gpu)[end]
                                if current_cpu_memory < (1-memory_margin)*initial_cpu_memory && cpu_hungry_worker == worker
                                    @warn "System running low on on CPU memory ($(Base.format_bytes(current_cpu_memory))/$(Base.format_bytes(initial_cpu_memory)) available); recycling worker $worker using $(Base.format_bytes(worker_memory_usage[worker].cpu))"
                                    kill_worker = true
                                end
                                if current_gpu_memory < (1-memory_margin)*initial_gpu_memory && gpu_hungry_worker == worker
                                    @warn "System running low on on GPU memory ($(Base.format_bytes(current_gpu_memory))/$(Base.format_bytes(initial_gpu_memory)) available); recycling worker $worker using $(Base.format_bytes(worker_memory_usage[worker].gpu))"
                                    kill_worker = true
                                end
                            catch err
                                config.status = "crashed"

                                bt = catch_backtrace()
                                log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                                @error "Unexpected exception on worker $worker: $log"
                                kill_worker = true
                            finally
                                delete!(worker_jobs, worker)
                                push!(results, (worker, i))

                                if kill_worker
                                    rmprocs(worker)
                                    delete!(worker_memory_usage, worker)
                                    worker = nothing
                                end

                                # Consider retrying failed configurations
                                if config.status in RETRY_STATUSSES
                                    push!(jobs, i)
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

                            push!(vals, ("problem", "$(problem) [$problem_idx/$(length(problems))]"))
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
                            cancel(p, "stopping after beating target time")
                            break
                        end
                        if (time() - sweep_start) > time_limits[problem_idx]
                            cancel(p, "stopping after hitting time limit")
                            break
                        end

                        sleep(5)
                    end

                    if isempty(jobs)
                        finish!(p)
                    end

                    empty!(jobs)
                end)
            end

            finish!(p)

            final_count = count(configs.status .!= "pending")
            println(" - final result: $(prettytime(best_time)) / $(prettytime(target_time)), after processing $(round(100*(final_count-initial_count)/total_count; digits=2))% ($(final_count-initial_count)/$(total_count-initial_count)) additional configurations")

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

        # Add coverage to best configs
        best_configs.coverage .= 0.0
        for problem in problems
            best_config = select_configs(best_configs, problem)
            configs = select_configs(all_configs, problem)
            nconfigs = count_configs(problem)
            nmeasured = count(configs.status .!= "pending")
            best_config.coverage .= nmeasured / nconfigs
        end

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
