using CUDA, GemmKernels
using DataFrames
using DataStructures: counter
using Dates
using Distributed
using Logging
using LoggingExtras
using ProgressMeter: Progress, ProgressUnknown, @showprogress,
                     next!, update!, finish!, cancel
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
# - create_configs(): create an empty database for configurations
# - config_iterator(problem): a generator that yields configurations for a specific problem.
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

# Whether we stop after beating the baseline, or continue until we've tested every config.
const EXHAUSTIVE = false

# The time limit for the entire sweep, in seconds.
# This will be used to determine a per-problem time limit.
const SWEEP_TIME_LIMIT = 24*3600

# The time limit for each single step (preparation, measurement, verification), in seconds.
const CONFIG_TIME_LIMIT = 60

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

function addworkers(X; gpu_mem_target=nothing, cpu_mem_target=nothing, cpu_mem_limit=nothing)
    env = [
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1"
    ]
    if gpu_mem_target !== nothing
        push!(env, "JULIA_CUDA_HARD_MEMORY_LIMIT" => string(gpu_mem_target))
    end

    exeflags = [
        "--project=$(Base.active_project())"
    ]
    if cpu_mem_target !== nothing
        push!(exeflags, "--heap-size-hint=$cpu_mem_target")
    end

    exename = first(Base.julia_cmd().exec)
    if cpu_mem_target !== nothing || cpu_mem_limit !== nothing
        # if limits are set, disable swap to prevent excessive swapping
        # (since we expect to be running close to the memory limit)
        runner = `systemd-run --quiet --scope --user -p MemorySwapMax=0`
        # XXX: MemoryHigh causes throttling, which we never want.
        #if cpu_mem_target !== nothing
        #    runner = `$runner -p MemoryHigh=$cpu_mem_target`
        #end
        if cpu_mem_limit !== nothing
            runner = `$runner -p MemoryMax=$cpu_mem_limit`
        end
        exename = `$runner $exename`
    end

    procs = addprocs(X; exename, exeflags, env)
    @everywhere procs include($(joinpath(@__DIR__, "tune.jl")))
    procs
end

############################################################################################

const prepared_state = Ref{Any}(nothing)
cached_data = nothing

# allocate data and compile kernels
function prepare_config(problem, config, fake=false)
    @info "Processing configuration $(repr_row(config))..."
    state = prepared_state[]

    # (re)allocate data
    data = if state === nothing
        try
            # this is the only place where we allocate device memory.
            # other allocation failures will be reported as crashes.
            allocate_data(problem; fake)
        catch err
            bt = catch_backtrace()
            log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

            if isa(err, OutOfGPUMemoryError)
                @info "Not enough memory for configuration $(repr_row(config))\n" * log
                return "oom"
            else
                rethrow()
            end
        end
    else
        state.data
    end
    prepared_state[] = (; data)

    # compile and warm-up
    params = create_params(config)
    try
        args = prepare(problem, data...; params...)

        prepared_state[] = (; data, params, args)
    catch err
        bt = catch_backtrace()
        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

        # determine the cause of the error
        if isa(err, GemmKernels.ConfigError)
            @info "Skipping unsupported configuration $(repr_row(config))\n" * log
            return "config_error"
        end
        if isa(err, CUDA.InvalidIRError)
            @info "Failed to compile $(repr_row(config))\n" * log
            return "compilation_error"
        end
        if isa(err, OutOfGPUMemoryError)
            # XXX: this shouldn't happen here as we already allocated all memory,
            #      however it can occur during kernel launches or other CUDA calls
            #      when very close to the memory limit.
            @info "Not enough memory for configuration $(repr_row(config))\n" * log
            return "oom"
        end

        @info "Unknown error processing $(repr_row(config))\n" * log
        return "unknown_error"
    end

    return "success"
end

# take time measurements
function measure_config(problem, config, max_time)
    state = prepared_state[]
    @assert state !== nothing
    (; data, args, params) = state

    # warm-up
    warmup = @elapsed begin
        execute(problem, data...; args...)
        synchronize()
    end

    # first measurement: check if the configuration is even worth measuring
    measurements = Float64[]
    measuring = @elapsed begin
        cur_time = CUDA.@elapsed blocking=true execute(problem, data...; args...)
    end
    push!(measurements, cur_time)
    if cur_time > max_time
        return measurements, nothing, (; measuring)
    end

    # make sure we're starting with an idle device
    settling = @elapsed wait_if_throttling()

    # second measurement: fetch results to verify
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

    return measurements, result, (; warmup, initializing, settling, measuring, copying)
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
    p = Progress(nbenchmarks * BENCHMARK_SAMPLES; desc="Benchmarking:", showspeed=true)
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

function remotecall_until(args...; timeout::Real=CONFIG_TIME_LIMIT)
    output = remotecall(args...)
    # XXX: output is a Future, which doesn't support close,
    #      so we need to throw an exception to end `fetch`
    #timer = Timer(timeout) do t
    #    isready(output) || close(output)
    #end
    #try
    #    return fetch(output)
    #finally
    #    close(timer)
    #end
    waiter = @async try
        fetch(output)
    catch e
        isa(e, InterruptException) && return nothing
        rethrow()
    end
    timer = Timer(timeout) do t
        isready(output) || Base.throwto(waiter, InterruptException())
    end
    try
        return fetch(waiter)
    finally
        close(timer)
    end
end

function main()
    #
    # Phase 1: Prepare
    #

    problems = generate_problems()

    # Gather baseline performance and results
    baseline_performances = Dict()
    baseline_data = Dict()

    reference_result_dir = get_scratch!(GemmKernels, "reference_results")
    reference_results = Dict()
    let
        worker = addworkers(1)[1]
        @showprogress desc="Measuring baselines:" for problem in problems
            path = tempname(reference_result_dir)
            baseline_performances[problem] = remotecall_fetch(worker) do
                data = allocate_data(problem)
                initialize_data(problem, data...)

                # calculate
                result = Array(calculate_reference(problem, data...))
                serialize(path, result)

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

                prof = CUDA.@profile concurrent=false execute_baseline(problem, data...; args...)
                baseline_data["$problem"] = Dict(
                    "times" => measurements,
                    "kernels" => prof.device.name,
                )

                for arr in data
                    CUDA.unsafe_free!(arr)
                end

                minimum(measurements)
            end
            @assert isfile(path)
            reference_results[problem] = path
        end
        rmprocs(worker)
    end

    serialize(joinpath(@__DIR__, "baseline-data.bin"), baseline_data)

    # Get the device we're working on
    cuda_dev = device()
    mig = uuid(cuda_dev) != parent_uuid(cuda_dev)
    nvml_dev = NVML.Device(uuid(cuda_dev); mig)

    # Load previous results from disk
    config_path = joinpath(@__DIR__, "configs.bin")
    all_configs = create_configs()
    if isfile(config_path)
        @info "Loading configurations from disk..."
        try
            all_configs = deserialize(config_path)

            @info "Loaded $(size(all_configs, 1)) configurations."
        catch err
            @error "Error while loading configurations from disk: $(sprint(Base.showerror, err)))"
            mv(config_path, "$(config_path).broken")
        end
    else
        all_configs.status = String[]
        all_configs.time = Float64[]
    end
    function checkpoint()
        temp_path = "$(config_path).$(getpid)"
        serialize(temp_path, all_configs)
        mv(temp_path, config_path; force=true)
    end

    # Give configurations that ran into an unknown error another change
    # (note that we don't retry them within a run)
    deleteat!(all_configs, in.(all_configs.status,
                               Ref([["unknown_error", "pending", "promising"];
                                    RETRY_STATUSSES])))

    # Find the best times so far
    best_times = Dict()
    for problem in problems
        configs = select_configs(all_configs, problem)
        best_times[problem] = minimum(filter(x->x.status == "success", configs).time; init=Inf)
    end

    # Process problems in order of current ratio, tackling the worst ones first
    perf_ratio(problem) = baseline_performances[problem] / best_times[problem]
    sort!(problems, by=perf_ratio)

    # Determine per-problem time limits
    weights = Dict()
    for problem in problems
        # start with the number of problems, and bias it towards problems with a bad ratio
        bias(x) = 100 * exp(-5 * x)
        weights[problem] = length(config_iterator(problem)) * bias(perf_ratio(problem))
    end
    total_weight = sum(values(weights))
    time_limits = Dict()
    for problem in problems
        time_limits[problem] = SWEEP_TIME_LIMIT * weights[problem] / total_weight
    end

    # Get rid of the CUDA context on the master process
    # NOTE: don't use CUDA API's after this point!
    CUDA.device_reset!()


    #
    # Phase 2: Sweep parameters
    #

    for (problem_idx, problem) in enumerate(problems)
        println("\nProcessing $problem [$problem_idx/$(length(problems))]...")
        configs = select_configs(all_configs, problem)
        sweep_start = time()

        # See if there's anything we need to do
        best_time = minimum(filter(x->x.status == "success", configs).time; init=Inf)
        target_time = baseline_performances[problem]
        println(" - target time: $(prettytime(target_time))")
        if best_time != Inf
            println(" - best time so far: $(prettytime(best_time))")
            if !EXHAUSTIVE && best_time < target_time
                println("   fast enough already")
                continue
            end
        end

        initial_count = size(configs, 1)
        total_count = length(config_iterator(problem))
        println(" - have processed $(round(100*initial_count/total_count; digits=2))% ($initial_count/$total_count) of configurations already")

        njobs = total_count - initial_count
        if njobs > 0
            # Determine memory usage
            cpu_memory_available = Sys.free_memory()
            gpu_memory_available = NVML.memory_info(nvml_dev).free
            memory_margin = 0.9

            # Determine memory requirement
            println(" - problem memory requirement: $(Base.format_bytes(sizeof(problem)))")

            # Spawn measurement workers
            add_measurement_worker = let
                gpu_mem_target = sizeof(problem) + 32*2^20      # allow minimal unaccounted allocations
                gpu_mem_limit = gpu_mem_target + 1000*2^20      # size of (reasonable) CUDA context
                cpu_mem_target = 3*sizeof(problem)              # 2 for the data, 1 for the comparison
                cpu_mem_limit = 3*sizeof(problem) + 1000*2^20   # headroom
                cpu_memory_available -= cpu_mem_limit
                gpu_memory_available -= gpu_mem_limit

                (X) -> addworkers(X; gpu_mem_target, cpu_mem_target, cpu_mem_limit)
            end
            measurement_workers = add_measurement_worker(1)

            # Spawn compilation workers
            max_compile_workers, add_compile_worker = let
                cpu_mem_target = 2000*2^20  # reasonable size of the heap
                cpu_mem_limit = 2500*2^20   # compilation headroom
                gpu_mem_limit = 500*2^20    # size of (minimal) CUDA context

                max_workers_cpu_mem = floor(Int, cpu_memory_available * memory_margin / cpu_mem_limit)
                max_workers_gpu_mem = floor(Int, gpu_memory_available * memory_margin / gpu_mem_limit)
                max_workers_cpu_threads = Sys.CPU_THREADS
                max_workers_njobs = njobs+1

                max_workers = min(
                    max_workers_cpu_mem,
                    max_workers_gpu_mem,
                    max_workers_cpu_threads,
                    max_workers_njobs
                )

                println("Determining max # of compilation workers:")
                println("Limit determined by CPU memory: $max_workers_cpu_mem")
                println("Limit determined by GPU memory: $max_workers_gpu_mem")
                println("Limit determined by #CPU threads: $max_workers_cpu_threads")
                println("Limit determined by #jobs: $max_workers_njobs")

                println("--> Overall limit: $max_workers")

                max_workers, (X) -> addworkers(X; cpu_mem_target, cpu_mem_limit)
            end
            compile_workers = add_compile_worker(max_compile_workers)

            # Functionality to quickly detect already seen configurations, by hashing
            # all columns except the status/time ones added by the tuning script.
            problem_cols = Symbol.(filter(!in(["status", "time"]), names(all_configs)))
            hash_config(config) = hash(((getproperty(config, col) for col in problem_cols)...,))
            seen_configs = Set(hash_config.(eachrow(configs)))

            # Process jobs!
            println(" - time limit: $(prettytime(time_limits[problem]))")
            reference_result = deserialize(reference_results[problem])
            initial_category_counters = Dict(counter(configs[!, "status"]))
            p = ProgressUnknown(desc="Measuring configurations:", showspeed=true)
            measurement_times = Dict{Symbol, Float64}()
            results = Channel(Inf)
            initial_jobs = Channel(100)
            promising_jobs = Channel(2 * max_compile_workers)
            @sync begin
                # Job queue
                job_submitter = errormonitor(@async begin
                    for config in config_iterator(problem)
                        try
                            # only process new configurations
                            if !in(hash_config(config), seen_configs)
                                push!(all_configs, (config..., "pending", Inf))
                                put!(initial_jobs, size(all_configs, 1))
                            end
                        catch err
                            isa(err, EOFError) || rethrow()
                            break
                        end
                    end
                end)

                # Compilation tasks
                for worker in compile_workers
                    errormonitor(@async begin
                        while isopen(initial_jobs)
                            # ensure we still have a worker
                            if worker === nothing
                                try
                                    worker = add_compile_worker(1)[1]
                                catch err
                                    # give up for this problem
                                    @error "Failed to add compilation worker: $(sprint(Base.showerror, err))"
                                    break
                                end
                            end

                            # get a job
                            i = try
                                take!(initial_jobs)
                            catch err
                                isa(err, EOFError) || rethrow()
                                break
                            end
                            config = all_configs[i, :]
                            kill_worker = false

                            status = try
                                # prepare
                                status = @something(
                                    remotecall_until(prepare_config, worker, problem, NamedTuple(config), true),
                                    error("Time-out preparing configuration")
                                )

                                if status != "success"
                                    config.status = status
                                    continue
                                end

                                config.status = "promising"
                            catch err
                                config.status = "crashed"

                                bt = catch_backtrace()
                                log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                                @error "Unexpected exception on worker $worker: $log"
                                kill_worker = true
                            finally
                                if config.status == "promising"
                                    # submit for further processing
                                    try
                                        put!(promising_jobs, i)
                                    catch err
                                        isa(err, EOFError) || rethrow()
                                        break
                                    end
                                else
                                    push!(results, (worker, i))
                                end

                                if kill_worker
                                    rmprocs(worker)
                                    worker = nothing
                                end
                            end

                            # submit for further processing
                            if config.status == "promising"
                                try
                                    put!(promising_jobs, i)
                                catch err
                                    isa(err, EOFError) || rethrow()
                                    break
                                end
                            end
                        end
                    end)
                end

                # Measurement tasks
                for worker in measurement_workers
                    errormonitor(@async begin
                        while isopen(promising_jobs)
                            # ensure we still have a worker
                            if worker === nothing
                                try
                                    worker = add_measurement_worker(1)[1]
                                catch err
                                    # give up for this problem
                                    @error "Failed to add measurement worker: $(sprint(Base.showerror, err))"
                                    break
                                end
                            end

                            # get a job
                            waiting = @elapsed begin
                                i = try
                                    take!(promising_jobs)
                                catch err
                                    isa(err, EOFError) || rethrow()
                                    break
                                end
                            end
                            config = all_configs[i, :]
                            kill_worker = false

                            try
                                # prepare
                                preparing = @elapsed begin
                                    status = @something(
                                        remotecall_until(prepare_config, worker, problem, NamedTuple(config)),
                                        error("Time-out preparing configuration")
                                    )
                                end
                                if status != "success"
                                    config.status = status
                                    continue
                                end

                                # measure
                                max_time = 3 * target_time
                                measurements, result, times = @something(
                                    remotecall_until(measure_config, worker, problem, NamedTuple(config), max_time),
                                    error("Time-out measuring configuration")
                                )
                                config.time = minimum(measurements)
                                times = Dict(pairs(times)...,
                                                   :waiting => waiting,
                                                   :preparing => preparing)
                                for k in keys(times)
                                    v = times[k]
                                    measurement_times[k] = get(measurement_times, k, 0.0) + v
                                end

                                # bail out if this is a slow config
                                if minimum(measurements) > max_time
                                    config.status = "slow"
                                    continue
                                end

                                # verify results
                                verified = @something(
                                    remotecall_until(verify, worker, problem, reference_result, result),
                                    error("Time-out verifying results")
                                )
                                if !verified
                                    @warn "Configuration produced invalid result: $(repr_row(config))"
                                    config.status = "invalid_result"
                                    continue
                                end

                                config.status = "success"
                            catch err
                                config.status = "crashed"

                                bt = catch_backtrace()
                                log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                                @error "Unexpected exception on worker $worker: $log"
                                kill_worker = true
                            finally
                                push!(results, (worker, i))

                                if kill_worker
                                    rmprocs(worker)
                                    worker = nothing
                                end
                            end
                        end
                    end)
                end

                # Result processing task
                errormonitor(@async begin
                    t_checkpoint = 0
                    nfinished = 0
                    while true
                        # process results
                        if isready(results)
                            while isready(results)
                                worker, i = take!(results)
                                config = all_configs[i, :]
                                nfinished += 1

                                # Update configuration
                                if config.status == "success"
                                    best_time = min(best_time, config.time)
                                end
                                @info "Result from worker $worker for $(repr_row(config)): $(config.status) -- $(prettytime(config.time))"
                            end

                            # save results every minute
                            if time() - t_checkpoint > 60
                                checkpoint()
                                t_checkpoint = time()
                            end
                        end

                        # update the progress bar
                        function showvalues()
                            vals = []

                            push!(vals, ("problem", "$(problem) [$problem_idx/$(length(problems))]"))
                            total_workers = length(compile_workers) + length(measurement_workers)
                            push!(vals, ("workers", "$(length(workers())) / $(total_workers)"))

                            push!(vals, ("", ""))

                            # configuration times
                            push!(vals, ("target time", prettytime(target_time)))
                            if !isinf(best_time)
                                push!(vals, ("best time", prettytime(best_time)))
                            end

                            push!(vals, ("", ""))

                            # how much time we spent measuring configurations
                            if !isempty(measurement_times)
                                for (k, v) in measurement_times
                                    push!(vals, (k, prettytime(v)))
                                end

                                push!(vals, ("", ""))
                            end

                            # job state
                            configs = select_configs(all_configs, problem)
                            category_counters = Dict(counter(configs[!, "status"]))
                            for k in sort(collect(keys(category_counters));
                                          by=k->category_counters[k])
                                initial = get(initial_category_counters, k, 0)
                                current = category_counters[k]
                                relative = round(100 * current / sum(values(category_counters)); sigdigits=3)
                                push!(vals, (k, "$(current-initial) + $(initial) ($(relative)%)"))
                            end

                            push!(vals, ("", ""))

                            # gpu stats
                            push!(vals, ("power usage", "$(NVML.power_usage(nvml_dev)) W"))
                            push!(vals, ("temperature", "$(NVML.temperature(nvml_dev)) °C"))
                            meminfo = NVML.memory_info(nvml_dev)
                            push!(vals, ("memory usage", "$(Base.format_bytes(meminfo.used)) / $(Base.format_bytes(meminfo.total))"))
                            utilization = NVML.utilization_rates(nvml_dev)
                            push!(vals, ("utilization", "$(round(100*utilization.compute; sigdigits=3))% compute, $(round(100*utilization.memory; sigdigits=3))% memory"))

                            vals
                        end
                        update!(p, nfinished; showvalues, valuecolor=:normal)

                        # see if we need to stop
                        if istaskdone(job_submitter)
                            finish!(p)
                            println(" - tested all configurations")
                            break
                        end
                        if !EXHAUSTIVE && best_time < target_time
                            finish!(p)
                            println(" - found a configuration that beats the baseline")
                            break
                        end
                        if (time() - sweep_start) > time_limits[problem]
                            cancel(p)
                            println(" - reached time limit")
                            break
                        end

                        sleep(5)
                    end

                    # Clean-up
                    close(initial_jobs, EOFError())
                    close(promising_jobs, EOFError())
                    if workers() != [myid()]
                        for worker in workers()
                            try
                                rmprocs(worker; waitfor=30)
                            catch err
                                @error "Failed to stop worker $worker" exception=(err, catch_backtrace())
                            end
                        end
                    end

                    # Printn a summary
                    final_count = size(configs, 1)
                    println(" - final result: $(prettytime(best_time)) / $(prettytime(target_time)), after processing $(round(100*(final_count-initial_count)/total_count; digits=2))% ($(final_count-initial_count)/$(total_count-initial_count)) additional configurations")
                end)
            end
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
        best_configs = let
            worker = addworkers(1)[1]
            try
                remotecall_fetch(worker) do
                    benchmark_configs(all_configs)
                end
            finally
                rmprocs(worker)
            end
        end

        # Add coverage to best configs
        best_configs.coverage .= 0.0
        for problem in problems
            configs = select_configs(all_configs, problem)
            nconfigs = length(config_iterator(problem))
            nmeasured = size(configs, 1)

            best_config = select_configs(best_configs, problem)
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
    main()
end
