using CUDA, GemmKernels
using DataFrames
using DataStructures: counter
using Dates
using Distributed
using FileWatching.Pidfile
using Logging
using LoggingExtras
using ProgressMeter: Progress, ProgressUnknown, next!, update!
using Serialization
using Statistics
using Random

if myid() == 1
    using Plots
    pythonplot()
end

#######

const N_vals = 2 .^ (7:14)

const AB_type = Float16
const CD_type = Float32

const zero_c = true

#######

# Stop sampling when a configuration is really slow
const BENCH_MAX_SAMPLE_SECONDS = 1

const BENCH_MEMORY_USAGE = maximum(N_vals)^2 * 2 * sizeof(AB_type) +
                           maximum(N_vals)^2 * 2 * sizeof(CD_type) +
                           128*2^20 # there's always some overhead

const WORKER_MEMORY_USAGE = BENCH_MEMORY_USAGE + 2^30   # includes CUDA context size, etc

const PIDFILE = joinpath(@__DIR__, "tuning.pid")

#####

# Stop gathering samples for plot if we have spent this much time...
# 60 seconds/configuration * 32 configurations = 32 minutes for plot.
const PLOT_NUM_SECONDS = 60

# ... but have at least 100 samples.
const PLOT_MIN_NUM_SAMPLES = 100

# Group samples in batches of 10 samples each.
const PLOT_BATCH_SIZE = 10

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
        time=Float64[]
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
            :time => Inf,
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

    # allocate inputs
    reference_mul!, a, b, c, d = try
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
        run_gemm(cf, a, b, c, d)
    catch err
        bt = catch_backtrace()
        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)

        if isa(err, GemmKernels.ConfigError)
            @info "Skipping configuration $(repr_row(row))\n" * log
            return Inf, "unsupported_config_post_run"
        end

        if isa(err, CuError)
            @error "Configuration failed: $(repr_row(row))\n" * log
            rethrow()
        end

        @info "Skipping configuration: $(repr_row(row))\n" * log
        return Inf, "error"
    end

    # initialize inputs and calculate the reference
    c_h = mkpidlock(PIDFILE) do
        rand!(a)
        rand!(b)
        rand!(c)
        d .= 0

        reference_mul!(c, a, b)
        Array(c)
    end

    # settle down
    device_synchronize()
    GC.gc(true)

    # benchmark, but be quick about it (using `CUDA.@elapsed` instead of `CUDA.@profile`,
    # only taking the minimum time, etc)
    time = Inf
    d_h = mkpidlock(PIDFILE) do
        # make sure we're starting with an idle device
        wait_if_throttling()

        # keep benchmarking until the time isn't improving anymore
        while true
            new_time = CUDA.@elapsed run_gemm(cf, a, b, c, d)
            if new_time > time
                break
            end
            time = new_time

            # if this configuration is really slow, don't even bother running a second time
            if time > BENCH_MAX_SAMPLE_SECONDS
                break
            end
        end

        # fetch the results
        Array(d)
    end

    # verify the results
    if !verify(cf, c_h, d_h)
        # NOTE: we verify on the CPU because it involves memory allocations
        @warn "Configuration produced invalid result: $(repr_row(row))"
        return [time], "invalid_result"
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
        @info "Throttling detected. Sleeping for one second..."
        sleep(0.1)
        cer = NVML.clock_event_reasons(dev)
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

    for transpose_a = [false, true],
        transpose_b = [false, true],
        N = N_vals

        relevant_configs = configs[(@. (configs[!, "transpose_a"] == transpose_a) & (configs[!, "transpose_b"] == transpose_b) & (configs[!, "N"] == N)), :]
        _, best_config_index = findmin(relevant_configs[!, "time"], init=Inf)
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

        p = ProgressUnknown(desc="Benchmarking")

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

                        start_time = time()

                        push!(config_row[if run_baseline "baseline_nvml" else "gemmkernels_nvml" end], get_nvml_data())

                        if run_baseline
                            prof = CUDA.@profile concurrent=false run_baseline(cf, a, b, c, d)
                        else
                            prof = CUDA.@profile concurrent=false run_gemm(cf, a, b, c, d)
                        end

                        push!(config_row[if run_baseline "baseline_times" else "gemmkernels_times" end], sum(prof.device[!, "stop"] - prof.device[!, "start"]))

                        config_row["time_spent"] += (time() - start_time)
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
        worker_jobs = Dict()
        for p in workers()
            errormonitor(@async begin
                while length(pending) > 0
                    start_time = time()

                    i = popfirst!(pending)
                    config_row = configs[i, :]
                    worker_jobs[p] = (; start_time, i)
                    try
                        config_row.time, config_row.category =
                            remotecall_fetch(measure_config, p, NamedTuple(config_row))
                        end_time = time()

                        push!(results, (p, i, start_time, end_time))

                        if config_row.category in ["oom"]
                            # retry later
                            push!(pending, i)
                        end

                        # keep memory usage under control
                        remotecall(p) do
                            CUDA.reclaim()
                        end
                    catch err
                        config_row.category = "crashed"

                        bt = catch_backtrace()
                        log = sprint(Base.showerror, err) * sprint(Base.show_backtrace, bt)
                        @error "Error while measuring configurations: $log"

                        # recycle our worker
                        rmprocs(p; waitfor=30)
                        p = addworkers(1)[1]
                    finally
                        delete!(worker_jobs, p)
                    end
                end
            end)
        end

        # monitor progress
        errormonitor(@async begin
            while !isempty(pending) || !isempty(results)
                # process results
                if isready(results)
                    while isready(results)
                        worker, i, start_time, end_time = take!(results)

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
                num_finished = num_pending - length(pending)
                update!(p, num_finished; showvalues)

                sleep(5)
            end
            finish!(p)
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
    isfile(PIDFILE) && error("Another tuning process is already running. If this is not the case, please remove the file $(PIDFILE).")

    # Spawn workers
    cpu_memory = Sys.free_memory()
    gpu_memory = CUDA.available_memory()
    let
        nworkers = min(
            floor(Int, cpu_memory / WORKER_MEMORY_USAGE),
            floor(Int, gpu_memory / WORKER_MEMORY_USAGE),
            Sys.CPU_THREADS
        )
        addworkers(max(1, nworkers-1))
    end
    @info "Starting WMMA tuning script for device $(name(device())) using $(nworkers()) workers..."

    main()
end
