using CUDA, GemmKernels
using DataFrames
using DataStructures
using Dates
using Logging
using LoggingExtras
using Plots
using ProgressMeter
using Serialization
using Statistics
using StatsBase

pythonplot()

#######

const N_vals = 2 .^ (7:14)

# Stop sampling when normalised 95p CI is smaller than this...
const BENCH_NORM_CI_THRESHOLD = 0.01

# ... or we have exceeded the time limit...
const BENCH_MAX_NUM_SECONDS = 5

# ... but have at least 10 samples.
const BENCH_MIN_NUM_SAMPLES = 10

#####

# Stop gathering samples for plot if the "error bars" are smaller than this...
const PLOT_RATIO_MAX_UNCERTAINTY = 0.05

# ... or we have exceeded the time limit...
# In my experience, only N <= 2^9 requires more than a handful of samples.
# That's only 3*4 configurations, so a limit of say 600 seconds will take ~2 hours.
const PLOT_MAX_NUM_SECONDS = 600

# ... but have at least 10 samples.
const PLOT_MIN_NUM_SAMPLES = 10

const AB_type = Float16
const CD_type = Float32

const zero_c = true

const OP_M, OP_N, OP_K = 16, 16, 16

#######

# Reuse inputs across iterations.
c_ref = nothing
a = nothing
b = nothing
c = nothing
d = nothing
input_transpose_a = nothing
input_transpose_b = nothing
input_N = nothing

include("../configs/configs.jl")

# Write logging messages to file for persistence.
timestamp_logger(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) $(log.message)"))
end
FileLogger("tuning/tuning.log"; append=true) |> timestamp_logger |> (x -> MinLevelLogger(x, Logging.Info)) |> global_logger

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
            :kernel_str => kernel_str,
            :category => "unknown",
            :times => [],
        ))
    end

    all_configs
end

function get_config(row)
    transpose_a = row["transpose_a"]
    transpose_b = row["transpose_b"]
    M = N = K = row["N"]
    BLOCK_M = row["BLOCK_M"]
    BLOCK_N = row["BLOCK_N"]
    BLOCK_K = row["BLOCK_K"]
    WARPS_M = row["WARPS_M"]
    WARPS_N = row["WARPS_N"]
    kernel = kernel_string_to_function(row["kernel_str"])

    @get_wmma_config
end

function generate_inputs_if_needed(row)
    global input_transpose_a, input_transpose_b, input_N, c_ref, a, b, c, d

    cf = get_config(row)

    if (input_transpose_a, input_transpose_b, input_N) != (row.transpose_a, row.transpose_b, row.N)
        c_ref, a, b, c, d = generate_inputs(cf)
        input_transpose_a, input_transpose_b, input_N = row.transpose_a, row.transpose_b, row.N
    end
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
    cf = get_config(row)

    generate_inputs_if_needed(row)

    d .= 0

    try
        run_gemm(cf, a, b, c, d)
    catch err
        if isa(err, GemmKernels.ConfigError)
            @info "Skipping configuration $(NamedTuple(row))" * sprint(Base.showerror, err)
            return [Inf], "unsupported_config_post_run"
        end

        if isa(err, CuError)
            @error "Configuration failed: $(NamedTuple(row))" * sprint(Base.showerror, err)
            rethrow()
        end

        @info "Skipping configuration: $(NamedTuple(row))" * sprint(Base.showerror, err)
        return [Inf], "error"
    end

    if !verify(cf, c_ref, d)
        @warn "Configuration produced invalid result: $(NamedTuple(row))"

        expected = c_ref
        actual = d

        mad, index = findmax(abs.(expected - actual))

        @warn "Maximum absolute deviation is $(mad) at index $(index)."

        return [Inf], "invalid_result"
    end

    times = Float64[]

    # Use CUDA.@elapsed instead of CUDA.@profile, because the latter is slower.
    device_synchronize()
    GC.gc(true)

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

    return times, "success"
end

confidence_interval_95(times) = 1.58 * iqr(times) / sqrt(length(times))

function prettytime(times)
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

    (row["time_spent"] >= PLOT_MAX_NUM_SECONDS) && return true

    uncertainty, _, _ = get_uncertainty(gk, bl)

    uncertainty < PLOT_RATIO_MAX_UNCERTAINTY
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
        kernel_str=String[],
        category=String[],
        uncertainty=Float64[],
        time_spent=Float64[],
        gemmkernels_times=Vector{Any}[],
        baseline_times=Vector{Any}[]
    )

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
            :kernel_str => best_config["kernel_str"],
            :category => "todo",
            :uncertainty => Inf,
            :time_spent => 0.0,
            :gemmkernels_times => [],
            :baseline_times => [],
        ))
    end

    # We will reuse matrix inputs across iterations. This takes about 4 GB of GPU memory for e.g. all matrix sizes for NN.
    # Group runs of the same transposition together, so we don't have to keep 4 * 4 GB of inputs in memory.
    for transpose_a in [false, true],
        transpose_b in [false, true]

        input_dict = Dict()

        p = ProgressUnknown(desc="Benchmarking (highest uncertainty)", dt=1.0)

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

                @info "Profiling configuration $(NamedTuple(config_row))..."

                start_time = Dates.now()

                prof = CUDA.@profile run_gemm(cf, a, b, c, d)
                push!(config_row["gemmkernels_times"], sum(prof.device[!, "stop"] - prof.device[!, "start"]))

                prof = CUDA.@profile run_baseline(cf, a, b, c, d)
                push!(config_row["baseline_times"], sum(prof.device[!, "stop"] - prof.device[!, "start"]))

                config_row["time_spent"] += (Dates.now() - start_time) / Second(1)
                old_uncertainty = config_row["uncertainty"]
                config_row["uncertainty"], _, _ = get_uncertainty(config_row["gemmkernels_times"], config_row["baseline_times"])

                if got_enough_samples(config_row)
                    config_row["category"] = "done"
                end

                # Update progress bar.
                highest_uncertainty = best_configs[(@. (best_configs[!, "transpose_a"] == transpose_a) & (best_configs[!, "transpose_b"] == transpose_b)), :]
                highest_uncertainty = maximum(highest_uncertainty[!, "uncertainty"])
                next!(p; showvalues = [
                    (:transpose_a, transpose_a),
                    (:transpose_b, transpose_b),
                    (:N, config_row["N"]),
                    (:num_samples, length(config_row["gemmkernels_times"])),
                    (:uncertainty, "$(config_row["uncertainty"]) (Δ = $(config_row["uncertainty"] - old_uncertainty))"),
                    (:time_spent_in_config, config_row["time_spent"]),
                    (:highest_uncertainty, highest_uncertainty),
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
    title!("$AB_type x $AB_type = $CD_type ($(CUDA.name(CUDA.device())))")
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

    savefig(p, "tuning/plot.pdf")
end

function main()
    @info "Starting WMMA tuning script..."

    configs = nothing

    if !isfile("tuning/configs.bin")
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

        open("tuning/configs.bin", "w") do io
            serialize(io, configs)
        end
    end

    @info "Loading configurations from disk..."
    configs = open("tuning/configs.bin", "r") do io
        deserialize(io)
    end
    @info "Loaded $(size(configs, 1)) configurations."

    # (3) Measure performance of configurations.
    num_unknown = counter(configs[!, "category"])["unknown"]
    p = Progress(num_unknown; desc="Parameter sweep", dt=1.0, showspeed=true)

    @info "Need to perform parameter sweep over $(num_unknown) configurations."

    # Generate inputs for the first configuration. This is not strictly
    # speaking necessary, but doing this outside of the loop means that the
    # first iteration will not be excessively slow, which improves the "ETA"
    # estimate.
    first_unknown_config = findfirst(configs[!, "category"] .== "unknown")
    !isnothing(first_unknown_config) && generate_inputs_if_needed(configs[first_unknown_config, :])

    for config_row in eachrow(configs)
        start_time = Dates.now()

        if config_row.category != "unknown"
            continue
        end

        config_row.category = "crashed"

        # Save results in case the process crashes.
        open("tuning/configs.bin", "w") do io
            serialize(io, configs)
        end

        @info "Measuring configuration $(NamedTuple(config_row))..."

        times, category = measure_config(config_row)

        @info "Result for $(NamedTuple(config_row)): $(category) -- $(prettytime(times .* 1e9))"

        config_row.category = category
        config_row.times = times

        counter_dict_abs = Dict(counter(configs[!, "category"]))
        counter_dict_rel = Dict(k => "$(round(100 * v / sum(values(counter_dict_abs)); sigdigits=3))%" for (k, v) in counter_dict_abs)

        next!(p; showvalues=[
            (:N, config_row.N),
            (:transpose, get_label(config_row.transpose_a, config_row.transpose_b)),
            (:block_shape, (config_row.BLOCK_M, config_row.BLOCK_N, config_row.BLOCK_K)),
            (:num_warps, (config_row.WARPS_M, config_row.WARPS_N)),
            (:kernel, config_row.kernel_str),
            (:counters, counter_dict_abs),
            (:counters_relative, counter_dict_rel),
            (:last_result, "$(category) -- $(prettytime(times .* 1e9))"),
            (:last_iteration_time, Dates.now() - start_time)
        ])
    end

    # Save data for final iteration.
    open("tuning/configs.bin", "w") do io
        serialize(io, configs)
    end

    # And load again, for good measure.
    configs = open("tuning/configs.bin", "r") do io
        deserialize(io)
    end

    # (4) Select best configurations, and benchmark.
    if !isfile("tuning/best-configs.bin")
        @info "Benchmarking configurations for plot..."
        best_configs = benchmark_best_configs(configs)

        open("tuning/best-configs.bin", "w") do io
            serialize(io, best_configs)
        end
    end

    @info "Loading best configurations from disk..."
    best_configs = open("tuning/best-configs.bin", "r") do io
        deserialize(io)
    end

    # (5) Plotting results
    @info "Plotting results..."
    plot_results(best_configs)
end


isinteractive() || main()
