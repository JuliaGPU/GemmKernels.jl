using GemmKernels, CUDA
using Git: git
import GitHub
using Printf
using Statistics
using JSON

using StableRNGs

# XXX: How to choose good values here?
const NUM_SAMPLES = 1000

if haskey(ENV, "BUILDKITE_BRANCH")
    @info "Cloning previous benchmark results"
    github_token = get(ENV, "GITHUB_TOKEN", nothing)
    benchmark_results = mktempdir()
    if github_token === nothing
        run(`$(git()) clone -q https://github.com/JuliaGPU/GemmKernels.jl -b benchmark-results $benchmark_results`)
    else
        run(`$(git()) clone -q https://$github_token:x-oauth-basic@github.com/JuliaGPU/GemmKernels.jl -b benchmark-results $benchmark_results`)
    end
    run(`$(git()) -C $benchmark_results config --local user.name "JuliaGPU BenchmarkBot"`)
    run(`$(git()) -C $benchmark_results config --local user.email "nobody@juliagpu.org"`)
end

# load previous results.
function load_results()
    results_file = joinpath(@__DIR__, "reference-results.json")
    details_file = joinpath(@__DIR__, "reference-details.json")
    commit = "local"

    if haskey(ENV, "BUILDKITE_BRANCH")
        proc = open(`$(git()) -C $benchmark_results log --first-parent --pretty=format:"%H" origin/master`)
        while !eof(proc)
            commit = readline(proc)

            details_file = joinpath(benchmark_results, "details-$commit.json")
            results_file = joinpath(benchmark_results, "results-$commit.json")
            if isfile(results_file)
                break
            end
        end
        close(proc)
    end

    details = if isfile(details_file)
        json = JSON.parsefile(details_file)
    else
        nothing
    end

    if isfile(results_file)
        timings = JSON.parsefile(results_file)
        return (; commit, timings, details)
    end

    return nothing
end
previous_results = load_results()
if previous_results === nothing
    @error "No previous benchmark results found"
else
    @info "Found previous timings for commit $(previous_results.commit)"
end

# result rendering functions
function prettyflops(times, matmul_shape)
    # in ns
    t = minimum(times)

    # in GFlops
    flops = (2 * prod(matmul_shape)) / t
    rounded_flops = round(flops; sigdigits=4)

    return "$(rounded_flops) GFlops"
end

function prettytime(times)
    t = mean(times)
    sigma = std(times)
    min = minimum(times)
    max = maximum(times)

    # timescale
    scale, unit = if t < 1e3
        1, "ns"
    elseif t < 1e6
        1e3, "μs"
    elseif t < 1e9
        1e6, "ms"
    else
        1e9, "s"
    end
    cv = round(100 * sigma / abs(t); sigdigits=3)

    rounded_t = round(t / scale; sigdigits=3)
    rounded_cv = round(cv; sigdigits=3)
    rounded_min = round(min / scale; sigdigits=3)
    rounded_max = round(max / scale; sigdigits=3)

    return "$(rounded_t) $unit ± $(rounded_cv)% ($(rounded_min) … $(rounded_max) $unit)"
end

@info "Running benchmarks"
include("../configs/configs.jl")

results = Dict()
details = Dict()

for cf in get_configs()
    @info "Running benchmark $( cf.name )..."
    c_h, a, b, c, d = generate_inputs(cf)

    # warmup
    run_gemm(cf, a, b, c, d)

    # benchmark
    profile_results = CUDA.@profile begin
        for sample in 1:NUM_SAMPLES
            run_gemm(cf, a, b, c, d)
        end
    end

    # XXX: This works for now, since every GEMM is one kernel, but later on we may want to benchmark
    # operations consisting of multiple kernel launches...
    # XXX: Will this always work with mangling?
    matmul_results = filter(row -> contains(row.name, String(Symbol(cf.kernel))), profile_results.device)

    @assert size(matmul_results, 1) == NUM_SAMPLES

    # get info
    details[cf.name] = Dict(
        "registers" => matmul_results[1, "registers"],
        "dynamic_shared_mem" => matmul_results[1, "shared_mem"].dynamic,
        "static_shared_mem" => matmul_results[1, "shared_mem"].static,
        "local_mem" => matmul_results[1, "local_mem"].thread
    )

    times = 1e9 .* (matmul_results[!, "stop"] - matmul_results[!, "start"])

    @info "\t$(prettytime(times)) $(prettyflops(times, cf.config.matmul_shape))"
    results[cf.name] = Dict("times" => times)
end

function save_results(results_file, details_file, results, details)
    open(results_file, "w") do io
        JSON.print(io, results)
    end

    open(details_file, "w") do io
        JSON.print(io, details)
    end
end

# write results
if get(ENV, "BUILDKITE_BRANCH", nothing) == "master"
    commit = ENV["BUILDKITE_COMMIT"]
    results_file = joinpath(benchmark_results, "results-$commit.json")
    details_file = joinpath(benchmark_results, "details-$commit.json")
    save_results(results_file, details_file, results, details)

    # commit and push
    run(`$(git()) -C $benchmark_results add $results_file`)
    run(`$(git()) -C $benchmark_results add $details_file`)
    run(`$(git()) -C $benchmark_results commit -q -m "Results for $commit."`)
    run(`$(git()) -C $benchmark_results push -q`)
else
    results_file = joinpath(@__DIR__, "results.json")
    details_file = joinpath(@__DIR__, "details.json")
    save_results(results_file, details_file, results, details)
end

function markdown_escaped_code(str)
    ticks = eachmatch(r"`+", str)
    isempty(ticks) && return "`$str`"
    ticks = maximum(x -> length(x.match), ticks) + 1
    ticks = "`"^ticks
    return string(ticks, startswith(str, '`') ? " " : "", str, endswith(str, '`') ? " " : "", ticks)
end

idrepr(ids::Vector) = join(map(markdown_escaped_code, ids), " ")

@enum TrialJudgement begin
    improvement
    regression
    invariant
end

ratio(old, new) = new / old

function judge(old, new; tolerance=0.05)
    r = ratio(old, new)

    if isnan(r) || (r - tolerance) > 1.0
        return regression
    elseif (r + tolerance) < 1.0
        return improvement
    else
        return invariant
    end
end

const REGRESS_MARK = ":x:"
const IMPROVE_MARK = ":white_check_mark:"

resultmark(j::TrialJudgement) = j == regression ? REGRESS_MARK : (j == improvement ? IMPROVE_MARK : "")

function resultrow(k, judgement, old, new, old_details, new_details)
    old_times = old["times"]
    new_times = new["times"]

    str_old = prettytime(old_times)
    str_new = prettytime(new_times)

    if old_details !== nothing
        if old_details["registers"] != new_details["registers"]
            str_old *= "<br>$(old_details["registers"]) regs"
            str_new *= "<br>$(new_details["registers"]) regs"
        end
        if old_details["dynamic_shared_mem"] != new_details["dynamic_shared_mem"]
            str_old *= "<br>$(Base.format_bytes(old_details["dynamic_shared_mem"])) dynamic shmem"
            str_new *= "<br>$(Base.format_bytes(new_details["dynamic_shared_mem"])) dynamic shmem"
        end
        if old_details["static_shared_mem"] != new_details["static_shared_mem"]
            str_old *= "<br>$(Base.format_bytes(old_details["static_shared_mem"])) static shmem"
            str_new *= "<br>$(Base.format_bytes(new_details["static_shared_mem"])) static shmem"
        end
        if old_details["local_mem"] != new_details["local_mem"]
            str_old *= "<br>$(Base.format_bytes(old_details["local_mem"])) local mem"
            str_new *= "<br>$(Base.format_bytes(new_details["local_mem"])) local mem"
        end
    end

    r = ratio(minimum(old_times), minimum(new_times))
    r = @sprintf("%+.1f%%", 100*(r-1))
    mark = resultmark(judgement)
    return "| $(markdown_escaped_code(k)) | $(str_old) | $(str_new) | $(r) $(mark) |"
end

# compare against previous timings
if previous_results !== nothing
    @info "Comparing results"

    before = previous_results.timings
    after = results

    before_min = Dict(k => minimum(v["times"]) for (k, v) in before)
    after_min = Dict(k => minimum(v["times"]) for (k, v) in after)

    judgements = Dict(k => judge(before_min[k], v) for (k, v) in after_min)

    println("Improvements:")

    for (k, v) in judgements
        (v == improvement) && println(k)
    end

    println("Regressions:")

    for (k, v) in judgements
        (v == regression) && println(k)
    end

    # generate some text
    io = IOBuffer()
    commit = get(ENV, "BUILDKITE_COMMIT", "HEAD")
    println(io, "Benchmark results for commit $commit (comparing to $(previous_results.commit)):")

    filter!(judgements) do (k, v)
        time_changed = (v != invariant)
        details_changed = false

        if previous_results.details !== nothing
            previous_details = previous_results.details[k]
            current_details = details[k]

            details_changed =
                previous_details["registers"] != current_details["registers"] ||
                previous_details["dynamic_shared_mem"] != current_details["dynamic_shared_mem"] ||
                previous_details["static_shared_mem"] != current_details["static_shared_mem"] ||
                previous_details["local_mem"] != current_details["local_mem"]
        end

        time_changed || details_changed
    end

    if isempty(judgements)
        println(io, "No regressions or improvements detected.")
    else
        print(io, """

            | test | master | PR | Δmin |
            |------|--------|----|------|
            """)
        for (k, v) in judgements
            old = before[k]
            new = after[k]

            old_details = previous_results.details === nothing ? nothing : previous_results.details[k]
            new_details = details[k]

            println(io, resultrow(k, v, old, new, old_details, new_details))
        end
    end

    body = String(take!(io))
    println(body)

    # comment on PR
    if get(ENV, "BUILDKITE_PULL_REQUEST", "false") !== "false" && github_token !== nothing
        auth = GitHub.authenticate(github_token)
        repo = GitHub.repo("JuliaGPU/GemmKernels.jl"; auth)
        pr = parse(Int, ENV["BUILDKITE_PULL_REQUEST"])

        # find a previous comment to edit
        function find_previous_comment()
            kwargs = Dict(:auth => auth, :page_limit => 1, :params => (; per_page=100))
            while true
                comments, pages = GitHub.comments(repo, pr; auth)
                for comment in comments
                    if startswith(comment.body, "Benchmark results for")
                        return comment
                    end
                end
                if haskey(pages, "next")
                    delete!(kwargs, :params)
                    kwargs[:start_page] = pages["next"]
                else
                    return nothing
                end
            end
        end
        previous_comment = find_previous_comment()

        # submit
        if previous_comment === nothing
            GitHub.create_comment(repo, pr, :pr; auth, params=(; body=body))
        else
            GitHub.edit_comment(repo, previous_comment, :pr; auth, params=(; body=body))
        end
    end
end
