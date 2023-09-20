using BenchmarkTools
using GemmKernels, CUDA
using Git: git
import GitHub
using Printf
using Statistics
using JSON

using StableRNGs


# we use setup/teardown phases to allocate/free GPU memory,
# so make sure to run a couple of evaluations to amortize
# the effects of using newly-allocated memory.
BenchmarkTools.DEFAULT_PARAMETERS.evals = 5

if haskey(ENV, "BUILDKITE_BRANCH")
    @info "Loading previous benchmark results"
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

# load timings
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
        # the named tuples got stored as dicts; convert them back to named tuples
        reconstruct_details(d::Dict) = (; Dict(Symbol(k)=>v for (k,v) in d)...)
        # the id arrays got stored as a string; parse them back
        reconstruct_ids(d::Dict, f=identity) = Dict(eval(Meta.parse(k))=>f(v) for (k,v) in json)
        reconstruct_ids(json, reconstruct_details)
    else
        nothing
    end

    if isfile(results_file)
        timings = BenchmarkTools.load(results_file)[1]
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

@info "Loading benchmarks"
SUITE = BenchmarkGroup()
include("blas.jl")

@info "Extracting execution details"
extract_details(group::BenchmarkGroup) = extract_details!([], [], group)
function extract_details!(results, parents, group::BenchmarkGroup)
    for (k, v) in group
        if isa(v, BenchmarkGroup)
            keys = Base.typed_vcat(Any, parents, k)
            extract_details!(results, keys, v)
        elseif endswith(k, " details")
            keys = Base.typed_vcat(Any, parents, replace(k, r" details$" => ""))
            push!(results, (keys, v))
        end
    end
    filter!(((k,v),) -> !endswith(k, " details"), group)
    return results
end
details = Dict(extract_details(SUITE))
display(details)

@info "Warming-up benchmarks"
warmup(SUITE; verbose=false)

@info "Running benchmarks"
timings = run(SUITE; verbose=true)
println(timings)

# write results
if get(ENV, "BUILDKITE_BRANCH", nothing) == "master"
    commit = ENV["BUILDKITE_COMMIT"]
    results_file = joinpath(benchmark_results, "results-$commit.json")
    BenchmarkTools.save(results_file, timings)
    details_file = joinpath(benchmark_results, "details-$commit.json")
    open(details_file, "w") do io
        JSON.print(io, details)
    end

    # commit and push
    run(`$(git()) -C $benchmark_results add $results_file`)
    run(`$(git()) -C $benchmark_results add $details_file`)
    run(`$(git()) -C $benchmark_results commit -q -m "Results for $commit."`)
    run(`$(git()) -C $benchmark_results push -q`)
else
    results_file = joinpath(@__DIR__, "results.json")
    BenchmarkTools.save(results_file, timings)

    details_file = joinpath(@__DIR__, "details.json")
    open(details_file, "w") do io
        JSON.print(io, details)
    end
end

# result rendering functions
function prettytime(t, std)
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
    t /= scale
    std /= scale

    # round according to the position of the first significant digit in the standard deviation
    rounded_std = round(std; sigdigits=2)
    pos = -floor(Int, log10(rounded_std))
    if pos <= 0
        rounded_std = round(Int, rounded_std)
        rounded_t = round(Int, t / 10^abs(pos)) * 10^abs(pos)
    else
        rounded_t = round(t; digits=pos)
    end

    return "$(rounded_t) ± $(rounded_std) $unit"
end
function markdown_escaped_code(str)
    ticks = eachmatch(r"`+", str)
    isempty(ticks) && return "`$str`"
    ticks = maximum(x -> length(x.match), ticks) + 1
    ticks = "`"^ticks
    return string(ticks, startswith(str, '`') ? " " : "", str, endswith(str, '`') ? " " : "", ticks)
end
idrepr(ids::Vector) = join(map(markdown_escaped_code, ids), " ")
function resultrow(ids, j::BenchmarkTools.TrialJudgement,
                   old::BenchmarkTools.Trial, new::BenchmarkTools.Trial,
                   old_details, new_details)
    str_old = prettytime(time(mean(old)), time(std(old)))
    str_new = prettytime(time(mean(new)), time(std(new)))
    if old_details !== nothing
        if old_details.registers != new_details.registers
            str_old *= "<br>$(old_details.registers) regs"
            str_new *= "<br>$(new_details.registers) regs"
        end
        if old_details.dynamic_shared_mem != new_details.dynamic_shared_mem
            str_old *= "<br>$(Base.format_bytes(old_details.dynamic_shared_mem)) dynamic shmem"
            str_new *= "<br>$(Base.format_bytes(new_details.dynamic_shared_mem)) dynamic shmem"
        end
        if old_details.static_shared_mem != new_details.static_shared_mem
            str_old *= "<br>$(Base.format_bytes(old_details.static_shared_mem)) static shmem"
            str_new *= "<br>$(Base.format_bytes(new_details.static_shared_mem)) static shmem"
        end
        if old_details.local_mem != new_details.local_mem
            str_old *= "<br>$(Base.format_bytes(old_details.local_mem)) local mem"
            str_new *= "<br>$(Base.format_bytes(new_details.local_mem)) local mem"
        end
        if old_details.const_mem != new_details.const_mem
            str_old *= "<br>$(Base.format_bytes(old_details.const_mem)) const mem"
            str_new *= "<br>$(Base.format_bytes(new_details.const_mem)) const mem"
        end
    end
    ratio = @sprintf("%.1f%%", 100*(1-time(BenchmarkTools.ratio(j))))
    mark = resultmark(time(j))
    return "| $(idrepr(ids)) | $(str_old) | $(str_new) | $(ratio) $(mark) |"
end
const REGRESS_MARK = ":x:"
const IMPROVE_MARK = ":white_check_mark:"
resultmark(sym::Symbol) = sym == :regression ? REGRESS_MARK : (sym == :improvement ? IMPROVE_MARK : "")

# compare against previous timings
if previous_results !== nothing
    @info "Comparing results"

    before = Dict(BenchmarkTools.leaves(previous_results.timings))
    after = Dict(BenchmarkTools.leaves(timings))

    comparison = judge(minimum(timings), minimum(previous_results.timings))

    println("Improvements:")
    println(improvements(comparison))

    println("Regressions:")
    println(regressions(comparison))

    # generate some text
    io = IOBuffer()
    commit = get(ENV, "BUILDKITE_COMMIT", "HEAD")
    println(io, "Benchmark results for commit $commit (comparing to $(previous_results.commit)):")
    judgements = BenchmarkTools.leaves(comparison)
    judgements = judgements[sortperm(map(string∘first, judgements))]
    filter!(judgements) do (ids, j)
        time_changed = BenchmarkTools.isregression(time, j) ||
                       BenchmarkTools.isimprovement(time, j)
        if previous_results.details !== nothing
            previous_details = previous_results.details[ids]
            time_changed ||
                previous_details.registers != details[ids].registers ||
                previous_details.dynamic_shared_mem != details[ids].dynamic_shared_mem ||
                previous_details.static_shared_mem != details[ids].static_shared_mem ||
                previous_details.local_mem != details[ids].local_mem ||
                previous_details.const_mem != details[ids].const_mem
        else
            time_changed
        end
    end
    if isempty(judgements)
        println(io, "No regressions or improvements detected.")
    else
        print(io, """

            | test | master | PR | Δmin |
            |------|--------|----|------|
            """)
        for (ids, j) in judgements
            old = rmskew(before[ids])
            new = rmskew(after[ids])
            old_details = previous_results.details === nothing ? nothing : previous_results.details[ids]
            new_details = details[ids]
            println(io, resultrow(ids, j, old, new, old_details, new_details))
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
