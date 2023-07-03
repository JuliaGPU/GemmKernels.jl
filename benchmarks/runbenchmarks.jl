using BenchmarkTools
using GemmKernels, CUDA
using Git: git
import GitHub
using Printf
using Statistics

using StableRNGs
rng = StableRNG(123)

# to find untuned benchmarks
BenchmarkTools.DEFAULT_PARAMETERS.evals = 0

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

# load timings
function load_results()
    proc = open(`$(git()) -C $benchmark_results log --first-parent --pretty=format:"%H" origin/master`)
    while !eof(proc)
        commit = readline(proc)
        results_file = joinpath(benchmark_results, "results-$commit.json")
        if isfile(results_file)
            timings = BenchmarkTools.load(results_file)[1]
            close(proc)
            return (; commit, timings)
        end
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

@info "Warming-up benchmarks"
warmup(SUITE; verbose=false)

# load params
params_file = joinpath(benchmark_results, "params.json")
params_updated = false
if !isfile(params_file)
    @info "Tuning benchmarks"
    tune!(SUITE)
    params_updated = true
else
    loadparams!(SUITE, BenchmarkTools.load(params_file)[1], :evals, :samples)

    # find untuned benchmarks for which we have the default evals==0
    function has_untuned(suite)
        for (ids, benchmark) in BenchmarkTools.leaves(suite)
            if params(benchmark).evals == 0
                return true
            end
        end
        return false
    end
    if has_untuned(SUITE)
        @info "Re-tuning benchmarks"
        tune!(SUITE)
        params_updated = true
    end
end
if params_updated
    BenchmarkTools.save(params_file, params(SUITE))
    if get(ENV, "BUILDKITE_BRANCH", nothing) == "master"
        BenchmarkTools.save(params_file, params(SUITE))

        # commit and push
        run(`$(git()) -C $benchmark_results add $params_file`)
        run(`$(git()) -C $benchmark_results commit -q -m "Re-tune benchmarks."`)
        run(`$(git()) -C $benchmark_results push -q`)
    end
end

@info "Running benchmarks"
timings = run(SUITE; verbose=true)
println(timings)

# write results
if get(ENV, "BUILDKITE_BRANCH", nothing) == "master"
    commit = ENV["BUILDKITE_COMMIT"]
    results_file = joinpath(benchmark_results, "results-$commit.json")
    BenchmarkTools.save(results_file, timings)

    # commit and push
    run(`$(git()) -C $benchmark_results add $results_file`)
    run(`$(git()) -C $benchmark_results commit -q -m "Results for $commit."`)
    run(`$(git()) -C $benchmark_results push -q`)
end

# result rendering functions
function markdown_escaped_code(str)
    ticks = eachmatch(r"`+", str)
    isempty(ticks) && return "`$str`"
    ticks = maximum(x -> length(x.match), ticks) + 1
    ticks = "`"^ticks
    return string(ticks, startswith(str, '`') ? " " : "", str, endswith(str, '`') ? " " : "", ticks)
end
idrepr(id::Vector) = sprint(idrepr, id)
function idrepr(io::IO, id::Vector)
    print(io, "[")
    first = true
    for i in id
        first ? (first = false) : print(io, ", ")
        show(io, i)
    end
    print(io, "]")
end
idrepr_md(id::Vector) = markdown_escaped_code(idrepr(id))
function resultrow(ids, j::BenchmarkTools.TrialJudgement,
                   old::BenchmarkTools.Trial, new::BenchmarkTools.Trial)
    t_old = @sprintf("%s ± %s", BenchmarkTools.prettytime(time(mean(old))),
                                BenchmarkTools.prettytime(time(std(old))))
    t_new = @sprintf("%s ± %s", BenchmarkTools.prettytime(time(mean(new))),
                                BenchmarkTools.prettytime(time(std(new))))
    ratio = @sprintf("%.1f%%", 100*(1-BenchmarkTools.time(BenchmarkTools.ratio(j))))
    mark = resultmark(BenchmarkTools.time(j))
    return "| $(idrepr_md(ids)) | $(t_old) | $(t_new) | $(ratio) $(mark) |"
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
        BenchmarkTools.isregression(time, j) || BenchmarkTools.isimprovement(time, j)
    end
    if isempty(judgements)
        println(io, "No regressions or improvements detected.")
    else
        print(io, """

            | ID | before | after | change |
            |----|--------|-------|--------|
            """)
        for (ids, j) in judgements
            old = rmskew(before[ids])
            new = rmskew(after[ids])
            println(io, resultrow(ids, j, old, new))
        end
    end
    body = String(take!(io))
    println(body)

    # comment on PR
    if github_token !== nothing && get(ENV, "BUILDKITE_PULL_REQUEST", "false") !== "false"
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
