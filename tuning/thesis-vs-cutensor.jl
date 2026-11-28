# Time the thesis's best configuration of every TCCG contraction, with today's code, next to
# cuTENSOR, in one session; verify each result against cuTENSOR. Prints a table and writes
# it as CSV. Times are the minimum over SAMPLES profiled runs (CUDA.@profile, kernel time only).
#
# usage (from the repository root, on the GPU machine):
#   julia +1.11 --project=tuning tuning/thesis-vs-cutensor.jl <thesis configs.arrow> [out.csv] [samples]
#
# Every contraction is compiled and measured in separate remote calls on one worker process,
# like in tune.jl: compiling a kernel after CUDA.@profile within the same call fails with
# CUDA.jl 5.11 (AssertionError: haskey(compiled, job.source)).

using Distributed

const CONFIGS = ARGS[1]
const OUT = get(ARGS, 2, "thesis-vs-cutensor.csv")
const SAMPLES = parse(Int, get(ARGS, 3, "10"))
const DIR = @__DIR__

addprocs(1; exeflags = "--project=$(Base.active_project())")

# load the packages first, in their own @everywhere: a macro like CUDA.@profile below is
# expanded before a `using CUDA` in the same block would have run
@everywhere begin
    using CUDA, GemmKernels, DataFrames, Adapt
    import Arrow
end
@everywhere include($(joinpath(DIR, "wmma-contraction.jl")))

@everywhere begin
    device_time(prof) = (events = DataFrame(prof.device); sum(events.stop .- events.start))

    const STATE = Ref{Any}(nothing)

    # allocate, compute the reference, compile both, warm up, verify ours
    function prepare_problem(problem, config)
        data = allocate_data(problem)
        initialize_data(problem, data...)
        reference = Array(calculate_reference(problem, data...))

        baseline_args = prepare_baseline(problem, data...)
        execute_baseline(problem, data...; baseline_args...)

        params = create_params(config)
        args = try
            prepare(problem, data...; params...)
        finally
            GemmKernels.Tensors.OVERRIDE_do_override = false
        end
        initialize_data(problem, data...; params...)
        valid = true
        for _ in 1:3
            result = adapt(Array, execute(problem, data...; args...))
            valid &= verify(problem, reference, result)
        end
        synchronize()
        STATE[] = (; problem, data, args, baseline_args)
        return valid
    end

    function measure_problem(samples)
        (; problem, data, args, baseline_args) = STATE[]
        ours = minimum(1:samples) do _
            device_time(CUDA.@profile concurrent=false execute(problem, data...; args...))
        end
        cutensor = minimum(1:samples) do _
            device_time(CUDA.@profile concurrent=false execute_baseline(problem, data...; baseline_args...))
        end
        return ours, cutensor
    end

    function release_problem()
        state = STATE[]
        state === nothing && return
        foreach(CUDA.unsafe_free!, state.data)
        STATE[] = nothing
        GC.gc(true); CUDA.reclaim()
        return
    end
end

w = workers()[1]
configs = DataFrame(Arrow.Table(CONFIGS))
problems = generate_problems()
rows = []
for (tc, problem) in enumerate(problems)
    sel = filter(:name => ==(problem.name), configs)
    if nrow(sel) != 1
        @warn "TC$tc ($(problem.name)): $(nrow(sel)) configurations in $CONFIGS; skipped"
        continue
    end
    config = NamedTuple(first(sel))
    row = try
        valid = remotecall_fetch(prepare_problem, w, problem, config)
        ours, cutensor = remotecall_fetch(measure_problem, w, SAMPLES)
        (; tc, name = problem.name, kernel = config.kernel_str,
           thesis_us = round(config.time * 1e6; digits = 1),
           ours_us = round(ours * 1e6; digits = 1), cutensor_us = round(cutensor * 1e6; digits = 1),
           relative_pct = round(100 * cutensor / ours; digits = 1), valid, error = "")
    catch err
        (; tc, name = problem.name, kernel = config.kernel_str,
           thesis_us = round(config.time * 1e6; digits = 1),
           ours_us = NaN, cutensor_us = NaN, relative_pct = NaN, valid = false,
           error = first(sprint(showerror, err), 200))
    finally
        remotecall_fetch(release_problem, w)
    end
    println(row)
    push!(rows, row)
end

df = DataFrame(rows)
println()
show(stdout, MIME("text/plain"), sort(df, :relative_pct); allrows = true, allcols = true)
println()
open(OUT, "w") do io
    println(io, join(names(df), ","))
    for r in eachrow(df)
        println(io, join((replace(string(x), "," => ";") for x in r), ","))
    end
end
println("\nwritten to $OUT")
