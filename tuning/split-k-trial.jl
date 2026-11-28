# Time split-K on contractions with a long K: the thesis's best configuration of each, with the
# kernels that support split-K (singlestage and the async kernels), for several numbers of
# parts, next to cuTENSOR. Every result is verified. Prints a table and writes it as CSV.
#
# usage (from the repository root, on the GPU machine, with nothing else on the GPU):
#   julia +1.11 --project=tuning tuning/split-k-trial.jl <thesis configs.arrow> [TCs, default 13,14] [out.csv]
#
# Like tune.jl, compiling and profiling happen in separate remote calls on one worker:
# compiling after CUDA.@profile within the same call fails with CUDA.jl 5.11.

using Distributed

const CONFIGS = ARGS[1]
const TCS = parse.(Int, split(get(ARGS, 2, "13,14"), ','))
const OUT = get(ARGS, 3, "split-k-trial.csv")
const SAMPLES = 10
const SPLITS = [1, 2, 4, 8, 16, 32]
const DIR = @__DIR__

addprocs(1; exeflags = "--project=$(Base.active_project())")

@everywhere begin
    using CUDA, GemmKernels, DataFrames, Adapt
    import Arrow
end
@everywhere include($(joinpath(DIR, "wmma-contraction.jl")))

@everywhere begin
    device_time(prof) = (events = DataFrame(prof.device); sum(events.stop .- events.start))
    const STATE = Ref{Any}(nothing)
    const VARIANT = Ref{Any}(nothing)

    function setup_problem(problem)
        data = allocate_data(problem)
        initialize_data(problem, data...)
        reference = Array(calculate_reference(problem, data...))
        baseline_args = prepare_baseline(problem, data...)
        execute_baseline(problem, data...; baseline_args...)
        synchronize()
        STATE[] = (; problem, data, reference, baseline_args)
        return
    end

    # compile and verify one variant; returns nothing if it is rejected (ConfigError)
    function prepare_variant(config, split_k)
        (; problem, data, reference) = STATE[]
        params = create_params(config)
        args = try
            prepare(problem, data...; params..., SPLIT_K = split_k)
        catch err
            err isa GemmKernels.ConfigError || rethrow()
            return sprint(showerror, err)
        finally
            GemmKernels.Tensors.OVERRIDE_do_override = false
        end
        initialize_data(problem, data...; params...)
        valid = true
        for _ in 1:3
            valid &= verify(problem, reference, adapt(Array, execute(problem, data...; args...)))
        end
        synchronize()
        VARIANT[] = args
        return valid
    end

    function measure_variant(samples)
        (; problem, data) = STATE[]
        args = VARIANT[]
        return minimum(1:samples) do _
            device_time(CUDA.@profile concurrent=false execute(problem, data...; args...))
        end
    end

    function measure_baseline(samples)
        (; problem, data, baseline_args) = STATE[]
        return minimum(1:samples) do _
            device_time(CUDA.@profile concurrent=false execute_baseline(problem, data...; baseline_args...))
        end
    end

    function release_problem()
        STATE[] === nothing && return
        foreach(CUDA.unsafe_free!, STATE[].data)
        STATE[] = nothing; VARIANT[] = nothing
        GC.gc(true); CUDA.reclaim()
        return
    end
end

w = workers()[1]
configs = DataFrame(Arrow.Table(CONFIGS))
problems = generate_problems()
rows = []
for tc in TCS
    problem = problems[tc]
    sel = filter(:name => ==(problem.name), configs)
    nrow(sel) == 1 || (@warn "TC$tc: $(nrow(sel)) configurations in $CONFIGS; skipped"; continue)
    thesis = NamedTuple(first(sel))
    println("== TC$tc $(problem.name): thesis configuration $(repr_row(thesis)), $(round(thesis.time * 1e6; digits=1)) μs")

    remotecall_fetch(setup_problem, w, problem)
    cutensor = remotecall_fetch(measure_baseline, w, SAMPLES)
    println("   cuTENSOR: $(round(cutensor * 1e6; digits=1)) μs")

    kernels = unique(["singlestage", "pipelined_async_2", "pipelined_async_3", "pipelined_async_4"])
    for kernel_str in kernels, split_k in SPLITS
        config = merge(thesis, (; kernel_str))
        outcome = remotecall_fetch(prepare_variant, w, config, split_k)
        if outcome isa String
            println("   $kernel_str, split_k = $split_k: rejected ($(first(outcome, 100)))")
            continue
        end
        ours = remotecall_fetch(measure_variant, w, SAMPLES)
        row = (; tc, kernel = kernel_str, split_k, ours_us = round(ours * 1e6; digits = 1),
                 cutensor_us = round(cutensor * 1e6; digits = 1),
                 relative_pct = round(100 * cutensor / ours; digits = 1), valid = outcome)
        println("   ", row)
        push!(rows, row)
    end
    remotecall_fetch(release_problem, w)
end

df = DataFrame(rows)
println()
show(stdout, MIME("text/plain"), sort(df, [:tc, order(:relative_pct, rev = true)]); allrows = true)
println()
open(OUT, "w") do io
    println(io, join(names(df), ","))
    foreach(r -> println(io, join(r, ",")), eachrow(df))
end
println("\nwritten to $OUT")
