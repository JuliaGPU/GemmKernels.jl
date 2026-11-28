# Time hand-picked variants of the thesis's best configurations next to cuTENSOR, in one
# session: the tile shapes, warps, kernels and splits that cuTENSOR's own kernels use according
# to the Nsight Compute profiles (tuning/profile-best.jl), and tile heights that need no
# padding. Each variant keeps the thesis configuration's layout parameters (permutations,
# column-major flags, CTA swizzle) and overrides the rest. Every result is verified; rejected
# variants are reported with GemmKernels' reason, and a crashing variant restarts the worker.
#
# usage (from the repository root, on the GPU machine, with nothing else on the GPU):
#   julia +1.11 --project=tuning tuning/try-variants.jl <thesis configs.arrow> [TCs] [out.csv] [set]
#
# TCs defaults to 9,10,11,12,13,14,19,20,22,24. With set = indexing, the variants are instead
# the thesis configuration, the same with pipelined_async_3, and the best variant found so
# far for TC12/20/22/24, each with both address arithmetics of the GETT layouts
# (INDEXING = default, lean32). Like tune.jl, compiling and profiling happen
# in separate remote calls: compiling after CUDA.@profile within the same call fails with
# CUDA.jl 5.11.

using Distributed

const CONFIGS = ARGS[1]
const TCS = parse.(Int, split(get(ARGS, 2, "9,10,11,12,13,14,19,20,22,24"), ','))
const OUT = get(ARGS, 3, "try-variants.csv")
const SET = get(ARGS, 4, "shapes")
const SAMPLES = 10
const DIR = @__DIR__

# (label, overrides of the thesis configuration, number of split-K parts)
v(label, split_k = 1; o...) = (; label, overrides = (; o...), split_k)
const ASYNC = ("pipelined_async_3", "pipelined_async_4")
op16 = (OP_M = 16, OP_N = 16, OP_K = 16)
op8 = (OP_M = 8, OP_N = 32, OP_K = 16)

function variants(tc)
    list = Any[v("thesis"), v("thesis, async 3"; kernel_str = "pipelined_async_3")]
    if tc in (12, 20, 22, 24)
        # cuTENSOR: 128×128×32 tiles, 2×2 warps, cp.async; for TC12 also split-K 3
        for k in ("pipelined_async_2", ASYNC..., "singlestage")
            push!(list, v("cuTENSOR shape, $k"; BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32,
                          WARPS_M = 2, WARPS_N = 2, op16..., kernel_str = k))
        end
        for k in ASYNC
            push!(list, v("128×128×32, 2×4 warps, $k"; BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32,
                          WARPS_M = 2, WARPS_N = 4, op16..., kernel_str = k))
            push!(list, v("128×64×32, 2×2 warps, $k"; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32,
                          WARPS_M = 2, WARPS_N = 2, op16..., kernel_str = k))
        end
        if tc == 12
            for k in ASYNC, s in (2, 3, 4)
                push!(list, v("cuTENSOR shape, $k", s; BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32,
                              WARPS_M = 2, WARPS_N = 2, op16..., kernel_str = k))
            end
        end
    elseif tc in (13, 14)
        # cuTENSOR: 64×64×32 tiles, 2×1 warps, ~28 parts (700 blocks for 25 tiles)
        for k in (ASYNC..., "singlestage"), s in (8, 16, 32)
            push!(list, v("cuTENSOR shape, $k", s; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32,
                          WARPS_M = 2, WARPS_N = 1, op16..., kernel_str = k))
        end
        for s in (16, 32)
            push!(list, v("128×64×32, 8×1 warps, async 4", s; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32,
                          WARPS_M = 8, WARPS_N = 1, op8..., kernel_str = "pipelined_async_4"))
        end
    elseif tc == 19
        # cuTENSOR: 64×64×32 tiles, 2×1 warps, split-K 4 (648 blocks); N = 72, so a BLOCK_N of
        # 64 pads N to 128 and one of 32 to 96
        for k in ASYNC, s in (1, 2, 4, 8)
            push!(list, v("cuTENSOR shape, $k", s; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32,
                          WARPS_M = 2, WARPS_N = 1, op16..., kernel_str = k))
        end
        for s in (1, 2, 4)
            push!(list, v("64×32×32, 2×1 warps, async 3", s; BLOCK_M = 64, BLOCK_N = 32, BLOCK_K = 32,
                          WARPS_M = 2, WARPS_N = 1, op16..., kernel_str = "pipelined_async_3"))
        end
    elseif tc in (9, 10, 11)
        # cuTENSOR covers all of M = 72 with one block (128×128); we pad M to a multiple of
        # BLOCK_M (128: D writes 1.8× larger), unless BLOCK_M is 72 or 80
        for k in ("singlestage", "pipelined_async_3")
            push!(list, v("128×128×16, 2×2 warps, $k"; BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16,
                          WARPS_M = 2, WARPS_N = 2, op16..., kernel_str = k))
            push!(list, v("128×64×16, 2×2 warps, $k"; BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 16,
                          WARPS_M = 2, WARPS_N = 2, op16..., kernel_str = k))
            push!(list, v("80×64×32, 1×1 warps, $k"; BLOCK_M = 80, BLOCK_N = 64, BLOCK_K = 32,
                          WARPS_M = 1, WARPS_N = 1, op16..., kernel_str = k))
            push!(list, v("80×128×32, 1×2 warps, $k"; BLOCK_M = 80, BLOCK_N = 128, BLOCK_K = 32,
                          WARPS_M = 1, WARPS_N = 2, op16..., kernel_str = k))
            push!(list, v("72×64×32, 1×1 warps, $k"; BLOCK_M = 72, BLOCK_N = 64, BLOCK_K = 32,
                          WARPS_M = 1, WARPS_N = 1, op8..., kernel_str = k))
        end
    end
    return list
end

function indexing_variants(tc)
    bases = Any[v("thesis"), v("thesis, async 3"; kernel_str = "pipelined_async_3")]
    tc in (12, 20, 22, 24) &&
        push!(bases, v("128×128×32, 2×4 warps, async 4"; BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32,
                       WARPS_M = 2, WARPS_N = 4, op16..., kernel_str = "pipelined_async_4"))
    return Any[v("$(b.label), $ix", b.split_k; b.overrides..., INDEXING = ix)
               for b in bases for ix in ("default", "lean32")]
end

const WORKER_CODE = quote
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

    # compile and verify one variant: (:ok, valid), (:rejected, reason) or (:error, message)
    function prepare_variant(config, split_k)
        (; problem, data, reference) = STATE[]
        VARIANT[] = nothing
        params = create_params(merge(config, (; SPLIT_K = split_k)))
        args = try
            prepare(problem, data...; params...)
        catch err
            err isa GemmKernels.ConfigError && return (:rejected, sprint(showerror, err))
            return (:error, first(sprint(showerror, err), 300))
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
        return (:ok, valid)
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

# package loading and definitions in separate evaluations: macros such as CUDA.@profile
# must be defined before the code that uses them is parsed
function start_worker()
    w = addprocs(1; exeflags = "--project=$(Base.active_project())")[1]
    Distributed.remotecall_eval(Main, w, :(using CUDA, GemmKernels, DataFrames, Adapt; import Arrow))
    Distributed.remotecall_eval(Main, w, :(include($(joinpath(DIR, "wmma-contraction.jl")))))
    Distributed.remotecall_eval(Main, w, WORKER_CODE)
    return w
end

# the master needs the same definitions: remote calls refer to the functions by name
using CUDA, GemmKernels, DataFrames, Adapt
import Arrow
include(joinpath(DIR, "wmma-contraction.jl"))
eval(WORKER_CODE)

function main()
    w = start_worker()
    configs = DataFrame(Arrow.Table(CONFIGS))
    delete!(ENV, "GK_PROBLEMS")   # problems[tc] must be TCCG contraction tc
    problems = generate_problems()
    rows = []
    for tc in TCS
        problem = problems[tc]
        sel = filter(:name => ==(problem.name), configs)
        nrow(sel) == 1 || (@warn "TC$tc: $(nrow(sel)) configurations in $CONFIGS; skipped"; continue)
        thesis = NamedTuple(first(sel))
        println("== TC$tc $(problem.name): thesis configuration $(repr_row(thesis)), $(round(thesis.time * 1e6; digits=1)) μs in the thesis")

        remotecall_fetch(setup_problem, w, problem)
        cutensor = remotecall_fetch(measure_baseline, w, SAMPLES)
        println("   cuTENSOR: $(round(cutensor * 1e6; digits=1)) μs")

        for var in (SET == "indexing" ? indexing_variants(tc) : variants(tc))
            (; label, overrides, split_k) = var
            config = merge(thesis, overrides)
            desc = repr_row(merge(config, (; SPLIT_K = split_k)))
            row = (; tc, label, split_k, config = desc, ours_us = NaN, cutensor_us = round(cutensor * 1e6; digits = 1),
                     relative_pct = NaN, valid = false, outcome = "")
            try
                status, info = remotecall_fetch(prepare_variant, w, config, split_k)
                if status == :ok
                    ours = remotecall_fetch(measure_variant, w, SAMPLES)
                    row = merge(row, (; ours_us = round(ours * 1e6; digits = 1),
                                        relative_pct = round(100 * cutensor / ours; digits = 1),
                                        valid = info, outcome = info ? "ok" : "invalid result"))
                else
                    row = merge(row, (; outcome = "$status: $(replace(info, r"\s+" => " "))"))
                end
            catch err
                # a crash (e.g. an illegal memory access) leaves the CUDA context unusable
                row = merge(row, (; outcome = "crashed: $(first(replace(sprint(showerror, err), r"\s+" => " "), 200))"))
                try rmprocs(w; waitfor = 30) catch end
                w = start_worker()
                remotecall_fetch(setup_problem, w, problem)
            end
            println("   $(rpad(label * (split_k > 1 ? ", split-K $split_k" : ""), 48)) ",
                    isnan(row.ours_us) ? row.outcome : "$(row.ours_us) μs ($(row.relative_pct)% of cuTENSOR's speed)$(row.valid ? "" : ", INVALID")")
            push!(rows, row)
        end
        remotecall_fetch(release_problem, w)
    end
    return DataFrame(rows)
end

df = main()
println("\nBest valid variant per contraction:")
for g in groupby(filter(r -> r.valid, df), :tc)
    best = g[argmin(g.ours_us), :]
    println("  TC$(best.tc): $(best.config): $(best.ours_us) μs, $(best.relative_pct)% of cuTENSOR's speed")
end
csvfield(x) = x isa AbstractString ? "\"" * replace(x, "\"" => "\"\"") * "\"" : string(x)
open(OUT, "w") do io
    println(io, join(names(df), ","))
    foreach(r -> println(io, join(csvfield.(collect(r)), ",")), eachrow(df))
end
println("\nwritten to $OUT")
