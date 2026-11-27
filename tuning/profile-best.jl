# Profile the best configuration of some TCCG contractions, and cuTENSOR on the same data,
# with Nsight Compute: one launch each, inside an NVTX range "TC<n> ours" or "TC<n> cutensor".
#
# usage (on the GPU machine, with nothing else running on the GPU; Nsight Compute needs access
# to the GPU performance counters, i.e. root, unless the driver allows it for all users):
#
#   ncu --profile-from-start off --nvtx --set full --target-processes all -o profile-best \
#       julia +1.11 --project=tuning tuning/profile-best.jl <configs.arrow> [9,10,11,...]
#   ncu --import profile-best.ncu-rep --csv --page raw > profile-best.csv
#
# <configs.arrow> holds one configuration per contraction (e.g. the thesis's best ones), or a
# whole sweep: then the fastest successful configuration per contraction is profiled. The
# second argument selects contractions by their TCCG number (default: all 48). The script
# also prints, per contraction, which configuration it profiled and whether its result is
# correct; profile-best-manifest.csv keeps the same information.
#
# Nsight Compute locks the clocks to the base clock by default (--clock-control base), so
# the kernel durations it reports are not those of the tuner.

using CUDA, GemmKernels, DataFrames, Adapt
import Arrow
include(joinpath(@__DIR__, "wmma-contraction.jl"))

const NVTX = CUDA.NVTX

configs = DataFrame(Arrow.Table(ARGS[1]))
if "status" in names(configs)
    configs = filter(:status => ==("success"), configs)
end
selected = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ',')) : collect(1:length(jsonData))

function best_config(configs, name)
    rows = filter(:name => ==(name), configs)
    nrow(rows) == 0 && return nothing
    return NamedTuple(rows[argmin(rows.time), :])
end

function profile_tc(tc, problem, config)
    data = allocate_data(problem)
    initialize_data(problem, data...)
    reference = Array(calculate_reference(problem, data...))

    baseline_args = prepare_baseline(problem, data...)
    params = create_params(config)
    args = try
        prepare(problem, data...; params...)
    finally
        GemmKernels.Tensors.OVERRIDE_do_override = false
    end
    initialize_data(problem, data...; params...)

    # compile and warm up both outside the profiled ranges, and verify ours
    valid = verify(problem, reference, adapt(Array, execute(problem, data...; args...)))
    execute_baseline(problem, data...; baseline_args...)
    synchronize()

    CUDA.@profile external=true NVTX.@range "TC$tc ours" execute(problem, data...; args...)
    CUDA.@profile external=true NVTX.@range "TC$tc cutensor" execute_baseline(problem, data...; baseline_args...)

    foreach(CUDA.unsafe_free!, data)
    return valid
end

manifest = []
for (tc, problem) in enumerate(generate_problems())
    tc in selected || continue
    config = best_config(configs, problem.name)
    if config === nothing
        @warn "TC$tc ($(problem.name)): no configuration in $(ARGS[1]); skipped"
        continue
    end
    valid = profile_tc(tc, problem, config)
    println("TC$tc: $(repr_row(config)), $(round(config.time * 1e6; digits=1)) μs in the tuner, valid: $valid")
    push!(manifest, (; tc, name = problem.name, config = repr_row(config),
                       tuner_us = round(config.time * 1e6; digits = 1), valid))
    GC.gc(true); CUDA.reclaim()
end

open("profile-best-manifest.csv", "w") do io
    println(io, "tc,name,config,tuner_us,valid")
    for m in manifest
        println(io, join((m.tc, m.name, "\"$(m.config)\"", m.tuner_us, m.valid), ","))
    end
end
