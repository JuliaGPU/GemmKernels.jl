# Check precheck_config against the outcomes recorded in a configs.arrow: a configuration it
# rejects must not have compiled. Needs no GPU.
#
# usage (from the repository root):
#   GEMMKERNELS_TUNE_ASYNC=0 julia +1.11 --project=tuning tuning/check-precheck.jl <configs.arrow> [max_shmem]
#
# max_shmem defaults to 101376 bytes (99 KiB, sm_89).

using CUDA, GemmKernels, DataFrames
import Arrow
include(joinpath(@__DIR__, "wmma-contraction.jl"))

const max_shmem = parse(Int, get(ARGS, 2, "101376"))
df = DataFrame(Arrow.Table(ARGS[1]))
verdict = [precheck_config(NamedTuple(row); max_shmem) for row in eachrow(df)]

rejected = verdict .== "config_error_during_prepare"
predicted_compile_error = verdict .== "compilation_error"
status = df.status
println("$(nrow(df)) configurations")
println("rejected by the pre-check: $(count(rejected))")
println("  of which recorded as config_error_during_prepare: $(count(rejected .& (status .== "config_error_during_prepare")))")
wrong = rejected .& .!in.(status, Ref(["config_error_during_prepare", "pending"]))
println("  of which recorded otherwise (must be 0): $(count(wrong))")
count(wrong) > 0 && show(stdout, MIME("text/plain"), first(df[wrong, :], 10)); println()
println("predicted compilation errors: $(count(predicted_compile_error))")
wrong_ce = predicted_compile_error .& .!in.(status, Ref(["compilation_error", "config_error_during_prepare", "pending"]))
println("  of which compiled after all (must be 0): $(count(wrong_ce))")
ce = status .== "config_error_during_prepare"
println("config errors caught: $(count(rejected .& ce)) of $(count(ce))")
