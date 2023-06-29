
using Distributed

# determine parallelism
cpu_jobs = Sys.CPU_THREADS
memory_jobs = Int(Sys.free_memory()) ÷ (2 * 2^30)
jobs = min(cpu_jobs, memory_jobs)
@info "Running $jobs tests in parallel. If this is too many, set the `JULIA_CPU_THREADS` environment variable."

# add workers
exeflags = Base.julia_cmd()
filter!(exeflags.exec) do c
    return !(startswith(c, "--depwarn") || startswith(c, "--check-bounds"))
end
push!(exeflags.exec, "--check-bounds=yes")
push!(exeflags.exec, "--startup-file=no")
push!(exeflags.exec, "--depwarn=yes")
push!(exeflags.exec, "--project=$(Base.active_project())")
exename = popfirst!(exeflags.exec)
withenv("JULIA_NUM_THREADS" => 1, "OPENBLAS_NUM_THREADS" => 1) do
    addprocs(jobs; exename, exeflags)
end

@everywhere using XUnit
runtests("tests.jl")
