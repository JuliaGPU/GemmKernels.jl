using Distributed
using CUDA

# Get the device we're working on
cuda_dev = device()
mig = uuid(cuda_dev) != parent_uuid(cuda_dev)
nvml_dev = NVML.Device(uuid(cuda_dev); mig)

# GPU memory per problem.
gpu_memory_per_problem = 201326592 + # maximum(map(c -> sizeof(c[1]), get_configs()))
                         1000*2^20 + # size of (reasonable) CUDA context
                         32*2^20     # allow minimal unaccounted allocations

# determine parallelism
cpu_jobs = Sys.CPU_THREADS
memory_jobs = Int(Sys.free_memory()) ÷ (2 * 2^30)
gpumem_jobs = Int((0.9 * NVML.memory_info(nvml_dev).free) ÷ gpu_memory_per_problem)

@info "CPU job limit: $cpu_jobs"
@info "RAM job limit: $memory_jobs"
@info "VRAM job limit: $gpumem_jobs"

jobs = min(cpu_jobs, memory_jobs, gpumem_jobs)
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
runtests("tests.jl", ARGS...)
