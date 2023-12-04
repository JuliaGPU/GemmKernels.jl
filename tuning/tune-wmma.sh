#!/usr/bin/env bash
set -Eeuo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )"

cd ..

echo "+++ :julia: Instantiating project"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=tuning -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

until julia --project=tuning -e '
    using CUDA, Distributed

    # determine how many workers to use
    memory_usage = 5*2^30
    cpu_memory = Sys.free_memory()
    gpu_memory = CUDA.available_memory()
    workers = min(
        floor(Int, cpu_memory / memory_usage),
        floor(Int, gpu_memory / memory_usage),
        Sys.CPU_THREADS
    )
    println("+++ :julia: Tuning using $workers workers")

    # launch workers
    using Distributed
    env = [
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => string(memory_usage),
    ]
    exeflags = [
        "--project=$(Base.active_project())",
        "--heap-size-hint=$memory_usage"
    ]
    addprocs(workers; exeflags, env)

    using Distributed
    @everywhere push!(LOAD_PATH, @__DIR__)
    @everywhere include("tuning/tune-wmma.jl")' "$@"; do

    echo "Tuning script crashed. Resuming after 1 second..." >&2
    sleep 1
done
