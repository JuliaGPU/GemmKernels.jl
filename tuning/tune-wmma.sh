#!/usr/bin/env bash
set -Eeuo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )"

cd ..

echo "+++ :julia: Instantiating project"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=tuning -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "+++ :julia: Tuning"
until julia --project=tuning -e '
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "4GiB"
    using Distributed
    @everywhere push!(LOAD_PATH, @__DIR__)
    @everywhere include("tuning/tune-wmma.jl")' "$@"; do

    echo "Tuning script crashed. Resuming after 1 second..." >&2
    sleep 1
done
