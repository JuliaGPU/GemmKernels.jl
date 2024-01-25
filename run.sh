#!/usr/bin/env bash
set -Eeuo pipefail

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile();'
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate(); Pkg.precompile();'

julia -p 48 --project=benchmarks -e '
    @everywhere push!(LOAD_PATH, @__DIR__)
    @everywhere include("benchmarks/runbenchmarks.jl")'
