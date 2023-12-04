#!/usr/bin/env bash
set -Eeuo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )"

cd ..

echo "+++ :julia: Instantiating project"
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project=tuning -e 'using Pkg; Pkg.instantiate()'

echo "+++ :julia: Tuning"
until julia --project=tuning -e '
    push!(LOAD_PATH, @__DIR__)
    include("tuning/tune-wmma.jl")'; do

    echo "Tuning script crashed. Resuming after 1 second..." >&2
    sleep 1
done
