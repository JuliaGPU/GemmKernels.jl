#!/usr/bin/env bash
set -Eeuo pipefail

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -f configs.bson
rm -f tuning.log

cd ..

until julia --project -e '
    println("--- :julia: Instantiating project")
    using Pkg
    Pkg.instantiate()
    Pkg.activate("tuning")
    Pkg.instantiate()
    push!(LOAD_PATH, @__DIR__)

    println("+++ :julia: Tuning")
    include("tuning/tune-wmma.jl")'; do

    echo "Tuning script crashed. Resuming after 1 second..." >&2
    sleep 1
done
