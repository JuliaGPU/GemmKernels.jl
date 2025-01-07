#!/usr/bin/env bash
set -Eeuo pipefail

for i in {1..48}; do
    export GK_PROBLEM_ID=$i
    export CUDA_VISIBLE_DEVICES=0

    rm -rf ~/.julia/scratchspaces/*
    rm -f tuning/best-configs.bin

    LD_LIBRARY_PATH=$(julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))') ncu --profile-from-start off julia --project=tuning profile.jl
done
