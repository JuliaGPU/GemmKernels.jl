#!/usr/bin/env bash
set -Eeuo pipefail

rm -f profile.*.ncu-rep

for i in {1..48}; do
    export GK_PROBLEM_ID=$i
    export CUDA_VISIBLE_DEVICES=0

    rm -rf ~/.julia/scratchspaces/*
    rm -f tuning/best-configs.bin

    LD_LIBRARY_PATH=$(julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))') ncu --set full --profile-from-start off -o profile.$i julia --project=tuning profile.jl
done
