#!/usr/bin/env bash
set -Eeuo pipefail

rm -f profile.*.ncu-rep

for kernel in gk cut; do
    for i in {1..48}; do
        export GK_KERNEL=$kernel
        export GK_PROBLEM_ID=$i
        export CUDA_VISIBLE_DEVICES=0

        rm -rf ~/.julia/scratchspaces/*

        LD_LIBRARY_PATH=$(julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))') ncu --set full --profile-from-start off -o profile.$kernel.$i julia --project=tuning profile.jl
    done
done
