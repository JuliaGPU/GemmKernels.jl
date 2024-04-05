#!/usr/bin/env bash
set -Eeuo pipefail

rm prof*.ncu-rep

for TC in 1.2-1.3-3.2 1.2.3.4-5.3-1.2.5.4; do
    for impl in cutensor gemmkernels-optimal-for-jupiter gemmkernels-optimal-for-ripper; do
        /usr/local/cuda-12.4/bin/ncu -f --profile-from-start off -o prof-host-$(hostname)-tc-$TC-impl-$impl julia --project=tuning ./profile.jl $TC $impl
    done
done
