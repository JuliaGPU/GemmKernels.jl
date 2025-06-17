#!/usr/bin/env bash
set -Eeuo pipefail

for tc in $(seq 1 48); do
    echo "Processing TC $tc/48..."
    /usr/local/cuda-12.8/bin/ncu --set=full --clock-control=none --nvtx -o prof-$tc ./benchmark-operator-fusion.sh $tc
done
