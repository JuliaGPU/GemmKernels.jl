#!/usr/bin/env bash
set -Eeuo pipefail

for conf in a b c d e; do
    /usr/local/cuda-12.8/bin/ncu --set=full --clock-control=none --nvtx -o prof-$conf ./benchmark-operator-fusion.sh $conf
done
