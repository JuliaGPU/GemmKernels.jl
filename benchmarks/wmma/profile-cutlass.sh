#!/usr/bin/env bash
set -Eeuo pipefail

# CUTLASS profiler path must be set
if [[ -z ${CUTLASS_PROF_PATH+x} ]]; then
    echo "CUTLASS_PROF_PATH is not set." 1>&2
    echo "Use: export CUTLASS_PROF_PATH=/path/to/cutlass/build/tools/profiler/cutlass_profiler" 1>&2
    exit 1
fi

cd "$( dirname "${BASH_SOURCE[0]}" )"

for a_layout in n t; do
    for b_layout in n t; do
        OUTPUT_FILE="cutlass-${a_layout}${b_layout}.csv"
        KERNEL="cutlass_wmma_tensorop_s161616gemm_f16_128x128_32x2_${a_layout}${b_layout}_align8"

        printf "N,runtime\n" >$OUTPUT_FILE

        for i in {7..14}; do
            N=$((2**i))

            # runtime is in nanoseconds
            runtime=$(nv-nsight-cu-cli --profile-from-start on -f --csv --units base -k Kernel --metrics 'gpu__time_duration.avg' ${CUTLASS_PROF_PATH} --warmup-iterations=0 --profiling-iterations=10 --verification-enabled=false --kernels=${KERNEL} --m=$N --n=$N --k=$N 2>/dev/null | grep 'gpu__time_duration' | awk -F',' '{print $NF}' | sed 's/"//g' | paste -sd ',')

            printf "$N,\"$runtime\"\n" >>$OUTPUT_FILE
        done
    done
done
