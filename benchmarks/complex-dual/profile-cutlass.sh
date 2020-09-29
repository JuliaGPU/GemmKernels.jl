#!/usr/bin/env bash
set -Eeuo pipefail

# CUTLASS examples build path must be set
if [[ -z ${CUTLASS_EXAMPLES_BUILD_PATH+x} ]]; then
    echo "CUTLASS_EXAMPLES_BUILD_PATH is not set." 1>&2
    echo "Use: export CUTLASS_EXAMPLES_BUILD_PATH=/path/to/cutlass/build/examples" 1>&2
    exit 1
fi

cd "$( dirname "${BASH_SOURCE[0]}" )"

OUTPUT_FILE="cutlass.csv"

printf "N,runtime\n" >$OUTPUT_FILE

for i in {7..14}; do
    N=$((2**i))

    # runtime is in nanoseconds
    runtime=$(nv-nsight-cu-cli --profile-from-start on -f --csv --units base -k Kernel --metrics 'gpu__time_duration.avg' ${CUTLASS_EXAMPLES_BUILD_PATH}/10_planar_complex/10_planar_complex --m=$N --n=$N --k=$N --batch=1 2>/dev/null | grep 'gpu__time_duration' | awk -F',' '{print $NF}' | sed 's/"//g' | paste -sd ',')

    printf "$N,\"$runtime\"\n" >>$OUTPUT_FILE
done
